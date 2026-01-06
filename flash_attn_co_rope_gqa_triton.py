"""
Fused CoRoPE Attention (GQA Skeleton)
=====================================

This module carries the Triton kernel backbone for the CoRoPE-GQA fused
attention path.  Step 1 focuses on locking down the CTA scheduling,
per-group Q loading, and the public Python interface.

Interface: attention(q, k, v, causal, sm_scale, theta)
- theta: RoPE base frequency (e.g., 10000.0), same as PyTorch version
- inv_freq is computed internally from theta
"""

import pytest
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64  # reserved for the streaming K/V sweep in later steps


@triton.jit
def _corope_fwd_backbone(
    Q, K, V, O,
    inv_freq_ptr,
    sm_scale,
    Z, H_Q, H_KV, group_size, N_CTX,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_vz, stride_vh, stride_vm, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_inv,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_g = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_CTX

    off_z = off_g // H_KV
    off_kv = off_g % H_KV

    head_base = off_kv * GROUP_SIZE

    half_dim: tl.constexpr = HEAD_DIM // 2
    offs_d_first = tl.arange(0, half_dim)
    offs_d_second = offs_d_first + half_dim

    col_mask = tl.arange(0, half_dim)[None, :] < half_dim
    mask_q = mask_m[:, None] & col_mask

    inv_idx = tl.arange(0, half_dim)
    inv_freq = tl.load(inv_freq_ptr + inv_idx * stride_inv, mask=inv_idx < half_dim, other=0.0).to(tl.float32)

    # 领航员预计算：独立加载 Leader (第一个 head)
    leader_head_idx = head_base
    leader_q_base = Q + off_z * stride_qz + leader_head_idx * stride_qh
    leader_q_base = tl.multiple_of(leader_q_base, 16)
    
    leader_q1_ptrs = leader_q_base + offs_m[:, None] * stride_qm + offs_d_first[None, :] * stride_qk
    leader_q2_ptrs = leader_q_base + offs_m[:, None] * stride_qm + offs_d_second[None, :] * stride_qk
    
    leader_q1 = tl.load(leader_q1_ptrs, mask=mask_q, other=0.0).to(tl.float32)
    leader_q2 = tl.load(leader_q2_ptrs, mask=mask_q, other=0.0).to(tl.float32)

    half_dim_range = tl.arange(0, half_dim)
    km_off = half_dim_range
    K_base = K + off_z * stride_kz + off_kv * stride_kh
    K_base = tl.multiple_of(K_base, 16)
    V_base = V + off_z * stride_vz + off_kv * stride_vh
    V_base = tl.multiple_of(V_base, 16)

    # Stage-1: 向量化计算全局里程 a_tt
    # 使用 leader Q 扫描所有 K，计算每个 Q token 的最终里程值
    a_tt = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX

        # 加载 K tile: (BLOCK_N, half_dim)
        k1_tile = tl.load(
            K_base + offs_n[:, None] * stride_km + km_off[None, :] * stride_kk,
            mask=mask_n[:, None] & col_mask,
            other=0.0,
        ).to(tl.float32)
        k2_tile = tl.load(
            K_base + offs_n[:, None] * stride_km + (km_off + half_dim)[None, :] * stride_kk,
            mask=mask_n[:, None] & col_mask,
            other=0.0,
        ).to(tl.float32)

        # 向量化计算能量矩阵: (BLOCK_M, BLOCK_N)
        # E_A = Q1 @ K1^T + Q2 @ K2^T
        ea_tile = tl.dot(leader_q1, tl.trans(k1_tile)) + tl.dot(leader_q2, tl.trans(k2_tile))
        ea_tile = ea_tile * sm_scale
        
        # Sigmoid: z = 1 / (1 + exp(-ea))
        z_tile = 1.0 / (1.0 + tl.exp(-ea_tile))
        
        # Causal mask: offs_m[:, None] >= offs_n[None, :]
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        z_tile = tl.where(valid_mask, z_tile, 0.0)
        
        # 行求和更新全局里程: a_tt += sum(z_tile, axis=1)
        a_tt += tl.sum(z_tile, axis=1)

    # Stage-2: 为每个 group 独立分配状态变量
    # 关键：不用大向量索引，而是用独立变量 + 条件更新
    
    # 为最多 8 个 group 预分配状态（GROUP_SIZE <= 8）
    acc_first_0 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    acc_second_0 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    m_0 = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_0 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    acc_first_1 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    acc_second_1 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    m_1 = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_1 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    acc_first_2 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    acc_second_2 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    m_2 = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_2 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    acc_first_3 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    acc_second_3 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    m_3 = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_3 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    acc_first_4 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    acc_second_4 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    m_4 = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_4 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    acc_first_5 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    acc_second_5 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    m_5 = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_5 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    acc_first_6 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    acc_second_6 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    m_6 = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_6 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    acc_first_7 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    acc_second_7 = tl.zeros([BLOCK_M, half_dim], dtype=tl.float32)
    m_7 = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_7 = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Leader 里程累积
    acc_z = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX

        # 加载共享的 K, V
        k1_tile = tl.load(
            K_base + offs_n[:, None] * stride_km + km_off[None, :] * stride_kk,
            mask=mask_n[:, None] & col_mask,
            other=0.0,
        ).to(tl.float32)
        k2_tile = tl.load(
            K_base + offs_n[:, None] * stride_km + (km_off + half_dim)[None, :] * stride_kk,
            mask=mask_n[:, None] & col_mask,
            other=0.0,
        ).to(tl.float32)

        v1_tile = tl.load(
            V_base + offs_n[:, None] * stride_vm + offs_d_first[None, :] * stride_vk,
            mask=mask_n[:, None] & col_mask,
            other=0.0,
        ).to(tl.float32)
        v2_tile = tl.load(
            V_base + offs_n[:, None] * stride_vm + offs_d_second[None, :] * stride_vk,
            mask=mask_n[:, None] & col_mask,
            other=0.0,
        ).to(tl.float32)

        # Leader 里程计算
        ea_leader = tl.dot(leader_q1, tl.trans(k1_tile)) + tl.dot(leader_q2, tl.trans(k2_tile))
        ea_leader = ea_leader * sm_scale
        z_tile = 1.0 / (1.0 + tl.exp(-ea_leader))
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        z_tile = tl.where(valid_mask, z_tile, 0.0)
        
        # 块内动态里程
        z_cumsum = tl.cumsum(z_tile, axis=1)
        a_block_tile = acc_z[:, None] + z_cumsum
        delta_tile = a_tt[:, None] - a_block_tile
        
        # 静态循环展开：为每个 group 成员计算 attention
        for g in tl.static_range(GROUP_SIZE):
            head_idx = head_base + g
            head_mask = head_idx < H_Q
            
            # 加载当前 group 的 Q
            q_head_base = Q + off_z * stride_qz + head_idx * stride_qh
            q_head_base = tl.multiple_of(q_head_base, 16)
            
            q1_ptrs = q_head_base + offs_m[:, None] * stride_qm + offs_d_first[None, :] * stride_qk
            q2_ptrs = q_head_base + offs_m[:, None] * stride_qm + offs_d_second[None, :] * stride_qk
            
            q1 = tl.load(q1_ptrs, mask=mask_q & head_mask, other=0.0).to(tl.float32)
            q2 = tl.load(q2_ptrs, mask=mask_q & head_mask, other=0.0).to(tl.float32)
            
            # Co-RoPE 相位校准：使用 3D broadcasting
            # 相位: phi = delta[:, :, None] * inv_freq[None, None, :]
            phi = delta_tile[:, :, None] * inv_freq[None, None, :]  # (BLOCK_M, BLOCK_N, half_dim)
            cos_phi = tl.cos(phi)
            sin_phi = tl.sin(phi)
            
            # 能量矩阵: (BLOCK_M, BLOCK_N, half_dim)
            ea = q1[:, None, :] * k1_tile[None, :, :] + q2[:, None, :] * k2_tile[None, :, :]
            eb = q2[:, None, :] * k1_tile[None, :, :] - q1[:, None, :] * k2_tile[None, :, :]
            
            # 相位校准后求和: (BLOCK_M, BLOCK_N)
            score = tl.sum(ea * cos_phi - eb * sin_phi, axis=2)
            
            # 应用 scale 和 mask
            score = score * sm_scale
            score = tl.where(valid_mask & head_mask, score, -float('inf'))
            
            # Online softmax - 根据 g 值更新对应的状态变量
            m_curr = tl.max(score, axis=1)
            
            # 展开：为每个 g 值分别处理
            if g == 0:
                m_new = tl.maximum(m_0, m_curr)
                alpha = tl.exp(m_0 - m_new)
                p = tl.exp(score - m_new[:, None])
                p = tl.where(valid_mask & head_mask, p, 0.0)
                l_0 = l_0 * alpha + tl.sum(p, axis=1)
                acc_first_0 = acc_first_0 * alpha[:, None] + tl.dot(p, v1_tile)
                acc_second_0 = acc_second_0 * alpha[:, None] + tl.dot(p, v2_tile)
                m_0 = m_new
            if g == 1:
                m_new = tl.maximum(m_1, m_curr)
                alpha = tl.exp(m_1 - m_new)
                p = tl.exp(score - m_new[:, None])
                p = tl.where(valid_mask & head_mask, p, 0.0)
                l_1 = l_1 * alpha + tl.sum(p, axis=1)
                acc_first_1 = acc_first_1 * alpha[:, None] + tl.dot(p, v1_tile)
                acc_second_1 = acc_second_1 * alpha[:, None] + tl.dot(p, v2_tile)
                m_1 = m_new
            if g == 2:
                m_new = tl.maximum(m_2, m_curr)
                alpha = tl.exp(m_2 - m_new)
                p = tl.exp(score - m_new[:, None])
                p = tl.where(valid_mask & head_mask, p, 0.0)
                l_2 = l_2 * alpha + tl.sum(p, axis=1)
                acc_first_2 = acc_first_2 * alpha[:, None] + tl.dot(p, v1_tile)
                acc_second_2 = acc_second_2 * alpha[:, None] + tl.dot(p, v2_tile)
                m_2 = m_new
            if g == 3:
                m_new = tl.maximum(m_3, m_curr)
                alpha = tl.exp(m_3 - m_new)
                p = tl.exp(score - m_new[:, None])
                p = tl.where(valid_mask & head_mask, p, 0.0)
                l_3 = l_3 * alpha + tl.sum(p, axis=1)
                acc_first_3 = acc_first_3 * alpha[:, None] + tl.dot(p, v1_tile)
                acc_second_3 = acc_second_3 * alpha[:, None] + tl.dot(p, v2_tile)
                m_3 = m_new
            if g == 4:
                m_new = tl.maximum(m_4, m_curr)
                alpha = tl.exp(m_4 - m_new)
                p = tl.exp(score - m_new[:, None])
                p = tl.where(valid_mask & head_mask, p, 0.0)
                l_4 = l_4 * alpha + tl.sum(p, axis=1)
                acc_first_4 = acc_first_4 * alpha[:, None] + tl.dot(p, v1_tile)
                acc_second_4 = acc_second_4 * alpha[:, None] + tl.dot(p, v2_tile)
                m_4 = m_new
            if g == 5:
                m_new = tl.maximum(m_5, m_curr)
                alpha = tl.exp(m_5 - m_new)
                p = tl.exp(score - m_new[:, None])
                p = tl.where(valid_mask & head_mask, p, 0.0)
                l_5 = l_5 * alpha + tl.sum(p, axis=1)
                acc_first_5 = acc_first_5 * alpha[:, None] + tl.dot(p, v1_tile)
                acc_second_5 = acc_second_5 * alpha[:, None] + tl.dot(p, v2_tile)
                m_5 = m_new
            if g == 6:
                m_new = tl.maximum(m_6, m_curr)
                alpha = tl.exp(m_6 - m_new)
                p = tl.exp(score - m_new[:, None])
                p = tl.where(valid_mask & head_mask, p, 0.0)
                l_6 = l_6 * alpha + tl.sum(p, axis=1)
                acc_first_6 = acc_first_6 * alpha[:, None] + tl.dot(p, v1_tile)
                acc_second_6 = acc_second_6 * alpha[:, None] + tl.dot(p, v2_tile)
                m_6 = m_new
            if g == 7:
                m_new = tl.maximum(m_7, m_curr)
                alpha = tl.exp(m_7 - m_new)
                p = tl.exp(score - m_new[:, None])
                p = tl.where(valid_mask & head_mask, p, 0.0)
                l_7 = l_7 * alpha + tl.sum(p, axis=1)
                acc_first_7 = acc_first_7 * alpha[:, None] + tl.dot(p, v1_tile)
                acc_second_7 = acc_second_7 * alpha[:, None] + tl.dot(p, v2_tile)
                m_7 = m_new
        
        acc_z += tl.sum(z_tile, axis=1)

    # 归一化并写回：为每个 group 独立处理
    for g in tl.static_range(GROUP_SIZE):
        head_idx = head_base + g
        head_mask_g = head_idx < H_Q
        
        # 根据 g 选择对应的状态变量
        if g == 0:
            safe_l = tl.maximum(l_0, 1e-9)
            inv_l = 1.0 / safe_l
            inv_l = tl.where(l_0 > 0.0, inv_l, 0.0)
            out_first = acc_first_0 * inv_l[:, None]
            out_second = acc_second_0 * inv_l[:, None]
        if g == 1:
            safe_l = tl.maximum(l_1, 1e-9)
            inv_l = 1.0 / safe_l
            inv_l = tl.where(l_1 > 0.0, inv_l, 0.0)
            out_first = acc_first_1 * inv_l[:, None]
            out_second = acc_second_1 * inv_l[:, None]
        if g == 2:
            safe_l = tl.maximum(l_2, 1e-9)
            inv_l = 1.0 / safe_l
            inv_l = tl.where(l_2 > 0.0, inv_l, 0.0)
            out_first = acc_first_2 * inv_l[:, None]
            out_second = acc_second_2 * inv_l[:, None]
        if g == 3:
            safe_l = tl.maximum(l_3, 1e-9)
            inv_l = 1.0 / safe_l
            inv_l = tl.where(l_3 > 0.0, inv_l, 0.0)
            out_first = acc_first_3 * inv_l[:, None]
            out_second = acc_second_3 * inv_l[:, None]
        if g == 4:
            safe_l = tl.maximum(l_4, 1e-9)
            inv_l = 1.0 / safe_l
            inv_l = tl.where(l_4 > 0.0, inv_l, 0.0)
            out_first = acc_first_4 * inv_l[:, None]
            out_second = acc_second_4 * inv_l[:, None]
        if g == 5:
            safe_l = tl.maximum(l_5, 1e-9)
            inv_l = 1.0 / safe_l
            inv_l = tl.where(l_5 > 0.0, inv_l, 0.0)
            out_first = acc_first_5 * inv_l[:, None]
            out_second = acc_second_5 * inv_l[:, None]
        if g == 6:
            safe_l = tl.maximum(l_6, 1e-9)
            inv_l = 1.0 / safe_l
            inv_l = tl.where(l_6 > 0.0, inv_l, 0.0)
            out_first = acc_first_6 * inv_l[:, None]
            out_second = acc_second_6 * inv_l[:, None]
        if g == 7:
            safe_l = tl.maximum(l_7, 1e-9)
            inv_l = 1.0 / safe_l
            inv_l = tl.where(l_7 > 0.0, inv_l, 0.0)
            out_first = acc_first_7 * inv_l[:, None]
            out_second = acc_second_7 * inv_l[:, None]
        
        # 构建输出指针并存储
        o_head_base = O + off_z * stride_oz + head_idx * stride_oh
        o_head_base = tl.multiple_of(o_head_base, 16)
        
        o_half0 = o_head_base + offs_m[:, None] * stride_om + offs_d_first[None, :] * stride_ok
        o_half1 = o_head_base + offs_m[:, None] * stride_om + offs_d_second[None, :] * stride_ok
        
        # 存储（用 mask 控制是否有效）
        tl.store(o_half0, out_first.to(tl.float16), mask=mask_q & head_mask_g)
        tl.store(o_half1, out_second.to(tl.float16), mask=mask_q & head_mask_g)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, theta, warp_specialize=False):
        if not causal:
            raise ValueError("CoRoPE fused kernel supports causal=True only.")

        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("Expected q, k, v to have shape (batch, heads, seqlen, dim).")

        if k.shape != v.shape:
            raise ValueError("k and v must share the same shape in GQA.")
        if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
            raise ValueError("k must align with q along batch/sequence/head_dim.")

        BATCH, H_Q, N_CTX, HEAD_DIM = q.shape
        device = q.device
        H_KV = k.shape[1]
        if H_Q % H_KV != 0:
            raise ValueError("Number of query heads must be divisible by KV heads.")
        group_size = H_Q // H_KV
        if group_size > 8:
            raise ValueError(f"CoRoPE backbone currently limits group_size <= 8 (got {group_size}).")

        # Compute RoPE frequencies dynamically (same as PyTorch version)
        inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))

        o = torch.empty_like(q)

        grid = (triton.cdiv(N_CTX, DEFAULT_BLOCK_M), BATCH * H_KV)

        _corope_fwd_backbone[grid](
            q, k, v, o,
            inv_freq,
            sm_scale,
            BATCH, H_Q, H_KV, group_size, N_CTX,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            inv_freq.stride(0),
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=DEFAULT_BLOCK_M,
            BLOCK_N=DEFAULT_BLOCK_N,
            GROUP_SIZE=group_size,
            num_warps=4,
            num_stages=1,
        )

        ctx.save_for_backward(q, k, v, inv_freq)
        ctx.group_size = group_size
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, inv_freq = ctx.saved_tensors
        group_size = ctx.group_size
        sm_scale = ctx.sm_scale

        grad_output = grad_output.contiguous()
        q_shape = q.shape
        BATCH, H_Q, N_CTX, HEAD_DIM = q_shape
        H_KV = k.shape[1]
        device = q.device

        half_dim = HEAD_DIM // 2

        q_fp32 = q.to(torch.float32)
        k_fp32 = k.to(torch.float32)
        v_fp32 = v.to(torch.float32)
        go_fp32 = grad_output.to(torch.float32)
        inv_freq_fp32 = inv_freq.to(torch.float32)

        leader_indices = torch.arange(0, H_Q, group_size, device=device)

        # Expand K/V to match Q heads (GQA expansion)
        k_expanded = k_fp32.view(BATCH, H_KV, 1, N_CTX, HEAD_DIM).expand(-1, -1, group_size, -1, -1)
        k_expanded = k_expanded.reshape(BATCH, H_Q, N_CTX, HEAD_DIM)
        v_expanded = v_fp32.view(BATCH, H_KV, 1, N_CTX, HEAD_DIM).expand(-1, -1, group_size, -1, -1)
        v_expanded = v_expanded.reshape(BATCH, H_Q, N_CTX, HEAD_DIM)

        q_leaders = q_fp32[:, leader_indices, :, :]
        k_leaders = k_fp32

        # Leader odometry recomputation
        dot_leader = torch.einsum("bhid,bhjd->bhij", q_leaders, k_leaders)
        dot_scaled = dot_leader * sm_scale
        z_raw = torch.sigmoid(dot_scaled)
        causal_tri = torch.tril(torch.ones((N_CTX, N_CTX), device=device, dtype=torch.float32))
        causal_mask_bool = causal_tri.bool()
        z_leader = z_raw * causal_tri
        a_leader = torch.cumsum(z_leader, dim=-1)
        a = a_leader.repeat_interleave(group_size, dim=1)
        a_tt = torch.diagonal(a, dim1=-2, dim2=-1)
        delta = a_tt.unsqueeze(-1) - a

        inv = inv_freq_fp32.view(1, 1, 1, 1, -1)
        phi = delta.unsqueeze(-1) * inv
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        q1 = q_fp32[..., :half_dim]
        q2 = q_fp32[..., half_dim:]
        k1 = k_expanded[..., :half_dim]
        k2 = k_expanded[..., half_dim:]

        E_A = q1.unsqueeze(-2) * k1.unsqueeze(-3) + q2.unsqueeze(-2) * k2.unsqueeze(-3)
        E_B = q2.unsqueeze(-2) * k1.unsqueeze(-3) - q1.unsqueeze(-2) * k2.unsqueeze(-3)

        base = (E_A * cos_phi - E_B * sin_phi).sum(dim=-1)
        score = base * sm_scale
        upper_mask = torch.triu(torch.ones((N_CTX, N_CTX), device=device, dtype=torch.bool), diagonal=1)
        score = score.masked_fill(upper_mask, float("-inf"))
        attn = torch.softmax(score, dim=-1)
        attn = attn.masked_fill(upper_mask, 0.0)

        # Gradients w.r.t V
        dv_expanded = torch.einsum("bhij,bhid->bhjd", attn, go_fp32)
        dv = dv_expanded.view(BATCH, H_KV, group_size, N_CTX, HEAD_DIM).sum(dim=2)

        # Gradients w.r.t attention scores
        datt = torch.einsum("bhid,bhjd->bhij", go_fp32, v_expanded)
        sum_term = (datt * attn).sum(dim=-1, keepdim=True)
        dscore = attn * (datt - sum_term)
        dscore = dscore.masked_fill(upper_mask, 0.0)

        dE_A = dscore.unsqueeze(-1) * (sm_scale * cos_phi)
        dE_B = dscore.unsqueeze(-1) * (-sm_scale * sin_phi)
        dcos = dscore.unsqueeze(-1) * (sm_scale * E_A)
        dsin = dscore.unsqueeze(-1) * (-sm_scale * E_B)

        dphi = -sin_phi * dcos + cos_phi * dsin

        d_delta = (dphi * inv).sum(dim=-1)

        # Gradients from trigonometric expansion into Q/K
        dq1 = (dE_A * k1.unsqueeze(-3)).sum(dim=-2) - (dE_B * k2.unsqueeze(-3)).sum(dim=-2)
        dq2 = (dE_A * k2.unsqueeze(-3)).sum(dim=-2) + (dE_B * k1.unsqueeze(-3)).sum(dim=-2)
        dk1 = (dE_A * q1.unsqueeze(-2)).sum(dim=-3) + (dE_B * q2.unsqueeze(-2)).sum(dim=-3)
        dk2 = (dE_A * q2.unsqueeze(-2)).sum(dim=-3) - (dE_B * q1.unsqueeze(-2)).sum(dim=-3)

        dq_scores = torch.cat([dq1, dq2], dim=-1)
        dk_scores_expanded = torch.cat([dk1, dk2], dim=-1)
        dk_scores = dk_scores_expanded.view(BATCH, H_KV, group_size, N_CTX, HEAD_DIM).sum(dim=2)

        # Reverse-scan gradient for odometry
        d_a = -d_delta
        d_a_tt = d_delta.sum(dim=-1, keepdim=True)
        d_a = d_a + torch.diag_embed(d_a_tt.squeeze(-1), dim1=-2, dim2=-1)
        d_a = d_a.contiguous()
        d_a_leader = d_a.view(BATCH, H_KV, group_size, N_CTX, N_CTX).sum(dim=2)
        dz_leader = torch.flip(torch.cumsum(torch.flip(d_a_leader, dims=[-1]), dim=-1), dims=[-1])
        dz_leader = dz_leader * causal_tri

        sigmoid_prime = z_raw * (1.0 - z_raw)
        dot_grad = dz_leader * sigmoid_prime * sm_scale
        dq_leader_from_sigmoid = torch.einsum("bhij,bhjd->bhid", dot_grad, k_leaders)
        dk_from_sigmoid = torch.einsum("bhij,bhid->bhjd", dot_grad, q_leaders)

        dk_total = dk_scores + dk_from_sigmoid
        dq_total = dq_scores.clone()
        dq_total[:, leader_indices, :, :] += dq_leader_from_sigmoid

        # Gradients w.r.t sm_scale
        dsm_from_score = (dscore * base).sum()
        dsm_from_sigmoid = (dz_leader * sigmoid_prime * dot_leader).sum()
        dsm_scale = dsm_from_score + dsm_from_sigmoid

        dq = dq_total.to(q.dtype).contiguous()
        dk = dk_total.to(k.dtype).contiguous()
        dv = dv.to(v.dtype).contiguous()

        return dq, dk, dv, None, None, None, None


attention = _attention.apply

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")


@pytest.mark.parametrize("Z", [1])
@pytest.mark.parametrize("H_KV", [2])
@pytest.mark.parametrize("GROUP_SIZE", [4, 8])
@pytest.mark.parametrize("N_CTX", [128])
@pytest.mark.parametrize("HEAD_DIM", [64])
def test_q_loading_backbone(Z, H_KV, GROUP_SIZE, N_CTX, HEAD_DIM, dtype=torch.float16):
    if Z * H_KV * GROUP_SIZE == 0:
        pytest.skip("degenerate case")

    H_Q = H_KV * GROUP_SIZE
    torch.manual_seed(0)

    q = torch.randn((Z, H_Q, N_CTX, HEAD_DIM), device=DEVICE, dtype=dtype)
    k = torch.randn((Z, H_KV, N_CTX, HEAD_DIM), device=DEVICE, dtype=dtype)
    v = torch.randn((Z, H_KV, N_CTX, HEAD_DIM), device=DEVICE, dtype=dtype)

    theta = 10000.0

    out = attention(q, k, v, causal=True, sm_scale=1.0, theta=theta)

    q_view = q.reshape(Z, H_KV, GROUP_SIZE, N_CTX, HEAD_DIM)
    o_view = out.reshape(Z, H_KV, GROUP_SIZE, N_CTX, HEAD_DIM)
    assert torch.allclose(o_view, q_view, atol=0, rtol=0), "Backbone should copy Q tiles verbatim at this stage."
