"""
Fused CoRoPE Attention (GQA Skeleton)
=====================================

This module carries the Triton kernel backbone for the CoRoPE-GQA fused
attention path.  Step 1 focuses on locking down the CTA scheduling,
per-group Q loading, and the public Python interface (no more static
cos/sin tables – only `inv_freq`).
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
    inv_freq = inv_freq

    q1_group = []
    q2_group = []

    q_dtype = tl.float16
    for g in range(GROUP_SIZE):
        head_idx = head_base + g
        if head_idx >= H_Q:
            break

        q_head_base = Q + off_z * stride_qz + head_idx * stride_qh
        q_head_base = tl.multiple_of(q_head_base, 16)

        q1_ptrs = q_head_base + offs_m[:, None] * stride_qm + offs_d_first[None, :] * stride_qk
        q2_ptrs = q_head_base + offs_m[:, None] * stride_qm + offs_d_second[None, :] * stride_qk

        q1_raw = tl.load(q1_ptrs, mask=mask_q, other=0.0)
        q2_raw = tl.load(q2_ptrs, mask=mask_q, other=0.0)

        if g == 0:
            q_dtype = q1_raw.dtype

        q1_group.append(q1_raw.to(tl.float32))
        q2_group.append(q2_raw.to(tl.float32))

    leader_q1 = q1_group[0]
    leader_q2 = q2_group[0]

    half_dim_range = tl.arange(0, half_dim)
    km_off = half_dim_range
    K_base = K + off_z * stride_kz + off_kv * stride_kh
    K_base = tl.multiple_of(K_base, 16)
    V_base = V + off_z * stride_vz + off_kv * stride_vh
    V_base = tl.multiple_of(V_base, 16)

    a_tt = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX

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

        for nn in range(BLOCK_N):
            k_index = start_n + nn
            if k_index >= N_CTX:
                break

            k1_vec = k1_tile[nn]
            k2_vec = k2_tile[nn]

            ea_leader = leader_q1 * k1_vec[None, :] + leader_q2 * k2_vec[None, :]
            dot = tl.sum(ea_leader, axis=1) * sm_scale
            z = 1.0 / (1.0 + tl.exp(-dot))

            causal_mask = (offs_m >= k_index)
            z = tl.where(causal_mask & mask_m, z, 0.0)
            a_tt += z

    num_loaded = len(q1_group)
    acc_list_first = []
    acc_list_second = []
    m_list = []
    l_list = []

    for _ in range(num_loaded):
        acc_list_first.append(tl.zeros([BLOCK_M, half_dim], dtype=tl.float32))
        acc_list_second.append(tl.zeros([BLOCK_M, half_dim], dtype=tl.float32))
        m_list.append(tl.zeros([BLOCK_M], dtype=tl.float32))
        l_list.append(tl.zeros([BLOCK_M], dtype=tl.float32))

    acc_z = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX

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

        block_cumsum = tl.zeros([BLOCK_M], dtype=tl.float32)

        for nn in range(BLOCK_N):
            k_index = start_n + nn
            if k_index >= N_CTX:
                break

            k1_vec = k1_tile[nn]
            k2_vec = k2_tile[nn]
            v1_vec = v1_tile[nn]
            v2_vec = v2_tile[nn]

            ea_leader = leader_q1 * k1_vec[None, :] + leader_q2 * k2_vec[None, :]
            dot_leader = tl.sum(ea_leader, axis=1) * sm_scale
            z = 1.0 / (1.0 + tl.exp(-dot_leader))

            causal_mask = (offs_m >= k_index)
            valid_mask = causal_mask & mask_m
            z = tl.where(valid_mask, z, 0.0)

            block_cumsum += z
            a_block = acc_z + block_cumsum
            delta = a_tt - a_block

            phi = delta[:, None] * inv_freq[None, :]
            cos_phi = tl.cos(phi)
            sin_phi = tl.sin(phi)

            for g in range(num_loaded):
                q1 = q1_group[g]
                q2 = q2_group[g]

                ea = q1 * k1_vec[None, :] + q2 * k2_vec[None, :]
                eb = q2 * k1_vec[None, :] - q1 * k2_vec[None, :]

                score = tl.sum(ea * cos_phi - eb * sin_phi, axis=1) * sm_scale
                score = tl.where(valid_mask, score, -float('inf'))

                acc_first = acc_list_first[g]
                acc_second = acc_list_second[g]
                m_prev = m_list[g]
                l_prev = l_list[g]

                m_new = tl.maximum(m_prev, score)
                alpha = tl.exp(m_prev - m_new)
                p = tl.exp(score - m_new)

                alpha = tl.where(mask_m, alpha, 0.0)
                p = tl.where(valid_mask, p, 0.0)

                l_new = l_prev * alpha + p
                acc_first = acc_first * alpha[:, None] + p[:, None] * v1_vec[None, :]
                acc_second = acc_second * alpha[:, None] + p[:, None] * v2_vec[None, :]

                acc_list_first[g] = acc_first
                acc_list_second[g] = acc_second
                m_list[g] = m_new
                l_list[g] = l_new

        acc_z += block_cumsum

    for g in range(num_loaded):
        head_idx = head_base + g
        o_head_base = O + off_z * stride_oz + head_idx * stride_oh
        o_head_base = tl.multiple_of(o_head_base, 16)

        o_half0 = o_head_base + offs_m[:, None] * stride_om + offs_d_first[None, :] * stride_ok
        o_half1 = o_head_base + offs_m[:, None] * stride_om + offs_d_second[None, :] * stride_ok

        acc_first = acc_list_first[g]
        acc_second = acc_list_second[g]
        l_final = l_list[g]
        safe_l = tl.maximum(l_final, 1e-9)
        inv_l = 1.0 / safe_l
        inv_l = tl.where(l_final > 0.0, inv_l, 0.0)

        out_first = acc_first * inv_l[:, None]
        out_second = acc_second * inv_l[:, None]

        tl.store(o_half0, out_first.to(q_dtype), mask=mask_q)
        tl.store(o_half1, out_second.to(q_dtype), mask=mask_q)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, inv_freq, warp_specialize=True):
        if not causal:
            raise ValueError("CoRoPE fused kernel supports causal=True only.")

        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("Expected q, k, v to have shape (batch, heads, seqlen, dim).")

        if q.shape != v.shape:
            raise ValueError("q and v must share the same shape for fused attention.")
        if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
            raise ValueError("k must align with q along batch/sequence/head_dim.")

        BATCH, H_Q, N_CTX, HEAD_DIM = q.shape
        H_KV = k.shape[1]
        if H_Q % H_KV != 0:
            raise ValueError("Number of query heads must be divisible by KV heads.")
        group_size = H_Q // H_KV
        if group_size > 8:
            raise ValueError(f"CoRoPE backbone currently limits group_size <= 8 (got {group_size}).")

        if inv_freq.ndim != 1:
            raise ValueError("inv_freq must be a 1D tensor of length head_dim/2.")
        if inv_freq.shape[0] != HEAD_DIM // 2:
            raise ValueError("inv_freq length must equal head_dim // 2.")
        if inv_freq.device != q.device:
            raise ValueError("inv_freq must reside on the same device as q.")

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
        d_inv = (dphi * delta.unsqueeze(-1)).sum(dim=(0, 1, 2, 3))

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
        d_inv = d_inv.to(inv_freq.dtype).contiguous()
        dsm_scale = torch.tensor(dsm_scale, device=device, dtype=torch.float32)

        return dq, dk, dv, None, dsm_scale, d_inv, None


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
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=DEVICE, dtype=torch.float32) / HEAD_DIM))

    out = attention(q, k, v, causal=True, sm_scale=1.0, inv_freq=inv_freq)

    q_view = q.reshape(Z, H_KV, GROUP_SIZE, N_CTX, HEAD_DIM)
    o_view = out.reshape(Z, H_KV, GROUP_SIZE, N_CTX, HEAD_DIM)
    assert torch.allclose(o_view, q_view, atol=0, rtol=0), "Backbone should copy Q tiles verbatim at this stage."
