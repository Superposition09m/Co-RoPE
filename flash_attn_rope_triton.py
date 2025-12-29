"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import os

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




@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    k_ptr, v_ptr,  #
                    stride_kn, stride_vn,  #
                    freqs_cos_ptr, freqs_sin_ptr,  #
                    stride_freqs_seq, stride_freqs_dim,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX

    half_dim: tl.constexpr = HEAD_DIM // 2
    offs_d_first = tl.arange(0, half_dim)
    
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_local = start_n + offs_n
        
        # -- 1. 物理级对半加载 K (Physical Split Loading) --
        # 直接算出前半部分和后半部分的物理地址，消灭加载后的 reshape/permute
        k1_ptrs = k_ptr + (offset_y + offs_n_local[:, None]) * stride_kn + offs_d_first[None, :]
        k2_ptrs = k_ptr + (offset_y + offs_n_local[:, None]) * stride_kn + (half_dim + offs_d_first[None, :])
        
        mask_k = (offset_y + offs_n_local[:, None] < N_CTX)
        k1 = tl.load(k1_ptrs, mask=mask_k)
        k2 = tl.load(k2_ptrs, mask=mask_k)

        # -- 2. 频率表加载优化 (Vectorized Loading) --
        # 使用 tl.multiple_of 强制开启 128-bit 向量化加载
        freqs_cos_k_ptrs = freqs_cos_ptr + (offs_n_local[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
        freqs_sin_k_ptrs = freqs_sin_ptr + (offs_n_local[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
        
        cos_k = tl.load(tl.multiple_of(freqs_cos_k_ptrs, 8), mask=mask_k).to(tl.float32)
        sin_k = tl.load(tl.multiple_of(freqs_sin_k_ptrs, 8), mask=mask_k).to(tl.float32)

        # -- 3. 直接计算旋转 (Fused RoPE) --
        k1_rot = k1 * cos_k - k2 * sin_k
        k2_rot = k2 * cos_k + k1 * sin_k

        # -- 4. 极致拼接 (Interleaved Join for Dot) --
        # 使用 tl.join 准备 dot，这在 Triton 中通常被优化为零搬运
        k = tl.join(k1_rot, k2_rot).reshape([BLOCK_N, HEAD_DIM]).T

        # -- compute qk ----
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
            
        # -- load V using physical pointers --
        v_ptrs = v_ptr + (offset_y + offs_n_local[:, None]) * stride_vn + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_ptrs, mask=mask_k)
        
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return acc, l_i, m_i


# Autotune with single config to avoid compiler bugs
@triton.autotune(
    configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4)],
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"]
)
@triton.jit
def _attn_fwd(sm_scale, M,  #
              freqs_cos_ptr, freqs_sin_ptr,  #
              stride_freqs_seq, stride_freqs_dim,  #
              Z, H, q_ptr, k_ptr, v_ptr, o_ptr, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Strides assuming [Z, H, N_CTX, HEAD_DIM]
    stride_n = HEAD_DIM
    
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    half_dim: tl.constexpr = HEAD_DIM // 2
    offs_d_first = tl.arange(0, half_dim)

    # -- 1. 物理级对半加载 Q (Physical Split Loading) --
    q1_ptrs = q_ptr + (qo_offset_y + tl.arange(0, BLOCK_M)[:, None]) * stride_n + offs_d_first[None, :]
    q2_ptrs = q_ptr + (qo_offset_y + tl.arange(0, BLOCK_M)[:, None]) * stride_n + (half_dim + offs_d_first[None, :])
    
    mask_q = (offs_m[:, None] < N_CTX)
    q1 = tl.load(q1_ptrs, mask=mask_q)
    q2 = tl.load(q2_ptrs, mask=mask_q)

    # -- 2. 频率表加载优化 (Vectorized Loading) --
    freqs_cos_q_ptrs = freqs_cos_ptr + (offs_m[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
    freqs_sin_q_ptrs = freqs_sin_ptr + (offs_m[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
    
    cos_q = tl.load(tl.multiple_of(freqs_cos_q_ptrs, 8), mask=mask_q).to(tl.float32)
    sin_q = tl.load(tl.multiple_of(freqs_sin_q_ptrs, 8), mask=mask_q).to(tl.float32)

    # -- 3. 直接计算旋转 (Fused RoPE) --
    q1_new = q1 * cos_q - q2 * sin_q
    q2_new = q2 * cos_q + q1 * sin_q

    # -- 4. 极致拼接 (Interleaved Join for Dot) --
    q = tl.join(q1_new, q2_new).reshape([BLOCK_M, HEAD_DIM]).to(dtype)

    # stage 1: off-band
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        k_ptr, v_ptr, stride_n, stride_n,  #
                                        freqs_cos_ptr, freqs_sin_ptr,  #
                                        stride_freqs_seq, stride_freqs_dim,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        k_ptr, v_ptr, stride_n, stride_n,  #
                                        freqs_cos_ptr, freqs_sin_ptr,  #
                                        stride_freqs_seq, stride_freqs_dim,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    
    o_ptrs = o_ptr + (qo_offset_y + tl.arange(0, BLOCK_M)[:, None]) * stride_n + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(o_ptrs, acc.to(dtype), mask=mask_q)


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk1, dk2, dv,  #
                   Q, k1, k2, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   freqs_cos_ptr, freqs_sin_ptr,  #
                   stride_freqs_seq, stride_freqs_dim,  #
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)

    half_dim: tl.constexpr = HEAD_DIM // 2
    offs_d_first = tl.arange(0, half_dim)
    offs_d_second = half_dim + tl.arange(0, half_dim)

    # -- 1. 重新构造已旋转的 K --
    freqs_cos_k_ptrs = freqs_cos_ptr + (offs_n[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
    freqs_sin_k_ptrs = freqs_sin_ptr + (offs_n[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
    mask_k_half = (offs_n[:, None] < N_CTX)
    cos_k = tl.load(tl.multiple_of(freqs_cos_k_ptrs, 8), mask=mask_k_half).to(tl.float32)
    sin_k = tl.load(tl.multiple_of(freqs_sin_k_ptrs, 8), mask=mask_k_half).to(tl.float32)

    k1_rot = k1 * cos_k - k2 * sin_k
    k2_rot = k2 * cos_k + k1 * sin_k
    # 拼接 K 用于点积
    k = tl.join(k1_rot, k2_rot).reshape([BLOCK_N1, HEAD_DIM]).to(tl.float16)

    # -- 2. 循环计算 dK/dV --
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        offs_m_curr = curr_m + tl.arange(0, BLOCK_M1)
        mask_q = (offs_m_curr[:, None] < N_CTX)
        
        freqs_cos_q_ptrs = freqs_cos_ptr + (offs_m_curr[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
        freqs_sin_q_ptrs = freqs_sin_ptr + (offs_m_curr[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
        cos_q = tl.load(tl.multiple_of(freqs_cos_q_ptrs, 8), mask=mask_q).to(tl.float32)
        sin_q = tl.load(tl.multiple_of(freqs_sin_q_ptrs, 8), mask=mask_q).to(tl.float32)

        # 物理级对半加载 Q
        q1_ptrs = Q + offs_m_curr[:, None] * stride_tok + offs_d_first[None, :]
        q2_ptrs = Q + offs_m_curr[:, None] * stride_tok + offs_d_second[None, :]
        q1 = tl.load(q1_ptrs, mask=mask_q)
        q2 = tl.load(q2_ptrs, mask=mask_q)
        
        # 计算前向旋转的 Q (用于重新计算)
        q1_rot = (q1 * cos_q - q2 * sin_q).to(tl.float16)
        q2_rot = (q2 * cos_q + q1 * sin_q).to(tl.float16)
        q = tl.join(q1_rot, q2_rot).reshape([BLOCK_M1, HEAD_DIM])

        m = tl.load(M + offs_m_curr)
        qkT = tl.dot(k, q.T)
        pT = tl.math.exp2(qkT - m[None, :])
        if MASK:
            mask = (offs_m_curr[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
            
        do = tl.load(do_ptrs, mask=mask_q)
        # 计算 dV
        dv += tl.dot(pT.to(tl.float16), do)
        
        # 计算 dP 和 dS
        Di = tl.load(D + offs_m_curr)
        dpT = tl.dot(v, do.T).to(tl.float32)
        dsT = (pT * (dpT - Di[None, :])).to(tl.float16)
        
        # -- 极致优化：分别计算两个半块的梯度，消灭 tl.dot 后的 shuffle --
        dk_rot1 = tl.dot(dsT, q1_rot)
        dk_rot2 = tl.dot(dsT, q2_rot)
        
        # 直接进行 Inverse RoPE 投影并累加
        dk1 += dk_rot1 * cos_k + dk_rot2 * sin_k
        dk2 += dk_rot2 * cos_k - dk_rot1 * sin_k
        
        curr_m += step_m
        do_ptrs += step_m * stride_tok

    return dk1, dk2, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq1, dq2, q1, q2, K, V,  #
                 do, m, D,
                 freqs_cos_ptr, freqs_sin_ptr,  #
                 stride_freqs_seq, stride_freqs_dim,  #
                 # shared by Q/K/V/DO.
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)

    HALF_DIM: tl.constexpr = HEAD_DIM // 2
    offs_d_first = tl.arange(0, HALF_DIM)
    offs_d_second = HALF_DIM + tl.arange(0, HALF_DIM)

    # -- 1. 重新构造已旋转的 Q --
    freqs_cos_q_ptrs = freqs_cos_ptr + (offs_m[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
    freqs_sin_q_ptrs = freqs_sin_ptr + (offs_m[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
    mask_q_half = (offs_m[:, None] < N_CTX)
    cos_q = tl.load(tl.multiple_of(freqs_cos_q_ptrs, 8), mask=mask_q_half).to(tl.float32)
    sin_q = tl.load(tl.multiple_of(freqs_sin_q_ptrs, 8), mask=mask_q_half).to(tl.float32)

    q1_rot = q1 * cos_q - q2 * sin_q
    q2_rot = q2 * cos_q + q1 * sin_q
    q = tl.join(q1_rot, q2_rot).reshape([BLOCK_M2, HEAD_DIM]).to(tl.float16)

    # -- 2. 循环计算 dQ --
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    Di = tl.load(D + offs_m)[:, None]
    
    for blk_idx in range(num_steps):
        offs_n_curr = curr_n + tl.arange(0, BLOCK_N2)
        mask_k = (offs_n_curr[None, :] < N_CTX)
        
        freqs_cos_k_ptrs = freqs_cos_ptr + (offs_n_curr[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
        freqs_sin_k_ptrs = freqs_sin_ptr + (offs_n_curr[:, None] * stride_freqs_seq + offs_d_first[None, :] * stride_freqs_dim)
        cos_k = tl.load(tl.multiple_of(freqs_cos_k_ptrs, 8), mask=mask_k.T).to(tl.float32)
        sin_k = tl.load(tl.multiple_of(freqs_sin_k_ptrs, 8), mask=mask_k.T).to(tl.float32)

        # 物理级对半加载 K
        k1_ptrs = K + offs_n_curr[:, None] * stride_tok + offs_d_first[None, :]
        k2_ptrs = K + offs_n_curr[:, None] * stride_tok + offs_d_second[None, :]
        k1 = tl.load(k1_ptrs, mask=mask_k.T)
        k2 = tl.load(k2_ptrs, mask=mask_k.T)
        
        # 计算前向旋转的 K (用于重新计算)
        k1_rot = (k1 * cos_k - k2 * sin_k).to(tl.float16)
        k2_rot = (k2 * cos_k + k1 * sin_k).to(tl.float16)
        k = tl.join(k1_rot, k2_rot).reshape([BLOCK_N2, HEAD_DIM])

        # 加载 V 并计算 qk
        v_ptrs = V + offs_n_curr[:, None] * stride_tok + tl.arange(0, HEAD_DIM)[None, :]
        v = tl.load(v_ptrs, mask=mask_k.T)
        
        qk = tl.dot(q, k.T)
        p = tl.math.exp2(qk - m)
        if MASK:
            mask_diag = (offs_m[:, None] >= offs_n_curr[None, :])
            p = tl.where(mask_diag, p, 0.0)
            
        # 计算 dP 和 dS
        dp = tl.dot(do, v.T).to(tl.float32)
        ds = (p * (dp - Di)).to(tl.float16)
        
        # -- 极致优化：分别计算两个半块的梯度，消灭 tl.dot 后的 shuffle --
        dq_rot1 = tl.dot(ds, k1_rot)
        dq_rot2 = tl.dot(ds, k2_rot)
        
        # 直接进行 Inverse RoPE 投影并累加
        dq1 += dq_rot1 * cos_q + dq_rot2 * sin_q
        dq2 += dq_rot2 * cos_q - dq_rot1 * sin_q
        
        curr_n += step_n

    return dq1, dq2


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              freqs_cos_ptr, freqs_sin_ptr,  #
              stride_freqs_seq, stride_freqs_dim,  #
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              CAUSAL: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    HALF_DIM: tl.constexpr = HEAD_DIM // 2

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)
    offs_d_first = tl.arange(0, HALF_DIM)
    offs_d_second = HALF_DIM + tl.arange(0, HALF_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk1 = tl.zeros([BLOCK_N1, HALF_DIM], dtype=tl.float32)
    dk2 = tl.zeros([BLOCK_N1, HALF_DIM], dtype=tl.float32)

    # -- 物理级对半加载 K --
    k1_ptrs = K + offs_n[:, None] * stride_tok + offs_d_first[None, :]
    k2_ptrs = K + offs_n[:, None] * stride_tok + offs_d_second[None, :]
    k1 = tl.load(k1_ptrs, mask=offs_n[:, None] < N_CTX)
    k2 = tl.load(k2_ptrs, mask=offs_n[:, None] < N_CTX)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_n[:, None] < N_CTX)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk1, dk2, dv = _attn_bwd_dkdv(dk1, dk2, dv,  #
                                      Q, k1, k2, v, sm_scale,  #
                                      DO,  #
                                      M, D,  #
                                      freqs_cos_ptr, freqs_sin_ptr,  #
                                      stride_freqs_seq, stride_freqs_dim,  #
                                      stride_tok, stride_d,  #
                                      H, N_CTX,  #
                                      MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                                      start_n, start_m, num_steps,  #
                                      MASK=True,  #
                                      )

        start_m += num_steps * MASK_BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk1, dk2, dv = _attn_bwd_dkdv(  #
        dk1, dk2, dv,  #
        Q, k1, k2, v, sm_scale,  #
        DO,  #
        M, D,  #
        freqs_cos_ptr, freqs_sin_ptr,  #
        stride_freqs_seq, stride_freqs_dim,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False,  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # -- 延迟投影双指针写回 dK (Dual-Store) --
    dk1_ptrs = DK + offs_n[:, None] * stride_tok + offs_d_first[None, :]
    dk2_ptrs = DK + offs_n[:, None] * stride_tok + offs_d_second[None, :]
    tl.store(dk1_ptrs, (dk1 * sm_scale).to(tl.float16))
    tl.store(dk2_ptrs, (dk2 * sm_scale).to(tl.float16))

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    # -- 物理级对半加载 Q --
    q1_ptrs = Q + offs_m[:, None] * stride_tok + offs_d_first[None, :]
    q2_ptrs = Q + offs_m[:, None] * stride_tok + offs_d_second[None, :]
    q1 = tl.load(q1_ptrs, mask=offs_m[:, None] < N_CTX)
    q2 = tl.load(q2_ptrs, mask=offs_m[:, None] < N_CTX)
    
    dq1 = tl.zeros([BLOCK_M2, HALF_DIM], dtype=tl.float32)
    dq2 = tl.zeros([BLOCK_M2, HALF_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, mask=offs_m[:, None] < N_CTX)

    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq1, dq2 = _attn_bwd_dq(dq1, dq2, q1, q2, K, V,  #
                                do, m, D,  #
                                freqs_cos_ptr, freqs_sin_ptr,  #
                                stride_freqs_seq, stride_freqs_dim,  #
                                stride_tok, stride_d,  #
                                H, N_CTX,  #
                                BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                                start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                                MASK=True,  #
                                )
        end_n -= num_steps * MASK_BLOCK_N2
        # stage 2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    dq1, dq2 = _attn_bwd_dq(dq1, dq2, q1, q2, K, V,  #
                            do, m, D,  #
                            freqs_cos_ptr, freqs_sin_ptr,  #
                            stride_freqs_seq, stride_freqs_dim,  #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                            start_m, start_n, num_steps,  #
                            MASK=False,  #
                            )
                            
    # -- 延迟投影双指针写回 dQ (Dual-Store) --
    dq1_ptrs = DQ + offs_m[:, None] * stride_tok + offs_d_first[None, :]
    dq2_ptrs = DQ + offs_m[:, None] * stride_tok + offs_d_second[None, :]
    tl.store(dq1_ptrs, (dq1 * LN2).to(tl.float16))
    tl.store(dq2_ptrs, (dq2 * LN2).to(tl.float16))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, freqs_cos, freqs_sin, warp_specialize=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        
        _attn_fwd[grid](
            sm_scale, M,  #
            freqs_cos, freqs_sin,  #
            freqs_cos.stride(0),  #
            freqs_cos.stride(1),  #
            q.shape[0], q.shape[1],  #
            q, k, v, o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M, freqs_cos, freqs_sin)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, freqs_cos, freqs_sin = ctx.saved_tensors
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  #
            delta,  #
            BATCH, N_HEAD, N_CTX,  #
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, k, v, ctx.sm_scale, do, dq, dk, dv,  #
            M, delta,  #
            freqs_cos, freqs_sin,  #
            freqs_cos.stride(0),  #
            freqs_cos.stride(1),  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            N_HEAD, N_CTX,  #
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
            CAUSAL=ctx.causal,  #
        )

        return dq, dk, dv, None, None, None, None, None


attention = _attention.apply

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')


@pytest.mark.parametrize("Z", [1, 4])
@pytest.mark.parametrize("H", [2, 48])
@pytest.mark.parametrize("N_CTX", [128, 1024, (2 if is_hip() else 4) * 1024])
@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("warp_specialize", [False, True] if is_blackwell() else [False])
@pytest.mark.parametrize("mode", ["fwd", "bwd"])
@pytest.mark.parametrize("provider", ["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []))
def test_op(Z, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, dtype=torch.float16):
    if mode == "fwd" and "fp16" in provider:
        pytest.skip("Avoid running the forward computation twice.")
    if mode == "bwd" and "fp8" in provider:
        pytest.skip("Backward pass with FP8 is not supported.")
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    ref_dtype = dtype
    if mode == "fwd" and "fp8" in provider:
        ref_dtype = torch.float32
    q = q.to(ref_dtype)
    k = k.to(ref_dtype)
    v = v.to(ref_dtype)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1)
    p = p.to(ref_dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v).half()
    if mode == "bwd":
        dout = torch.randn_like(q)
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    if mode == "fwd" and "fp8" in provider:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
    tri_out = attention(q, k, v, causal, sm_scale, warp_specialize).half()
    if mode == "fwd":
        atol = 3 if "fp8" in provider else 1e-2
        torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
        return
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS = 4, 32
# vary seq length for fixed head and batch=4
configs = []
for HEAD_DIM in [64, 128]:
    for mode in ["fwd", "bwd"]:
        for causal in [True, False]:
            # Enable warpspec for causal fwd on Hopper
            enable_ws = mode == "fwd" and (is_blackwell() or (is_hopper() and not causal))
            for warp_specialize in [False, True] if enable_ws else [False]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],
                        x_vals=[2**i for i in range(10, 15)],
                        line_arg="provider",
                        line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                        (["flash"] if HAS_FLASH else []),
                        line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                        (["Flash-2"] if HAS_FLASH else []),
                        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                        ylabel="TFLOPS",
                        plot_name=
                        f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
                        args={
                            "H": N_HEADS,
                            "BATCH": BATCH,
                            "HEAD_DIM": HEAD_DIM,
                            "mode": mode,
                            "causal": causal,
                            "warp_specialize": warp_specialize,
                        },
                    ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, warp_specialize)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
