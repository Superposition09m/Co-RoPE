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
from triton.tools.tensor_descriptor import TensorDescriptor

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
def _attn_fwd_inner(acc, l_i, m_i, Q_ptr, Q_leader_ptr,  #
                    K_ptr, desc_v,  #
                    offset_q_y, offset_kv_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr,
                    stride_q_seq: tl.constexpr, stride_q_dim: tl.constexpr,  #
                    stride_k_seq: tl.constexpr, stride_k_dim: tl.constexpr,  #
                    inv_freq_ptr, stride_inv_freq: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX

    # ============================================
    # Co-RoPE Phase 1: Discovery Pass - compute mileage a_m
    # ============================================
    half_dim: tl.constexpr = HEAD_DIM // 2
    offs_d_first = tl.arange(0, half_dim)
    offs_d_second = offs_d_first + half_dim

    # Load inv_freq once into registers
    inv_freq = tl.load(inv_freq_ptr + offs_d_first * stride_inv_freq).to(tl.float32)

    # Initialize mileage accumulator (must be float32)
    a_m = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load q_leader using dual-pointer for first and second half
    offs_q_m = tl.arange(0, BLOCK_M)
    q_leader_1_ptrs = Q_leader_ptr + offs_q_m[:, None] * stride_q_seq + offs_d_first[None, :] * stride_q_dim
    q_leader_2_ptrs = Q_leader_ptr + offs_q_m[:, None] * stride_q_seq + offs_d_second[None, :] * stride_q_dim
    q_leader_1 = tl.load(q_leader_1_ptrs).to(tl.float32)
    q_leader_2 = tl.load(q_leader_2_ptrs).to(tl.float32)

    # Phase 1 loop: scan K to compute mileage
    offsetk_y_phase1 = offset_kv_y + lo
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K using dual-pointer for first and second half
        offs_k_n = tl.arange(0, BLOCK_N)
        k1_ptrs = K_ptr + (offsetk_y_phase1 + offs_k_n[:, None]) * stride_k_seq + offs_d_first[None, :] * stride_k_dim
        k2_ptrs = K_ptr + (offsetk_y_phase1 + offs_k_n[:, None]) * stride_k_seq + offs_d_second[None, :] * stride_k_dim

        k1 = tl.load(k1_ptrs).to(tl.float32)
        k2 = tl.load(k2_ptrs).to(tl.float32)

        # Compute raw dot product: q_leader @ k^T (without RoPE)
        qk_raw = tl.dot(q_leader_1, tl.trans(k1)) + tl.dot(q_leader_2, tl.trans(k2))

        # Compute z = sigmoid(qk_raw * qk_scale)
        energy = qk_raw * qk_scale
        z = 1.0 / (1.0 + tl.math.exp2(-energy * 1.44269504))

        # Apply causal mask for STAGE 2
        if STAGE == 2:
            mask = offs_m[:, None] > (start_n + offs_n[None, :])
            z = tl.where(mask, z, 0.0)

        # Accumulate mileage along K dimension (axis=1)
        a_m += tl.sum(z, axis=1)

        offsetk_y_phase1 += BLOCK_N

    # ============================================
    # Co-RoPE Phase 2: Computation Pass with EA-EB rotation
    # ============================================
    offsetk_y = offset_kv_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_kv_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_kv_y + lo

    # Initialize running mileage (must be float32)
    a_n_running = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load actual q using dual-pointer for first and second half
    q1_ptrs = Q_ptr + offs_q_m[:, None] * stride_q_seq + offs_d_first[None, :] * stride_q_dim
    q2_ptrs = Q_ptr + offs_q_m[:, None] * stride_q_seq + offs_d_second[None, :] * stride_q_dim
    q1 = tl.load(q1_ptrs).to(tl.float32)
    q2 = tl.load(q2_ptrs).to(tl.float32)

    # Precompute Q phase outside loop (avoid 3D tensor in loop)
    # phi_m = a_m * inv_freq: [BLOCK_M, half_dim]
    phi_m = a_m[:, None] * inv_freq[None, :]
    cos_am = tl.cos(phi_m)
    sin_am = tl.sin(phi_m)

    # Phase 2 loop: compute attention with Co-RoPE rotation
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K using dual-pointer for first and second half
        offs_k_n = tl.arange(0, BLOCK_N)
        k1_ptrs = K_ptr + (offsetk_y + offs_k_n[:, None]) * stride_k_seq + offs_d_first[None, :] * stride_k_dim
        k2_ptrs = K_ptr + (offsetk_y + offs_k_n[:, None]) * stride_k_seq + offs_d_second[None, :] * stride_k_dim

        k1 = tl.load(k1_ptrs).to(tl.float32)
        k2 = tl.load(k2_ptrs).to(tl.float32)

        # Compute raw dot for leader to update running mileage
        qk_raw_leader = tl.dot(q_leader_1, tl.trans(k1)) + tl.dot(q_leader_2, tl.trans(k2))
        energy_leader = qk_raw_leader * qk_scale
        z_current = 1.0 / (1.0 + tl.math.exp2(-energy_leader * 1.44269504))

        # Compute cumsum within current block
        z_cumsum = tl.cumsum(z_current, axis=1)

        # Compute K mileage for this block: a_n = a_n_running + z_cumsum
        # a_n_tile: [BLOCK_M, BLOCK_N]
        a_n_tile = a_n_running[:, None] + z_cumsum

        # Compute K phase (2D only to avoid 3D SFU explosion)
        # Take row-wise average of a_n to get representative mileage per K token
        a_n_avg = tl.sum(a_n_tile, axis=0) / BLOCK_M  # [BLOCK_N]
        
        # phi_n: [BLOCK_N, half_dim]
        phi_n = a_n_avg[:, None] * inv_freq[None, :]
        cos_an = tl.cos(phi_n)
        sin_an = tl.sin(phi_n)

        # Use addition formula: cos(am - an) = cos_am * cos_an + sin_am * sin_an
        # cos_am: [BLOCK_M, half_dim], cos_an: [BLOCK_N, half_dim]
        # Broadcasting creates [BLOCK_M, BLOCK_N, half_dim] with FMA, not SFU
        cos_phi = cos_am[:, None, :] * cos_an[None, :, :] + sin_am[:, None, :] * sin_an[None, :, :]
        sin_phi = sin_am[:, None, :] * cos_an[None, :, :] - cos_am[:, None, :] * sin_an[None, :, :]

        # Compute EA and EB (先点积，后旋转)
        E_A = tl.dot(q1, tl.trans(k1)) + tl.dot(q2, tl.trans(k2))
        E_B = tl.dot(q2, tl.trans(k1)) - tl.dot(q1, tl.trans(k2))

        # Apply rotation: qk = (E_A * cos_phi - E_B * sin_phi).sum(-1)
        qk = tl.sum(E_A[:, :, None] * cos_phi - E_B[:, :, None] * sin_phi, axis=2)
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
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # Update running mileage (sum along K dimension)
        a_n_running += tl.sum(z_current, axis=1)

        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    # Check if desc_v is a TensorDescriptor
    if not isinstance(nargs["desc_v"], TensorDescriptor):
        return
    # Set block shapes for V and O descriptors
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]

    # Filter out configs where BLOCK_M > N_CTX
    # Filter out configs where BLOCK_M < BLOCK_N when causal is True
    return [
        conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX and (
            conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# Temporarily disable autotune for debugging
# @triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
#                  prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H_Q, H_KV, Q_ptr, K_ptr, desc_v, desc_o, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              GROUP_SIZE: tl.constexpr,  #
              dtype: tl.constexpr,  #
              stride_q_z: tl.constexpr, stride_q_h: tl.constexpr, stride_q_seq: tl.constexpr, stride_q_dim: tl.constexpr,  #
              stride_k_z: tl.constexpr, stride_k_h: tl.constexpr, stride_k_seq: tl.constexpr, stride_k_dim: tl.constexpr,  #
              inv_freq_ptr,  #
              ):

    # dtype is now passed from forward function
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h_q = off_hz % H_Q
    off_h_kv = off_h_q // GROUP_SIZE

    y_dim_q = Z * H_Q * N_CTX
    y_dim_kv = Z * H_KV * N_CTX

    # Compute Q and K pointers with batch and head offset
    Q_ptr_actual = Q_ptr + off_z * stride_q_z + off_h_q * stride_q_h + start_m * BLOCK_M * stride_q_seq
    K_ptr_actual = K_ptr + off_z * stride_k_z + off_h_kv * stride_k_h

    # Compute V pointer with batch and head offset, then create descriptor
    # V needs proper pointer offset before creating descriptor
    if isinstance(desc_v, tl.tensor_descriptor):
        # Already a descriptor, use as-is (will be handled by _maybe_make_tensor_desc)
        pass
    else:
        # It's a pointer, add batch and head offset
        # Assuming V has same strides as K since they have same shape
        desc_v = desc_v + off_z * stride_k_z + off_h_kv * stride_k_h

    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, N_CTX], strides=[N_CTX, stride_k_dim],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[N_CTX, HEAD_DIM], strides=[stride_k_seq, stride_k_dim],
                                         block_shape=[BLOCK_N, HEAD_DIM])

    # Similarly for O
    if isinstance(desc_o, tl.tensor_descriptor):
        pass
    else:
        desc_o = desc_o + off_z * stride_q_z + off_h_q * stride_q_h

    desc_o = _maybe_make_tensor_desc(desc_o, shape=[N_CTX, HEAD_DIM], strides=[stride_q_seq, stride_q_dim],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    # Compute leader Q pointer
    leader_h_q = (off_h_q // GROUP_SIZE) * GROUP_SIZE
    Q_leader_ptr_actual = Q_ptr + off_z * stride_q_z + leader_h_q * stride_q_h + start_m * BLOCK_M * stride_q_seq

    # For V and O descriptors, offset is now relative to the head (batch and head offset already applied)
    # So we only need the sequence offset
    offset_kv_y = 0  # V descriptor already points to correct batch/head
    offset_q_y = 0   # Q pointer already points to correct position
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

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, Q_ptr_actual, Q_leader_ptr_actual,  #
                                        K_ptr_actual, desc_v,  #
                                        offset_q_y, offset_kv_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER,
                                        stride_q_seq, stride_q_dim,  #
                                        stride_k_seq, stride_k_dim,  #
                                        inv_freq_ptr, 1)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, Q_ptr_actual, Q_leader_ptr_actual,  #
                                        K_ptr_actual, desc_v,  #
                                        offset_q_y, offset_kv_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER,
                                        stride_q_seq, stride_q_dim,  #
                                        stride_k_seq, stride_k_dim,  #
                                        inv_freq_ptr, 1)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    # Store output using descriptor
    qo_offset_y = offset_q_y + start_m * BLOCK_M
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


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
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
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
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(do.dtype)
        dv += tl.dot(ppT, do).to(tl.float32)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(qT.dtype)
        dk += tl.dot(dsT, tl.trans(qT)).to(tl.float32)
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
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
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(kT.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT)).to(tl.float32)
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
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

    start_n = pid * BLOCK_N1
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(dk, dv,  #
                                Q, k, v, sm_scale,  #
                                DO,  #
                                M, D,  #
                                stride_tok, stride_d,  #
                                H, N_CTX,  #
                                MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                                start_n, start_m, num_steps,  #
                                MASK=True,  #
                                )

        start_m += num_steps * MASK_BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False,  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(dq, q, K, V,  #
                          do, m, D,  #
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

    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, start_n, num_steps,  #
                      MASK=False,  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        H_Q = q.shape[1]
        H_KV = k.shape[1]
        GROUP_SIZE = H_Q // H_KV
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], H_Q, q.shape[2]), device=q.device, dtype=torch.float32)

        # Compute inv_freq for Co-RoPE (theta=10000.0)
        theta = 10000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM_K, 2, device=q.device, dtype=torch.float32) / HEAD_DIM_K))

        # For now, always use plain pointers (not TensorDescriptor)
        # since autotune is disabled and pre_hook won't update block_shape
        desc_v = v
        desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * H_Q, 1)

        ctx.grid = grid
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        
        # Determine triton dtype based on input dtype
        if q.dtype == torch.float8_e5m2:
            triton_dtype = tl.float8e5
        elif q.dtype == torch.float16:
            triton_dtype = tl.float16
        elif q.dtype == torch.bfloat16:
            triton_dtype = tl.bfloat16
        else:  # torch.float32
            triton_dtype = tl.float32
        
        # Fixed configuration for debugging (no autotune)
        _attn_fwd[grid](
            sm_scale, M,  #
            q.shape[0], H_Q, H_KV,  #
            q, k, desc_v, desc_o,  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            BLOCK_M=64,  #
            BLOCK_N=64,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            GROUP_SIZE=GROUP_SIZE,  #
            dtype=triton_dtype,  #
            stride_q_z=q.stride(0), stride_q_h=q.stride(1), stride_q_seq=q.stride(2), stride_q_dim=q.stride(3),  #
            stride_k_z=k.stride(0), stride_k_h=k.stride(1), stride_k_seq=k.stride(2), stride_k_dim=k.stride(3),  #
            inv_freq_ptr=inv_freq,  #
            num_warps=4,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        assert do.is_contiguous()
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        N_HEAD_KV = k.shape[1]
        GROUP_SIZE = N_HEAD // N_HEAD_KV
        
        # For GQA, expand k and v to match q's head count for backward computation
        # This simplifies the kernel logic at the cost of some memory
        if GROUP_SIZE > 1:
            k_expanded = k.view(BATCH, N_HEAD_KV, 1, N_CTX, ctx.HEAD_DIM).expand(
                BATCH, N_HEAD_KV, GROUP_SIZE, N_CTX, ctx.HEAD_DIM
            ).reshape(BATCH, N_HEAD, N_CTX, ctx.HEAD_DIM).contiguous()
            v_expanded = v.view(BATCH, N_HEAD_KV, 1, N_CTX, ctx.HEAD_DIM).expand(
                BATCH, N_HEAD_KV, GROUP_SIZE, N_CTX, ctx.HEAD_DIM
            ).reshape(BATCH, N_HEAD, N_CTX, ctx.HEAD_DIM).contiguous()
        else:
            k_expanded = k
            v_expanded = v
        
        assert q.stride() == o.stride() == do.stride() == k_expanded.stride() == v_expanded.stride()
        dq = torch.empty_like(q)
        dk_expanded = torch.empty_like(k_expanded)
        dv_expanded = torch.empty_like(v_expanded)
        
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k_expanded
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        # Adjust PRE_BLOCK to handle smaller sequence lengths
        PRE_BLOCK = min(128, N_CTX)
        while N_CTX % PRE_BLOCK != 0:
            PRE_BLOCK //= 2
        assert PRE_BLOCK >= 16, f"N_CTX={N_CTX} is too small"
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
            q, arg_k, v_expanded, ctx.sm_scale, do, dq, dk_expanded, dv_expanded,  #
            M, delta,  #
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
        
        # For GQA, sum gradients across the group dimension
        if GROUP_SIZE > 1:
            dk = dk_expanded.view(BATCH, N_HEAD_KV, GROUP_SIZE, N_CTX, ctx.HEAD_DIM).sum(dim=2)
            dv = dv_expanded.view(BATCH, N_HEAD_KV, GROUP_SIZE, N_CTX, ctx.HEAD_DIM).sum(dim=2)
        else:
            dk = dk_expanded
            dv = dv_expanded

        return dq, dk, dv, None, None, None, None


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
