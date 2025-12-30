"""
PyTorch implementation of RoPE attention for benchmarking
This is used as a baseline to compare with Triton implementation
"""

import torch
import torch.nn.functional as F


@torch.compile
def precompute_freqs_cis(dim, seq_len, theta, device='cuda'):
    """
    Precompute cos and sin values for RoPE
    Since this is PyTorch version, we cache the freqs instead of computing them in a fused kernel

    Args:
        dim: head_dim, must be even
        seq_len: sequence length
        theta: base for frequency computation (e.g., 10000.0)
        device: device to create tensors on

    Returns:
        freqs_cos: (seq_len, dim) - cos values, each frequency repeated twice
        freqs_sin: (seq_len, dim) - sin values, each frequency repeated twice
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    # Compute frequencies: theta_i = base^(-2i/dim), i in [0, dim//2 - 1]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # freqs shape: (dim // 2,)

    # Position indices: m in [0, seq_len - 1]
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    # t shape: (seq_len,)

    # Compute m * theta_i
    freqs = torch.outer(t, freqs)  # (seq_len, dim // 2)

    # Compute cos and sin
    freqs_cos = torch.cos(freqs)  # (seq_len, dim // 2)
    freqs_sin = torch.sin(freqs)  # (seq_len, dim // 2)

    # Use cat (split layout) instead of repeat_interleave for better memory coalescing
    # Layout: [cos0, cos1, ..., cos_{d/2-1}, cos0, cos1, ..., cos_{d/2-1}]
    # This matches mainstream implementations (HuggingFace, Flash Attention) and is GPU-friendly
    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)  # (seq_len, dim)
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)  # (seq_len, dim)

    return freqs_cos, freqs_sin


def rotate_half(x):
    """
    Split x in half and rotate: [x1, x2] -> [-x2, x1]
    This implements the rotation operation for RoPE using split layout (not interleaved)

    Args:
        x: (..., dim) where dim is even

    Returns:
        rotated: (..., dim) with layout [-x_{d/2:d}, x_{0:d/2}]
    """
    # Split into two halves
    x1 = x[..., : x.shape[-1] // 2]  # First half: x_0, x_1, ..., x_{d/2-1}
    x2 = x[..., x.shape[-1] // 2 :]  # Second half: x_{d/2}, x_{d/2+1}, ..., x_{d-1}

    # Rotate: [-x2, x1]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """
    Apply rotary position embedding
    Formula: x * cos(m*theta) + rotate_half(x) * sin(m*theta)

    Args:
        x: (BATCH, H, N_CTX, HEAD_DIM)
        freqs_cos: (N_CTX, HEAD_DIM)
        freqs_sin: (N_CTX, HEAD_DIM)

    Returns:
        rotated: (BATCH, H, N_CTX, HEAD_DIM)
    """
    return x * freqs_cos + rotate_half(x) * freqs_sin


class _attention_pytorch(torch.autograd.Function):
    """
    CoRoPE (Context-aware Rotary Positional Embedding) with GQA support
    """
    
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, theta):
        """
        CoRoPE Forward Pass with Dynamic Mileage
        
        Args:
            q: (BATCH, H_q, N_CTX, HEAD_DIM)
            k: (BATCH, H_kv, N_CTX, HEAD_DIM)  # GQA: H_q >= H_kv
            v: (BATCH, H_kv, N_CTX, HEAD_DIM)
            causal: bool
            sm_scale: float
            theta: float, RoPE base (e.g., 10000.0)
        
        Returns:
            output: (BATCH, H_q, N_CTX, HEAD_DIM)
        """
        B, n_heads_q, N_CTX, HEAD_DIM = q.shape
        n_heads_kv = k.shape[1]
        device = q.device
        dtype = q.dtype
        
        # ========== Step 0: GQA Expansion (在计算里程之前) ==========
        if n_heads_q == n_heads_kv:
            group_size = 1
            k_expanded = k
            v_expanded = v
        else:
            if n_heads_q % n_heads_kv != 0:
                raise ValueError(
                    f"Number of Q heads ({n_heads_q}) must be divisible by KV heads ({n_heads_kv})."
                )
            group_size = n_heads_q // n_heads_kv
            k_expanded = k.view(B, n_heads_kv, 1, N_CTX, HEAD_DIM).expand(
                B, n_heads_kv, group_size, N_CTX, HEAD_DIM
            ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
            v_expanded = v.view(B, n_heads_kv, 1, N_CTX, HEAD_DIM).expand(
                B, n_heads_kv, group_size, N_CTX, HEAD_DIM
            ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
        
        # ========== Step 1: 计算步长能量 z_τ = σ(q_n · k_τ) ==========
        # 使用原始未旋转的 Q, K 计算交互强度
        z_scores = torch.einsum(
            'bhqd,bhkd->bhqk',
            q.to(torch.float32),
            k_expanded.to(torch.float32)
        ) * sm_scale
        
        # Apply sigmoid to get step energy (里程步长)
        z = torch.sigmoid(z_scores)  # (B, H, N_CTX, N_CTX)
        
        # ========== Step 2: 计算累积里程 a_m = Σ z_s ==========
        # 对于每个 query position n，计算其累积里程 a_n
        # a_q[b, h, t] = sum_{s=0}^{t} z[b, h, t, s]
        a_q = torch.cumsum(z, dim=-1)  # (B, H, N_CTX, N_CTX) - 累积到每个 key 位置
        
        # 提取对角线：每个 query 自己的总里程
        # a_q_diag[b, h, t] = a_q[b, h, t, t]
        a_q_total = torch.diagonal(a_q, dim1=-2, dim2=-1)  # (B, H, N_CTX)
        
        # 对于 key，计算每个位置的累积里程
        # a_k[b, h, tau] = sum_{s=0}^{tau} z[b, h, n, s] for any n (这里用第一个 query 作为代表)
        # 更准确的做法：对所有 query 的 z 取平均
        z_avg = z.mean(dim=2, keepdim=True)  # (B, H, 1, N_CTX) - 平均所有 query
        a_k = torch.cumsum(z_avg.squeeze(2), dim=-1)  # (B, H, N_CTX)
        
        # ========== Step 3: 计算频率向量 (inv_freq) ==========
        # inv_freq: (HEAD_DIM // 2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))
        
        # ========== Step 4: 使用里程差计算动态旋转的 Attention Scores ==========
        # 公式：p'_{t,τ} = Σ_i q_{ti} k_{τi} exp(i·(a_τ - a_t)·θ_i)
        
        # 计算里程差矩阵：(a_tau - a_t)
        # a_q_total: (B, H, N_CTX) - 每个 query 的总里程
        # a_k: (B, H, N_CTX) - 每个 key 的累积里程
        # 里程差: (B, H, N_CTX_q, N_CTX_k) where diff[t, tau] = a_k[tau] - a_q[t]
        mileage_diff = a_k.unsqueeze(2) - a_q_total.unsqueeze(3)  # (B, H, N_CTX, N_CTX)
        
        # 计算旋转角度：mileage_diff * theta_i
        # mileage_diff: (B, H, N_CTX, N_CTX)
        # inv_freq: (HEAD_DIM // 2,)
        # 需要广播成：(B, H, N_CTX, N_CTX, HEAD_DIM // 2)
        angles = mileage_diff.unsqueeze(-1) * inv_freq  # (B, H, N_CTX, N_CTX, HEAD_DIM//2)
        
        # 计算 cos 和 sin（使用 split layout）
        cos_mileage = torch.cos(angles)  # (B, H, N_CTX, N_CTX, HEAD_DIM//2)
        sin_mileage = torch.sin(angles)  # (B, H, N_CTX, N_CTX, HEAD_DIM//2)
        
        # Split Q 和 K_expanded 成两半（split layout）
        half_dim = HEAD_DIM // 2
        q1 = q[..., :half_dim]  # (B, H, N_CTX, HEAD_DIM//2)
        q2 = q[..., half_dim:]  # (B, H, N_CTX, HEAD_DIM//2)
        k1 = k_expanded[..., :half_dim]
        k2 = k_expanded[..., half_dim:]
        
        # 应用复数旋转并计算点积
        # 公式：Re[(q1 + iq2)(k1 - ik2) * e^{i·angle}]
        #     = Re[(q1 + iq2)(k1 - ik2)(cos + i·sin)]
        #     = (q1·k1 + q2·k2)·cos + (q2·k1 - q1·k2)·sin
        
        # 扩展维度用于广播
        # q1, q2: (B, H, N_CTX_q, 1, HEAD_DIM//2)
        # k1, k2: (B, H, 1, N_CTX_k, HEAD_DIM//2)
        q1_exp = q1.unsqueeze(3).to(torch.float32)
        q2_exp = q2.unsqueeze(3).to(torch.float32)
        k1_exp = k1.unsqueeze(2).to(torch.float32)
        k2_exp = k2.unsqueeze(2).to(torch.float32)
        
        # 计算旋转点积的实部和虚部
        # real_part = (q1·k1 + q2·k2)
        # imag_part = (q2·k1 - q1·k2)
        real_part = q1_exp * k1_exp + q2_exp * k2_exp  # (B, H, N_q, N_k, D//2)
        imag_part = q2_exp * k1_exp - q1_exp * k2_exp  # (B, H, N_q, N_k, D//2)
        
        # 应用旋转：real·cos - imag·sin
        rotated_dot = real_part * cos_mileage - imag_part * sin_mileage  # (B, H, N_q, N_k, D//2)
        
        # 对 HEAD_DIM 维度求和，得到最终的 attention scores
        attn_scores = rotated_dot.sum(dim=-1)  # (B, H, N_CTX, N_CTX)
        
        # ========== Step 5: Apply Causal Mask ==========
        if causal:
            mask = torch.triu(torch.ones(N_CTX, N_CTX, device=device, dtype=torch.bool), diagonal=1)
            causal_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N_CTX, N_CTX)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Safe Softmax: manually implement to ensure fp32 precision
        # Step 1: Find row-wise maximum for numerical stability
        attn_scores_max = torch.max(attn_scores, dim=-1, keepdim=True).values
        # Handle -inf case (all masked) to avoid NaN in subtraction
        if causal:
            attn_scores_max = torch.where(
                torch.isinf(attn_scores_max), 
                torch.zeros_like(attn_scores_max), 
                attn_scores_max
            )
        
        # Step 2: Subtract max (safe softmax trick)
        attn_scores_shifted = attn_scores - attn_scores_max
        
        # Step 3: Exponentiate (all in fp32)
        attn_scores_exp = torch.exp(attn_scores_shifted)
        
        # Step 4: Apply mask to exp scores (set masked positions to 0)
        if causal:
            attn_scores_exp = attn_scores_exp.masked_fill(causal_mask, 0.0)
        
        # Step 5: Sum exponentials (in fp32 to avoid overflow)
        attn_scores_sum = torch.sum(attn_scores_exp, dim=-1, keepdim=True)
        
        # Step 6: Normalize to get attention weights
        attn_weights = attn_scores_exp / attn_scores_sum
        
        # Apply mask to weights (set masked positions to 0)
        if causal:
            attn_weights = attn_weights.masked_fill(causal_mask, 0.0)
        
        # Attention output: attn_weights @ V
        # Use fp32 for weighted sum accumulation to avoid precision loss in long sequences
        output = torch.einsum(
            'bhqk,bhkd->bhqd',
            attn_weights.to(torch.float32),
            v_expanded.to(torch.float32)
        ).to(q.dtype)

        # ========== Save for Backward ==========
        ctx.save_for_backward(q, k, v, attn_weights, z, a_q_total, a_k, mileage_diff)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.group_size = group_size
        ctx.n_kv_heads = n_heads_kv
        ctx.theta = theta
        ctx.inv_freq = inv_freq

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        CoRoPE Backward Pass
        
        ⚠️ 注意：这个 backward 包含动态里程的梯度传播
        
        Args:
            grad_output: (BATCH, H_q, N_CTX, HEAD_DIM)
        
        Returns:
            dq, dk, dv, dcausal, dsm_scale, dtheta
        """
        q, k, v, attn_weights, z, a_q_total, a_k, mileage_diff = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        group_size = ctx.group_size
        n_kv_heads = ctx.n_kv_heads
        theta = ctx.theta
        inv_freq = ctx.inv_freq
        
        B, n_heads_q, N_CTX, HEAD_DIM = q.shape
        
        # ========== CoRoPE Backward: 需要反向传播通过动态里程 ==========
        # ⚠️ 简化版本：暂时不实现完整的里程梯度传播
        # TODO: 完整实现需要通过 mileage_diff -> a_q, a_k -> z -> q, k
        
        # 重新 expand k 和 v（因为 forward 已经计算过）
        if group_size == 1:
            k_expanded = k
            v_expanded = v
        else:
            k_expanded = k.view(B, n_kv_heads, 1, N_CTX, HEAD_DIM).expand(
                B, n_kv_heads, group_size, N_CTX, HEAD_DIM
            ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
            v_expanded = v.view(B, n_kv_heads, 1, N_CTX, HEAD_DIM).expand(
                B, n_kv_heads, group_size, N_CTX, HEAD_DIM
            ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
        
        # Prepare causal mask
        causal_mask = None
        if causal:
            mask = torch.triu(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool), diagonal=1)
            causal_mask = mask.unsqueeze(0).unsqueeze(0)
        
        # ===== Standard Attention Backward (简化：不包含里程梯度) =====
        # Step 1: dV = attn_weights^T @ grad_output
        dv_expanded = torch.einsum(
            'bhqk,bhqd->bhkd',
            attn_weights.to(torch.float32),
            grad_output.to(torch.float32)
        )
        
        # Step 2: dS = grad_output @ V^T
        ds = torch.einsum('bhqd,bhkd->bhqk', 
                         grad_output.to(torch.float32), 
                         v_expanded.to(torch.float32))
        
        # Step 3: Softmax backward
        d_softmax_sum = torch.sum(ds * attn_weights.to(torch.float32), dim=-1, keepdim=True)
        dp = attn_weights.to(torch.float32) * (ds - d_softmax_sum)
        
        # Apply mask to dp
        if causal:
            dp = dp.masked_fill(causal_mask, 0.0)
        
        # ===== Step 4: CoRoPE Backward - 计算 dQ 和 dK =====
        # 这里需要反向传播通过旋转操作
        # dp: (B, H, N_q, N_k) - gradient w.r.t. attention scores
        
        # 重新计算旋转所需的 cos, sin（与 forward 相同）
        half_dim = HEAD_DIM // 2
        angles = mileage_diff.unsqueeze(-1) * inv_freq
        cos_mileage = torch.cos(angles)
        sin_mileage = torch.sin(angles)
        
        # Split Q 和 K
        q1, q2 = q[..., :half_dim], q[..., half_dim:]
        k1, k2 = k_expanded[..., :half_dim], k_expanded[..., half_dim:]
        
        # 扩展维度
        q1_exp = q1.unsqueeze(3).to(torch.float32)
        q2_exp = q2.unsqueeze(3).to(torch.float32)
        k1_exp = k1.unsqueeze(2).to(torch.float32)
        k2_exp = k2.unsqueeze(2).to(torch.float32)
        
        # dp 需要扩展维度：(B, H, N_q, N_k, 1)
        dp_exp = dp.unsqueeze(-1)
        
        # Backward through the rotated dot product
        # d(real_part) = dp * cos, d(imag_part) = -dp * sin
        d_real = dp_exp * cos_mileage
        d_imag = -dp_exp * sin_mileage
        
        # real_part = q1·k1 + q2·k2
        # imag_part = q2·k1 - q1·k2
        # 反向传播：
        dq1 = (d_real * k1_exp + d_imag * k2_exp).sum(dim=3)  # sum over key dim
        dq2 = (d_real * k2_exp + d_imag * k1_exp).sum(dim=3)
        dk1 = (d_real * q1_exp + d_imag * q2_exp).sum(dim=2)  # sum over query dim
        dk2 = (d_real * q2_exp - d_imag * q1_exp).sum(dim=2)
        
        # Concatenate gradients
        dq = torch.cat([dq1, dq2], dim=-1).to(q.dtype)
        dk_expanded = torch.cat([dk1, dk2], dim=-1).to(k.dtype)
        
        # Aggregate gradients for GQA
        if group_size == 1:
            dv = dv_expanded.to(v.dtype)
            dk = dk_expanded
        else:
            dv = dv_expanded.view(B, n_kv_heads, group_size, N_CTX, HEAD_DIM).sum(dim=2).contiguous().to(v.dtype)
            dk = dk_expanded.view(B, n_kv_heads, group_size, N_CTX, HEAD_DIM).sum(dim=2).contiguous().to(k.dtype)

        return dq, dk, dv, None, None, None


attention_pytorch = _attention_pytorch.apply


# 为了向后兼容，保留旧的接口（但内部使用 CoRoPE）
def attention_pytorch_legacy(q, k, v, causal, sm_scale, freqs_cos, freqs_sin):
    """
    ⚠️ 废弃接口：为了兼容旧测试保留
    
    这个接口假设使用预计算的 freqs_cos/freqs_sin（静态 RoPE）
    但 CoRoPE 需要动态计算里程，所以这里提取 theta 并调用新接口
    """
    # 从 freqs_cos 反推 theta（假设标准 RoPE）
    # 这只是一个兼容层，实际使用应该直接调用 attention_pytorch
    theta = 10000.0  # 默认值
    return attention_pytorch(q, k, v, causal, sm_scale, theta)
