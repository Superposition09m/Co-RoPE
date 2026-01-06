"""
PyTorch implementation of RoPE attention for benchmarking
This is used as a baseline to compare with Triton implementation
"""

import torch
import torch.nn.functional as F


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

def forward_literal(ctx, q, k, v, causal, sm_scale, theta):
    """
    Args:
        q: (BATCH, H, N_CTX, HEAD_DIM)
        k: (BATCH, H, N_CTX, HEAD_DIM)
        v: (BATCH, H, N_CTX, HEAD_DIM)
        causal: bool
        sm_scale: float, scaling factor for attention scores
        theta: float, RoPE base (e.g., 10000.0)
    Returns:
        output: (BATCH, H, N_CTX, HEAD_DIM)
    """
    B, n_heads_q, N_CTX, HEAD_DIM = q.shape
    device = q.device

    # Compute RoPE frequencies dynamically
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))

    n_heads_kv = k.shape[1]
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
        B, _, N_CTX, HEAD_DIM = k.shape
        k_expanded = k.view(B, n_heads_kv, 1, N_CTX, HEAD_DIM).expand(
            B, n_heads_kv, group_size, N_CTX, HEAD_DIM
        ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
        v_expanded = v.view(B, n_heads_kv, 1, N_CTX, HEAD_DIM).expand(
            B, n_heads_kv, group_size, N_CTX, HEAD_DIM
        ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
    
    # --- Co-RoPE Leader-Driven Logic: BEGIN ---
    # 1. 采样领航头：每 group_size 个头取一个 (Repeat Interleave 布局下首位即领航员)
    q_leaders = q[:, ::group_size, :, :] # (B, H_kv, N, D)
    
    # 2. 计算组共享里程 (计算量缩小至 1/group_size)
    z_leader = torch.sigmoid(torch.einsum('bhqd,bhkd->bhqk', q_leaders, k) * sm_scale)
    a_leader = torch.cumsum(z_leader, dim=-1)
    
    # 3. 驱动全组：广播里程分布
    a = a_leader.repeat_interleave(group_size, dim=1) # (B, H_q, N, N)
    a_tt = torch.diagonal(a, dim1=-2, dim2=-1).unsqueeze(-1)
    # --- Co-RoPE Leader-Driven Logic: END ---
    
    q_phi = a_tt * inv_freq.view(1, 1, 1, -1).repeat(1, 1, 1, 2)
    q = q * torch.cos(q_phi) + rotate_half(q) * torch.sin(q_phi)
    
    k_phi = a.unsqueeze(-1) * inv_freq.view(1, 1, 1, 1, -1).repeat(1, 1, 1, 1, 2)
    k_expanded = k_expanded.unsqueeze(2) * torch.cos(k_phi) + rotate_half(k_expanded.unsqueeze(2)) * torch.sin(k_phi)
    
    # Compute attention scores: Q @ K^T * sm_scale
    # Use fp32 for dot product accumulation to avoid precision loss in long sequences
    attn_scores = (q.unsqueeze(3) * k_expanded).sum(dim=-1) * sm_scale
    
    # Apply causal mask if needed
    if causal:
        N_CTX = q.shape[2]
        mask = torch.triu(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool), diagonal=1)
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

    # Save for backward
    if ctx is not None:
        ctx.save_for_backward(q, k, v, attn_weights)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.group_size = group_size
        ctx.n_kv_heads = n_heads_kv
        ctx.theta = theta

    return output


def forward_calibrated(ctx, q, k, v, causal, sm_scale, theta):
    """
    Args:
        q: (BATCH, H, N_CTX, HEAD_DIM)
        k: (BATCH, H, N_CTX, HEAD_DIM)
        v: (BATCH, H, N_CTX, HEAD_DIM)
        causal: bool
        sm_scale: float, scaling factor for attention scores
        theta: float, RoPE base (e.g., 10000.0)
    Returns:
        output: (BATCH, H, N_CTX, HEAD_DIM)
    """
    B, n_heads_q, N_CTX, HEAD_DIM = q.shape
    device = q.device

    # Compute RoPE frequencies dynamically
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))

    n_heads_kv = k.shape[1]
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
        B, _, N_CTX, HEAD_DIM = k.shape
        k_expanded = k.view(B, n_heads_kv, 1, N_CTX, HEAD_DIM).expand(
            B, n_heads_kv, group_size, N_CTX, HEAD_DIM
        ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
        v_expanded = v.view(B, n_heads_kv, 1, N_CTX, HEAD_DIM).expand(
            B, n_heads_kv, group_size, N_CTX, HEAD_DIM
        ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
    
    # --- Co-RoPE Leader-Driven Logic: BEGIN ---
    # 1. 采样领航头并计算组共享里程
    q_leaders = q[:, ::group_size, :, :]
    z_leader = torch.sigmoid(torch.einsum('bhqd,bhkd->bhqk', q_leaders, k) * sm_scale)
    a_leader = torch.cumsum(z_leader, dim=-1)
    
    # 2. 广播里程分布并计算里程差 delta_a
    a = a_leader.repeat_interleave(group_size, dim=1)
    delta_a = torch.diagonal(a, dim1=-2, dim2=-1).unsqueeze(-1) - a
    # --- Co-RoPE Leader-Driven Logic: END ---
    
    d_half = HEAD_DIM // 2
    q1, q2 = q[..., :d_half], q[..., d_half:]
    k1, k2 = k_expanded[..., :d_half], k_expanded[..., d_half:]
    
    E_A = q1.unsqueeze(3) * k1.unsqueeze(2) + q2.unsqueeze(3) * k2.unsqueeze(2)
    E_B = q2.unsqueeze(3) * k1.unsqueeze(2) - q1.unsqueeze(3) * k2.unsqueeze(2)
    
    phi = delta_a.unsqueeze(-1) * inv_freq.view(1, 1, 1, 1, -1)
    
    # Compute attention scores: Q @ K^T * sm_scale
    # Use fp32 for dot product accumulation to avoid precision loss in long sequences
    attn_scores = (E_A * torch.cos(phi) - E_B * torch.sin(phi)).sum(dim=-1) * sm_scale
    
    # Apply causal mask if needed
    if causal:
        N_CTX = q.shape[2]
        mask = torch.triu(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool), diagonal=1)
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

    # Save for backward
    if ctx is not None:
        ctx.save_for_backward(q, k, v, attn_weights)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.group_size = group_size
        ctx.n_kv_heads = n_heads_kv
        ctx.theta = theta

    return output


if __name__ == "__main__":
    import torch
    
    # 1. 环境准备
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 128, 64
    sm_scale = D ** -0.5
    theta = 10000.0
    causal = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🚀 Starting Verification (Device: {device})")
    print(f"Config: B={B}, H={H}, N={N}, D={D}, causal={causal}")

    # 准备输入并开启梯度追踪
    q = torch.randn(B, H, N, D, device=device, requires_grad=True)
    k = torch.randn(B, H // 2, N, D, device=device, requires_grad=True) # 测试 GQA
    v = torch.randn(B, H // 2, N, D, device=device, requires_grad=True)

    # 2. 前向对拍
    out_a = forward_literal(None, q, k, v, causal, sm_scale, theta)
    out_b = forward_calibrated(None, q, k, v, causal, sm_scale, theta)

    max_diff_fwd = (out_a - out_b).abs().max().item()
    print(f"\n🔍 Forward Max Diff: {max_diff_fwd:.2e}")

    # 3. 梯度对拍
    # 为避免累积误差，克隆梯度进行独立测试
    loss_a = out_a.sum()
    loss_a.backward()
    grad_q_a = q.grad.clone()
    grad_k_a = k.grad.clone()
    q.grad.zero_()
    k.grad.zero_()

    loss_b = out_b.sum()
    loss_b.backward()
    grad_q_b = q.grad.clone()
    grad_k_b = k.grad.clone()

    max_diff_grad_q = (grad_q_a - grad_q_b).abs().max().item()
    max_diff_grad_k = (grad_k_a - grad_k_b).abs().max().item()
    print(f"🔍 Gradient Max Diff (dQ): {max_diff_grad_q:.2e}")
    print(f"🔍 Gradient Max Diff (dK): {max_diff_grad_k:.2e}")

    # 4. 相似度评估
    cos_sim_q = torch.nn.functional.cosine_similarity(grad_q_a.flatten(), grad_q_b.flatten(), dim=0)
    cos_sim_k = torch.nn.functional.cosine_similarity(grad_k_a.flatten(), grad_k_b.flatten(), dim=0)
    print(f"✅ Grad Q Cosine Similarity: {cos_sim_q.item():.8f}")
    print(f"✅ Grad K Cosine Similarity: {cos_sim_k.item():.8f}")

    if max_diff_fwd < 1e-5 and max_diff_grad_q < 1e-5:
        print("\n✨ [PASS] Both variants are numerically equivalent!")
    else:
        print("\n❌ [FAIL] Numerical discrepancy detected!")
