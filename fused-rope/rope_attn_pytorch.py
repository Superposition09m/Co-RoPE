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
    Plain PyTorch Attention with manual backward pass
    Interface compatible with flash_attn_v2_triton.py
    """
    
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, freqs_cos, freqs_sin):
        """
        Args:
            q: (BATCH, H, N_CTX, HEAD_DIM)
            k: (BATCH, H, N_CTX, HEAD_DIM)
            v: (BATCH, H, N_CTX, HEAD_DIM)
            causal: bool
            sm_scale: float, scaling factor for attention scores
            freqs_cos: (N_CTX, HEAD_DIM) - precomputed cos values for RoPE
            freqs_sin: (N_CTX, HEAD_DIM) - precomputed sin values for RoPE
        Returns:
            output: (BATCH, H, N_CTX, HEAD_DIM)
        """
        # Apply RoPE to Q and K (V does not need RoPE)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        # Compute attention scores: Q @ K^T * sm_scale
        # Use fp32 for dot product accumulation to avoid precision loss in long sequences
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q.to(torch.float32), k.to(torch.float32)) * sm_scale
        
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
        output = torch.einsum('bhqk,bhkd->bhqd', attn_weights.to(torch.float32), v.to(torch.float32)).to(q.dtype)

        # Save for backward
        ctx.save_for_backward(q, k, v, attn_weights, freqs_cos, freqs_sin)
        ctx.sm_scale = sm_scale
        ctx.causal = causal

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: (BATCH, H, N_CTX, HEAD_DIM) - gradient w.r.t. output
        Returns:
            dq, dk, dv: (BATCH, H, N_CTX, HEAD_DIM)
            dcausal: None
            dsm_scale: None
            dfreqs_cos: None
            dfreqs_sin: None
        """
        q, k, v, attn_weights, freqs_cos, freqs_sin = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        
        # Prepare causal mask if needed
        causal_mask = None
        if causal:
            N_CTX = q.shape[2]
            mask = torch.triu(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool), diagonal=1)
            causal_mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Step 1: dV = attn_weights^T @ grad_output
        # Use fp32 for matrix multiplication accumulation
        dv = torch.einsum('bhqk,bhqd->bhkd', attn_weights.to(torch.float32), grad_output.to(torch.float32)).to(grad_output.dtype)
        
        # Step 2: dS = grad_output @ V^T (gradient w.r.t. attn_weights)
        # Use fp32 for matrix multiplication accumulation
        ds = torch.einsum('bhqd,bhkd->bhqk', grad_output.to(torch.float32), v.to(torch.float32))
        
        # Step 3: Softmax backward
        # Use fp32 for accumulation to avoid precision loss in mixed precision training
        d_softmax_sum = torch.sum(
            ds * attn_weights.to(torch.float32), 
            dim=-1, 
            keepdim=True
        )
        dp = attn_weights.to(torch.float32) * (ds - d_softmax_sum)
        
        # Apply mask to dp (gradient w.r.t. attn_scores)
        if causal:
            dp = dp.masked_fill(causal_mask, 0.0)
        
        # Step 4: dQ = dp @ K * sm_scale
        # Use fp32 for matrix multiplication accumulation
        dq = torch.einsum('bhqk,bhkd->bhqd', dp, k.to(torch.float32)) * sm_scale

        # Step 5: dK = dp^T @ Q * sm_scale
        # Use fp32 for matrix multiplication accumulation
        dk = torch.einsum('bhqk,bhqd->bhkd', dp, q.to(torch.float32)) * sm_scale
        
        # Convert back to original dtype
        dq = dq.to(q.dtype)
        dk = dk.to(k.dtype)

        # Apply inverse RoPE to gradients (use -freqs_sin for inverse rotation)
        dq = apply_rotary_emb(dq, freqs_cos, -freqs_sin)
        dk = apply_rotary_emb(dk, freqs_cos, -freqs_sin)

        return dq, dk, dv, None, None, None, None


attention_pytorch = _attention_pytorch.apply

