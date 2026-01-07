"""
PyTorch implementation of plain attention for benchmarking
This is used as a baseline to compare with Triton implementation
"""

import torch
import torch.nn.functional as F


class _attention_pytorch(torch.autograd.Function):
    """
    Plain PyTorch Attention with manual backward pass
    Interface compatible with flash_attn_v2_triton.py
    """
    
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        """
        Args:
            q: (BATCH, H, N_CTX, HEAD_DIM)
            k: (BATCH, H, N_CTX, HEAD_DIM)
            v: (BATCH, H, N_CTX, HEAD_DIM)
            causal: bool
            sm_scale: float, scaling factor for attention scores
        Returns:
            output: (BATCH, H, N_CTX, HEAD_DIM)
        """
        n_heads_q = q.shape[1]
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
        
        # Compute attention scores: Q @ K^T * sm_scale
        # Use fp32 for dot product accumulation to avoid precision loss in long sequences
        attn_scores = torch.einsum(
            'bhqd,bhkd->bhqk',
            q.to(torch.float32),
            k_expanded.to(torch.float32)
        ) * sm_scale
        
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
        ctx.save_for_backward(q, k, v, attn_weights)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.group_size = group_size
        ctx.n_kv_heads = n_heads_kv
        
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
        """
        q, k, v, attn_weights = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        group_size = ctx.group_size
        n_kv_heads = ctx.n_kv_heads
        n_heads_q = q.shape[1]
        B, _, N_CTX, HEAD_DIM = q.shape
        
        # Prepare causal mask if needed
        causal_mask = None
        if causal:
            N_CTX = q.shape[2]
            mask = torch.triu(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool), diagonal=1)
            causal_mask = mask.unsqueeze(0).unsqueeze(0)
        
        if group_size == 1:
            k_expanded_fp32 = k.to(torch.float32)
            v_expanded_fp32 = v.to(torch.float32)
        else:
            k_expanded_fp32 = k.view(B, n_kv_heads, 1, N_CTX, HEAD_DIM).expand(
                B, n_kv_heads, group_size, N_CTX, HEAD_DIM
            ).reshape(B, n_heads_q, N_CTX, HEAD_DIM).to(torch.float32)
            v_expanded_fp32 = v.view(B, n_kv_heads, 1, N_CTX, HEAD_DIM).expand(
                B, n_kv_heads, group_size, N_CTX, HEAD_DIM
            ).reshape(B, n_heads_q, N_CTX, HEAD_DIM).to(torch.float32)
        
        # Step 1: dV = attn_weights^T @ grad_output
        # Use fp32 for matrix multiplication accumulation
        dv_expanded = torch.einsum(
            'bhqk,bhqd->bhkd',
            attn_weights.to(torch.float32),
            grad_output.to(torch.float32)
        )
        
        # Step 2: dS = grad_output @ V^T (gradient w.r.t. attn_weights)
        # Use fp32 for matrix multiplication accumulation
        ds = torch.einsum('bhqd,bhkd->bhqk', grad_output.to(torch.float32), v_expanded_fp32)
        
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
        dq = torch.einsum('bhqk,bhkd->bhqd', dp, k_expanded_fp32) * sm_scale

        # Step 5: dK = dp^T @ Q * sm_scale
        # Use fp32 for matrix multiplication accumulation
        dk_expanded = torch.einsum('bhqk,bhqd->bhkd', dp, q.to(torch.float32)) * sm_scale
        
        if group_size == 1:
            dv = dv_expanded.to(v.dtype)
            dk = dk_expanded.to(k.dtype)
        else:
            dv = dv_expanded.view(B, n_kv_heads, group_size, N_CTX, HEAD_DIM).sum(dim=2).contiguous().to(v.dtype)
            dk = dk_expanded.view(B, n_kv_heads, group_size, N_CTX, HEAD_DIM).sum(dim=2).contiguous().to(k.dtype)
        
        # Convert back to original dtype
        dq = dq.to(q.dtype)
        
        return dq, dk, dv, None, None


attention_pytorch = _attention_pytorch.apply
