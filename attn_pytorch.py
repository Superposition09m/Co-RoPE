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
        ctx.save_for_backward(q, k, v, attn_weights)
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
        """
        q, k, v, attn_weights = ctx.saved_tensors
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
        
        return dq, dk, dv, None, None


attention_pytorch = _attention_pytorch.apply

