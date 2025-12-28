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
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * sm_scale
        
        # Apply causal mask if needed
        if causal:
            N_CTX = q.shape[2]
            mask = torch.triu(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool), diagonal=1)
            causal_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N_CTX, N_CTX)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply mask to weights (set masked positions to 0)
        if causal:
            attn_weights = attn_weights.masked_fill(causal_mask, 0.0)
        
        # Attention output: attn_weights @ V
        output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
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
        dv = torch.einsum('bhqk,bhqd->bhkd', attn_weights, grad_output)
        
        # Step 2: dS = grad_output @ V^T (gradient w.r.t. attn_weights)
        ds = torch.einsum('bhqd,bhkd->bhqk', grad_output, v)
        
        # Step 3: Softmax backward
        d_softmax_sum = torch.sum(ds * attn_weights, dim=-1, keepdim=True)
        dp = attn_weights * (ds - d_softmax_sum)
        
        # Apply mask to dp (gradient w.r.t. attn_scores)
        if causal:
            dp = dp.masked_fill(causal_mask, 0.0)
        
        # Step 4: dQ = dp @ K * sm_scale
        dq = torch.einsum('bhqk,bhkd->bhqd', dp, k) * sm_scale
        
        # Step 5: dK = dp^T @ Q * sm_scale
        dk = torch.einsum('bhqk,bhqd->bhkd', dp, q) * sm_scale
        
        return dq, dk, dv, None, None


attention_pytorch = _attention_pytorch.apply

