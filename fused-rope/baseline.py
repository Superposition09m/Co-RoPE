"""
Baseline implementations using standard libraries
Baseline 1: Transformers RoPE + PyTorch standard attention
Baseline 2: Transformers RoPE + Flash Attention (Official CUDA)
Baseline 3: Transformers RoPE + Flash Attention v2 (Triton - 无RoPE融合)
"""

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # 添加上一层目录

from utils import calc_sim, assert_similar, print_red_warning
from flash_attn_v2_triton import attention as flash_attn_v2_triton

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash_attn not installed. Baseline 2 will not be available.")


def compute_rope_cos_sin(seq_len, head_dim, device, theta=10000.0, dtype=torch.float32):
    """
    Compute RoPE cos and sin values (transformers format)
    
    Returns:
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    
    # Compute position indices
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    
    # Compute frequencies: outer product of positions and inv_freq
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim // 2)
    
    # Compute cos and sin
    cos = torch.cos(freqs)  # (seq_len, head_dim // 2)
    sin = torch.sin(freqs)  # (seq_len, head_dim // 2)
    
    # Use cat layout (same as transformers)
    cos = torch.cat([cos, cos], dim=-1).to(dtype)  # (seq_len, head_dim)
    sin = torch.cat([sin, sin], dim=-1).to(dtype)  # (seq_len, head_dim)
    
    return cos, sin


def baseline1_rope_pytorch_attn(q, k, v, causal=True, sm_scale=None, theta=10000.0):
    """
    Baseline 1: Transformers RoPE + PyTorch F.scaled_dot_product_attention
    
    Args:
        q: (batch, n_heads, seq_len, head_dim)
        k: (batch, n_heads, seq_len, head_dim)
        v: (batch, n_heads, seq_len, head_dim)
        causal: bool, whether to use causal masking
        sm_scale: attention scale (if None, use 1/sqrt(head_dim))
        theta: RoPE base frequency
    
    Returns:
        output: (batch, n_heads, seq_len, head_dim)
    """
    batch, n_heads, seq_len, head_dim = q.shape
    device = q.device
    dtype = q.dtype
    
    # Compute RoPE cos/sin: (seq_len, head_dim)
    cos, sin = compute_rope_cos_sin(seq_len, head_dim, device, theta, dtype)
    
    # Expand to (batch, seq_len, head_dim) as required by apply_rotary_pos_emb
    cos = cos.unsqueeze(0).expand(batch, -1, -1)
    sin = sin.unsqueeze(0).expand(batch, -1, -1)
    
    # Apply RoPE to Q and K
    # apply_rotary_pos_emb expects (batch, seq_len, n_heads, head_dim)
    q_t = q.transpose(1, 2)  # (batch, seq_len, n_heads, head_dim)
    k_t = k.transpose(1, 2)
    
    # Apply rotary embedding with unsqueeze_dim=2
    # (because n_heads is at dim 2 in [batch, seq_len, n_heads, head_dim])
    q_rot, k_rot = apply_rotary_pos_emb(q_t, k_t, cos, sin, unsqueeze_dim=2)
    
    # Transpose back to (batch, n_heads, seq_len, head_dim)
    q_rot = q_rot.transpose(1, 2)
    k_rot = k_rot.transpose(1, 2)
    
    # Use PyTorch's built-in scaled_dot_product_attention
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    output = F.scaled_dot_product_attention(
        q_rot, k_rot, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=sm_scale
    )
    
    return output


def baseline2_rope_flashattn(q, k, v, causal=True, sm_scale=None, theta=10000.0):
    """
    Baseline 2: Transformers RoPE + Flash Attention (Official CUDA)
    
    Args:
        q: (batch, n_heads, seq_len, head_dim)
        k: (batch, n_heads, seq_len, head_dim)
        v: (batch, n_heads, seq_len, head_dim)
        causal: bool, whether to use causal masking
        sm_scale: attention scale (if None, use 1/sqrt(head_dim))
        theta: RoPE base frequency
    
    Returns:
        output: (batch, n_heads, seq_len, head_dim)
    """
    if not HAS_FLASH_ATTN:
        raise ImportError("flash_attn is not installed. Please install it with: pip install flash-attn")
    
    batch, n_heads, seq_len, head_dim = q.shape
    device = q.device
    dtype = q.dtype
    
    # Compute RoPE cos/sin: (seq_len, head_dim)
    cos, sin = compute_rope_cos_sin(seq_len, head_dim, device, theta, dtype)
    
    # Expand to (batch, seq_len, head_dim) as required by apply_rotary_pos_emb
    cos = cos.unsqueeze(0).expand(batch, -1, -1)
    sin = sin.unsqueeze(0).expand(batch, -1, -1)
    
    # Apply RoPE to Q and K
    # apply_rotary_pos_emb expects (batch, seq_len, n_heads, head_dim)
    q_t = q.transpose(1, 2)  # (batch, seq_len, n_heads, head_dim)
    k_t = k.transpose(1, 2)
    
    # Apply rotary embedding with unsqueeze_dim=2
    q_rot, k_rot = apply_rotary_pos_emb(q_t, k_t, cos, sin, unsqueeze_dim=2)
    
    # Flash Attention expects (batch, seq_len, n_heads, head_dim)
    v_t = v.transpose(1, 2)  # (batch, seq_len, n_heads, head_dim)
    
    # Use Flash Attention (Official CUDA implementation)
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    output = flash_attn_func(
        q_rot, k_rot, v_t,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal
    )
    
    # Transpose back to (batch, n_heads, seq_len, head_dim)
    output = output.transpose(1, 2)
    
    return output


def baseline3_rope_flashattn_triton(q, k, v, causal=True, sm_scale=None, theta=10000.0):
    """
    Baseline 3: Transformers RoPE + Flash Attention v2 (Triton - 无RoPE融合)
    
    Args:
        q: (batch, n_heads, seq_len, head_dim)
        k: (batch, n_heads, seq_len, head_dim)
        v: (batch, n_heads, seq_len, head_dim)
        causal: bool, whether to use causal masking
        sm_scale: attention scale (if None, use 1/sqrt(head_dim))
        theta: RoPE base frequency
    
    Returns:
        output: (batch, n_heads, seq_len, head_dim)
    """
    batch, n_heads, seq_len, head_dim = q.shape
    device = q.device
    dtype = q.dtype
    
    # Compute RoPE cos/sin: (seq_len, head_dim)
    cos, sin = compute_rope_cos_sin(seq_len, head_dim, device, theta, dtype)
    
    # Expand to (batch, seq_len, head_dim) as required by apply_rotary_pos_emb
    cos = cos.unsqueeze(0).expand(batch, -1, -1)
    sin = sin.unsqueeze(0).expand(batch, -1, -1)
    
    # Apply RoPE to Q and K
    # apply_rotary_pos_emb expects (batch, seq_len, n_heads, head_dim)
    q_t = q.transpose(1, 2)  # (batch, seq_len, n_heads, head_dim)
    k_t = k.transpose(1, 2)
    
    # Apply rotary embedding with unsqueeze_dim=2
    q_rot, k_rot = apply_rotary_pos_emb(q_t, k_t, cos, sin, unsqueeze_dim=2)
    
    # Transpose back to (batch, n_heads, seq_len, head_dim) for Triton
    q_rot = q_rot.transpose(1, 2)
    k_rot = k_rot.transpose(1, 2)
    
    # Use Flash Attention v2 Triton implementation (without fused RoPE)
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Note: torch.autograd.Function.apply() only accepts positional arguments
    output = flash_attn_v2_triton(q_rot, k_rot, v, causal, sm_scale, False)
    
    return output


if __name__ == "__main__":
    # Simple test
    batch, n_heads, seq_len, head_dim = 2, 8, 512, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    q = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    print("Testing Baseline 1 (Transformers RoPE + PyTorch Attention)...")
    out1 = baseline1_rope_pytorch_attn(q, k, v, causal=True)
    print(f"✓ Output shape: {out1.shape}")
    
    if HAS_FLASH_ATTN:
        print("\nTesting Baseline 2 (Transformers RoPE + Flash Attention Official)...")
        out2 = baseline2_rope_flashattn(q, k, v, causal=True)
        print(f"✓ Output shape: {out2.shape}")
    else:
        print("\n✗ Flash Attention not available, skipping Baseline 2")
    
    print("\nTesting Baseline 3 (Transformers RoPE + Flash Attention v2 Triton)...")
    out3 = baseline3_rope_flashattn_triton(q, k, v, causal=True)
    print(f"✓ Output shape: {out3.shape}")
    
    # Compare baselines
    if HAS_FLASH_ATTN:
        print("\n" + "="*80)
        print("Comparing Baselines")
        print("="*80)
        
        print("\nBaseline 1 vs Baseline 2:")
        sim_12 = calc_sim(out1, out2, name="output")
        print(f"  Similarity: {sim_12:.8f}, Difference: {1.0 - sim_12:.8e}")
        
        print("\nBaseline 1 vs Baseline 3:")
        sim_13 = calc_sim(out1, out3, name="output")
        print(f"  Similarity: {sim_13:.8f}, Difference: {1.0 - sim_13:.8e}")
        
        print("\nBaseline 2 vs Baseline 3:")
        sim_23 = calc_sim(out2, out3, name="output")
        print(f"  Similarity: {sim_23:.8f}, Difference: {1.0 - sim_23:.8e}")
    else:
        print("\n" + "="*80)
        print("Comparing Baseline 1 vs Baseline 3")
        print("="*80)
        sim_13 = calc_sim(out1, out3, name="output")
        print(f"Similarity: {sim_13:.8f}, Difference: {1.0 - sim_13:.8e}")