"""
TEST VERSION: Replace sin/cos with constants to diagnose performance bottleneck
This is a diagnostic version to test if MUFU (sin/cos) is the bottleneck
"""

# Copy the entire flash_attn_rope_triton.py and modify _compute_rope_freqs
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import everything from the original file
from flash_attn_rope_triton import *
import triton
import triton.language as tl

# Override the _compute_rope_freqs function with a constant version
@triton.jit
def _compute_rope_freqs_const(positions, theta, HEAD_DIM: tl.constexpr):
    """
    TEST VERSION: Return constant values instead of computing sin/cos
    This is to test if MUFU computation is the bottleneck
    """
    half_dim: tl.constexpr = HEAD_DIM // 2
    
    # Create constant arrays (all 1.0 for cos, all 0.0 for sin)
    # This eliminates MUFU computation completely
    const_val = tl.full([positions.shape[0], half_dim], 1.0, dtype=tl.float32)
    
    # Use split layout: concatenate using tl.join
    cos_joined = tl.join(const_val, const_val)  # [BLOCK, half_dim, 2]
    sin_joined = tl.join(const_val * 0.0, const_val * 0.0)  # All zeros
    
    # Permute to [BLOCK, 2, half_dim] and reshape to [BLOCK, HEAD_DIM]
    cos_vals = cos_joined.permute([0, 2, 1]).reshape([positions.shape[0], HEAD_DIM])
    sin_vals = sin_joined.permute([0, 2, 1]).reshape([positions.shape[0], HEAD_DIM])
    
    return cos_vals, sin_vals

# Override _apply_rope to use the constant version
@triton.jit
def _apply_rope_const(x, positions, theta, HEAD_DIM: tl.constexpr):
    """Apply RoPE with constant cos/sin (for testing)"""
    cos, sin = _compute_rope_freqs_const(positions, theta, HEAD_DIM)
    x_rot_half = _rotate_half(x, HEAD_DIM)
    return x * cos + x_rot_half * sin

# We need to redefine the kernels to use the constant version
# This is getting complex, let me create a simpler benchmark instead

