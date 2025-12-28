"""
Performance Bottleneck Diagnosis
================================

Test 1: Replace sin/cos with constants to see if MUFU is the bottleneck
Test 2: Check register pressure
Test 3: Profile memory access patterns
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _rope_compute_real(positions, theta, HEAD_DIM: tl.constexpr):
    """Real RoPE computation with sin/cos"""
    half_dim: tl.constexpr = HEAD_DIM // 2
    i = tl.arange(0, half_dim).to(tl.float32)
    exponent = 2.0 * i / HEAD_DIM
    log_theta = tl.log(theta)
    freqs = 1.0 / tl.exp(log_theta * exponent)
    angles = positions[:, None].to(tl.float32) * freqs[None, :]
    
    # THE BOTTLENECK: tl.cos and tl.sin
    cos_half = tl.cos(angles)
    sin_half = tl.sin(angles)
    
    cos_joined = tl.join(cos_half, cos_half)
    sin_joined = tl.join(sin_half, sin_half)
    cos_vals = cos_joined.permute([0, 2, 1]).reshape([positions.shape[0], HEAD_DIM])
    sin_vals = sin_joined.permute([0, 2, 1]).reshape([positions.shape[0], HEAD_DIM])
    return cos_vals, sin_vals

@triton.jit
def _rope_compute_const(positions, theta, HEAD_DIM: tl.constexpr):
    """Fake RoPE computation with constants (no sin/cos)"""
    half_dim: tl.constexpr = HEAD_DIM // 2
    
    # NO MUFU OPERATIONS - just create constants
    cos_half = tl.full([positions.shape[0], half_dim], 1.0, dtype=tl.float32)
    sin_half = tl.full([positions.shape[0], half_dim], 0.0, dtype=tl.float32)
    
    cos_joined = tl.join(cos_half, cos_half)
    sin_joined = tl.join(sin_half, sin_half)
    cos_vals = cos_joined.permute([0, 2, 1]).reshape([positions.shape[0], HEAD_DIM])
    sin_vals = sin_joined.permute([0, 2, 1]).reshape([positions.shape[0], HEAD_DIM])
    return cos_vals, sin_vals

@triton.jit
def test_kernel_real(output, positions, theta: float, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr):
    """Test kernel with REAL sin/cos computation"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pos = tl.load(positions + offs)
    
    # Compute RoPE freqs with REAL sin/cos
    cos_vals, sin_vals = _rope_compute_real(pos, theta, HEAD_DIM)
    
    # Dummy computation to prevent optimization
    result = cos_vals + sin_vals
    result_sum = tl.sum(result, axis=1)
    tl.store(output + offs, result_sum)

@triton.jit  
def test_kernel_const(output, positions, theta: float, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr):
    """Test kernel with CONSTANT sin/cos (no MUFU)"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pos = tl.load(positions + offs)
    
    # Compute RoPE freqs with CONSTANTS (no sin/cos)
    cos_vals, sin_vals = _rope_compute_const(pos, theta, HEAD_DIM)
    
    # Dummy computation to prevent optimization
    result = cos_vals + sin_vals
    result_sum = tl.sum(result, axis=1)
    tl.store(output + offs, result_sum)

def benchmark_mufu_impact():
    """Benchmark to isolate MUFU impact"""
    
    SEQ_LEN = 2048
    HEAD_DIM = 64
    BLOCK_SIZE = 128
    THETA = 10000.0
    device = 'cuda'
    
    # Prepare inputs
    positions = torch.arange(SEQ_LEN, device=device, dtype=torch.int32)
    output = torch.zeros(SEQ_LEN, device=device, dtype=torch.float32)
    
    grid = (triton.cdiv(SEQ_LEN, BLOCK_SIZE),)
    
    print("\n" + "="*80)
    print("MUFU Bottleneck Diagnosis")
    print("="*80)
    print(f"Configuration: SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}, BLOCK_SIZE={BLOCK_SIZE}")
    print()
    
    # Benchmark REAL version (with sin/cos)
    fn_real = lambda: test_kernel_real[grid](output, positions, THETA, BLOCK_SIZE, HEAD_DIM)
    ms_real = triton.testing.do_bench(fn_real, warmup=25, rep=100)
    
    # Benchmark CONST version (without sin/cos)
    fn_const = lambda: test_kernel_const[grid](output, positions, THETA, BLOCK_SIZE, HEAD_DIM)
    ms_const = triton.testing.do_bench(fn_const, warmup=25, rep=100)
    
    # Calculate overhead
    overhead_pct = ((ms_real - ms_const) / ms_const) * 100
    slowdown = ms_real / ms_const
    
    print("Results:")
    print(f"  With REAL sin/cos (MUFU):     {ms_real:.4f} ms")
    print(f"  With CONST (no MUFU):          {ms_const:.4f} ms")
    print(f"  MUFU Overhead:                 {overhead_pct:+.1f}%")
    print(f"  Slowdown Factor:               {slowdown:.2f}x")
    print()
    
    if overhead_pct > 200:
        print("  🔴 DIAGNOSIS: MUFU is THE BOTTLENECK!")
        print("     → Solution: Pre-compute sin/cos cache and load from memory")
        print("     → Expected speedup: ~3-5x")
    elif overhead_pct > 50:
        print("  🟡 DIAGNOSIS: MUFU contributes significantly")
        print("     → Solution: Consider caching frequently used values")
    else:
        print("  🟢 DIAGNOSIS: MUFU is NOT the bottleneck")
        print("     → Look for register spilling or memory access issues")
    
    print("="*80 + "\n")
    
    return ms_real, ms_const, overhead_pct

if __name__ == "__main__":
    benchmark_mufu_impact()

