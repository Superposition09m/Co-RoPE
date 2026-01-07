"""
简单的 Co-RoPE Forward Pass 测速对比
只测试小样本，用于快速验证
"""

import torch
import triton
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flash_attn_co_rope_gqa_triton import attention as attention_triton
from corope_attn_gqa_pytorch import attention_pytorch


def bench_fwd_simple():
    """简单的forward测速 - GQA版本"""
    print("=" * 60)
    print("Co-RoPE Forward Pass 简单测速 (GQA)")
    print("=" * 60)
    
    # GQA 配置：H_Q != H_KV
    B = 1
    H_Q = 4      # Query heads
    H_KV = 2     # Key/Value heads (GROUP_SIZE = 2)
    N = 4096
    D = 64
    causal = True
    sm_scale = 1.0 / (D ** 0.5)
    theta = 10000.0
    warp_specialize = False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    
    print(f"\n配置: B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
    print(f"Device: {device}, Dtype: {dtype}, GROUP_SIZE={H_Q // H_KV}")
    
    # 准备输入 - GQA 配置
    torch.manual_seed(42)
    q = torch.randn(B, H_Q, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H_KV, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H_KV, N, D, device=device, dtype=dtype)
    
    # Warmup
    print("\nWarmup...")
    for _ in range(5):
        _ = attention_pytorch(q, k, v, causal, sm_scale, theta)
        _ = attention_triton(q, k, v, causal, sm_scale, warp_specialize)
        if device == 'cuda':
            torch.cuda.synchronize()
    
    # PyTorch 测速 - 使用 triton.testing.do_bench（与原文件一致）
    print("\n[PyTorch] 测速中...")
    fn_pytorch = lambda: attention_pytorch(q, k, v, causal, sm_scale, theta)
    pytorch_time = triton.testing.do_bench(fn_pytorch)
    print(f"  Time: {pytorch_time:.3f} ms")
    
    # Triton 测速 - 使用 triton.testing.do_bench（与原文件一致，自动处理autotune）
    print("\n[Triton] 测速中...")
    fn_triton = lambda: attention_triton(q, k, v, causal, sm_scale, warp_specialize)
    triton_time = triton.testing.do_bench(fn_triton)
    print(f"  Time: {triton_time:.3f} ms")
    
    # 获取输出用于正确性检查
    out_pytorch = attention_pytorch(q, k, v, causal, sm_scale, theta)
    out_triton = attention_triton(q, k, v, causal, sm_scale, warp_specialize)
    
    # 结果对比
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    print(f"PyTorch: {pytorch_time:.3f} ms")
    print(f"Triton:  {triton_time:.3f} ms")
    
    if triton_time > 0:
        speedup = pytorch_time / triton_time
        print(f"Speedup: {speedup:.2f}x")
    
    # 简单正确性检查
    print("\n正确性检查...")
    diff = (out_pytorch - out_triton).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  Max Diff:  {max_diff:.6f}")
    print(f"  Mean Diff: {mean_diff:.6f}")
    
    if max_diff < 1e-2:
        print("  ✅ 输出基本一致")
    else:
        print("  ⚠️  输出差异较大，请检查实现")


if __name__ == "__main__":
    bench_fwd_simple()

