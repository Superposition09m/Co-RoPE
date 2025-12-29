"""
Flash Attention v2 原版 vs Fused RoPE 性能对比
测试三个版本的 FLOPS：
1. Flash Attention v2 原版（无 RoPE）
2. Fused RoPE v2（最小化修改版本）
3. Fused RoPE（过度优化版本）

⚠️ 重要：使用 Triton 官方的 triton.testing.do_bench 进行 benchmark
   - 正确处理 autotune（第一次调用时完成配置选择）
   - 充分的 warmup 确保性能稳定
   - 返回可靠的中位数时间
"""

import torch
import time
import sys
import os
import triton

sys.path.insert(0, os.path.dirname(__file__))

from flash_attn_v2_triton import attention as attention_v2
from flash_attn_rope_v2_triton import attention as attention_rope_v2
from flash_attn_rope_triton import attention as attention_rope_opt
from utils import calc_sim, assert_similar, print_red_warning


def benchmark_kernel(fn, warmup=25, rep=100):
    """
    Benchmark a kernel with Triton's do_bench (handles autotune correctly)
    
    Args:
        fn: Function to benchmark (no-arg lambda)
        warmup: Number of warmup iterations (default: 25)
        rep: Number of measurement iterations (default: 100)
    
    Returns:
        median_time (ms), min_time (ms), max_time (ms)
    """
    # 使用 Triton 官方的 do_bench，它会：
    # 1. 自动处理 autotune（第一次调用时完成）
    # 2. 充分的 warmup
    # 3. 返回稳定的中位数时间
    median_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.0, 1.0])
    
    # do_bench 返回 [median, min, max] 如果指定了 quantiles=[0.5, 0.0, 1.0]
    # 否则只返回 median
    if isinstance(median_ms, list) or isinstance(median_ms, tuple):
        median_time, min_time, max_time = median_ms[0], median_ms[1], median_ms[2]
    else:
        # 如果只返回了 median，min/max 设为 median
        median_time = median_ms
        min_time = median_ms
        max_time = median_ms
    
    return median_time, min_time, max_time


def compute_flops(B, H, N, D, time_ms, mode='fwd'):
    """计算 FLOPS"""
    # Forward: 2 次矩阵乘法，每次 2*B*H*N*N*D FLOPs
    flops_per_matmul = 2.0 * B * H * N * N * D
    total_flops = 2 * flops_per_matmul
    
    if mode == 'bwd':
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    
    tflops = total_flops * 1e-12 / (time_ms * 1e-3)
    return tflops


def test_correctness(B, H, N, D, causal=False):
    """测试正确性（仅 Forward）"""
    print(f"\n{'='*80}")
    print(f"正确性测试 (Forward Only): B={B}, H={H}, N={N}, D={D}, causal={causal}")
    print(f"{'='*80}")
    
    device = 'cuda'
    dtype = torch.float16
    
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    freqs_cos = torch.randn(N, D // 2, device=device, dtype=dtype)
    freqs_sin = torch.randn(N, D // 2, device=device, dtype=dtype)
    sm_scale = 0.5
    
    # Flash v2 原版（不带 RoPE）
    print("\n[1. Flash v2 原版]")
    o_v2 = attention_v2(q, k, v, causal, sm_scale, False)
    print(f"  Output: mean={o_v2.mean().item():.6f}, std={o_v2.std().item():.6f}")
    
    # Fused RoPE v2（最小化修改版本）
    print("\n[2. Fused RoPE v2 (Minimal Change)]")
    try:
        o_rope_v2 = attention_rope_v2(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
        if torch.isnan(o_rope_v2).any() or torch.isinf(o_rope_v2).any():
            print(f"  ❌ 输出包含 NaN/Inf")
            return False
        print(f"  ✅ Output: mean={o_rope_v2.mean().item():.6f}, std={o_rope_v2.std().item():.6f}")
    except Exception as e:
        print(f"  ❌ 失败: {str(e)[:100]}")
        return False
    
    # Fused RoPE（过度优化版本）
    print("\n[3. Fused RoPE (Over-Optimized)]")
    try:
        o_rope_opt = attention_rope_opt(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
        if torch.isnan(o_rope_opt).any() or torch.isinf(o_rope_opt).any():
            print(f"  ❌ 输出包含 NaN/Inf")
            return False
        print(f"  ✅ Output: mean={o_rope_opt.mean().item():.6f}, std={o_rope_opt.std().item():.6f}")
    except Exception as e:
        print(f"  ❌ 失败: {str(e)[:100]}")
        return False
    
    print("\n✅ 所有版本数值稳定性验证通过（无 NaN/Inf）")
    return True


def test_performance(B, H, N, D, causal=False, mode='fwd', warmup=25, rep=100):
    """
    性能测试三个版本（使用 Triton 官方 benchmark，正确处理 autotune）
    
    Args:
        warmup: Warmup iterations（默认 25，确保 autotune 完成）
        rep: Measurement iterations（默认 100）
    """
    device = 'cuda'
    dtype = torch.float16
    
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    freqs_cos = torch.randn(N, D // 2, device=device, dtype=dtype)
    freqs_sin = torch.randn(N, D // 2, device=device, dtype=dtype)
    sm_scale = 0.5
    
    results = {}
    
    # 1. Flash v2 原版
    print("\n  [1. Flash v2 原版]", end=' ', flush=True)
    fn_v2 = lambda: attention_v2(q, k, v, causal, sm_scale, False)
    
    try:
        median, min_t, max_t = benchmark_kernel(fn_v2, warmup=warmup, rep=rep)
        tflops = compute_flops(B, H, N, D, median, mode)
        results['v2'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['v2'] = None
    
    # 2. Fused RoPE v2（最小化修改）
    print("  [2. Fused RoPE v2]", end=' ', flush=True)
    fn_rope_v2 = lambda: attention_rope_v2(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
    
    try:
        median, min_t, max_t = benchmark_kernel(fn_rope_v2, warmup=warmup, rep=rep)
        tflops = compute_flops(B, H, N, D, median, mode)
        results['rope_v2'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['rope_v2'] = None
    
    # 3. Fused RoPE（过度优化）
    print("  [3. Fused RoPE Opt]", end=' ', flush=True)
    fn_rope_opt = lambda: attention_rope_opt(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
    
    try:
        median, min_t, max_t = benchmark_kernel(fn_rope_opt, warmup=warmup, rep=rep)
        tflops = compute_flops(B, H, N, D, median, mode)
        results['rope_opt'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['rope_opt'] = None
    
    return results


def main():
    print("="*80)
    print("三版本性能对比: Flash v2 | RoPE v2 (Minimal) | RoPE Opt")
    print("="*80)
    
    # 测试配置：(BATCH, H, N_CTX, HEAD_DIM, causal, name)
    configs = [
        # D=64 配置 - 从 512 到 512K
        (2, 8, 512, 64, False, "512-D64"),
        (2, 8, 1024, 64, False, "1K-D64"),
        (2, 8, 2048, 64, False, "2K-D64"),
        (2, 8, 4096, 64, False, "4K-D64"),
        (1, 8, 8192, 64, False, "8K-D64"),
        (1, 8, 16384, 64, False, "16K-D64"),
        (1, 8, 32768, 64, False, "32K-D64"),
        (1, 4, 65536, 64, False, "64K-D64"),
        (1, 2, 131072, 64, False, "128K-D64"),
        (1, 2, 262144, 64, False, "256K-D64"),
        (1, 1, 524288, 64, False, "512K-D64"),
        # D=128 配置 - 从 512 到 256K
        (2, 8, 512, 128, False, "512-D128"),
        (2, 8, 1024, 128, False, "1K-D128"),
        (2, 8, 2048, 128, False, "2K-D128"),
        (2, 8, 4096, 128, False, "4K-D128"),
        (1, 8, 8192, 128, False, "8K-D128"),
        (1, 8, 16384, 128, False, "16K-D128"),
        (1, 4, 32768, 128, False, "32K-D128"),
        (1, 2, 65536, 128, False, "64K-D128"),
        (1, 2, 131072, 128, False, "128K-D128"),
        (1, 1, 262144, 128, False, "256K-D128"),
    ]
    
    # 正确性测试
    print("\n" + "="*80)
    print("第一步：正确性验证")
    print("="*80)
    correctness_passed = test_correctness(1, 2, 128, 64, causal=False)
    
    if not correctness_passed:
        print("\n⚠️  正确性测试未通过，停止性能测试")
        return
    
    # 性能测试
    print("\n" + "="*80)
    print("第二步：性能对比（Forward Pass，使用 Triton do_bench，正确处理 autotune）")
    print("="*80)
    
    all_results = []
    
    for B, H, N, D, causal, name in configs:
        print(f"\n[配置: {name}] B={B}, H={H}, N={N}, D={D}")
        
        # 根据序列长度动态调整测试次数（使用 Triton benchmark）
        # warmup: 确保 autotune 完成
        # rep: 测量次数
        if N >= 262144:
            warmup, rep = 10, 20   # 256K+ 测试少一点
        elif N >= 131072:
            warmup, rep = 15, 30   # 128K
        elif N >= 65536:
            warmup, rep = 20, 50   # 64K
        elif N >= 32768:
            warmup, rep = 25, 75   # 32K
        elif N >= 8192:
            warmup, rep = 25, 100  # 8K-16K
        else:
            warmup, rep = 25, 100  # ≤4K 标准测试
        
        print(f"  (Warmup={warmup}, Rep={rep}, 使用 Triton do_bench)")
        
        try:
            results = test_performance(B, H, N, D, causal, mode='fwd', warmup=warmup, rep=rep)
            results['config'] = name
            all_results.append(results)
        except Exception as e:
            print(f"  ❌ 失败: {e}")
    
    # 总结
    print("\n" + "="*100)
    print("性能对比总结 (Forward Pass)")
    print("="*100)
    print(f"{'配置':<10} | {'v2 Time':<10} | {'RoPE v2':<10} | {'RoPE Opt':<11} | {'v2 TFLOPS':<10} | {'RoPEv2 TF':<10} | {'Opt TF':<10}")
    print("-"*100)
    
    for result in all_results:
        if result.get('v2'):
            name = result['config']
            v2_t = result['v2']['median']
            v2_tflops = result['v2']['tflops']
            
            rope_v2_t = result.get('rope_v2', {}).get('median', 0)
            rope_v2_tflops = result.get('rope_v2', {}).get('tflops', 0)
            
            rope_opt_t = result.get('rope_opt', {}).get('median', 0)
            rope_opt_tflops = result.get('rope_opt', {}).get('tflops', 0)
            
            v2_str = f"{v2_t:.2f}ms" if v2_t > 0 else "N/A"
            v2_rope_str = f"{rope_v2_t:.2f}ms" if rope_v2_t > 0 else "N/A"
            opt_str = f"{rope_opt_t:.2f}ms" if rope_opt_t > 0 else "N/A"
            
            print(f"{name:<10} | {v2_str:<10} | {v2_rope_str:<10} | {opt_str:<11} | {v2_tflops:>9.2f}  | {rope_v2_tflops:>9.2f}  | {rope_opt_tflops:>9.2f}")
    
    print("="*100)
    
    # 性能差异分析
    print("\n" + "="*100)
    print("性能差异分析（相对于 Flash v2 原版）")
    print("="*100)
    print(f"{'配置':<10} | {'RoPE v2 Speedup':<18} | {'RoPE Opt Speedup':<19} | {'RoPE v2 TFLOPS Δ':<18} | {'Opt TFLOPS Δ'}")
    print("-"*100)
    
    for result in all_results:
        if result.get('v2'):
            name = result['config']
            v2_t = result['v2']['median']
            v2_tflops = result['v2']['tflops']
            
            rope_v2_data = result.get('rope_v2', {})
            rope_opt_data = result.get('rope_opt', {})
            
            if rope_v2_data and rope_v2_data.get('median'):
                speedup_v2 = v2_t / rope_v2_data['median']
                tflops_diff_v2 = rope_v2_data['tflops'] - v2_tflops
                speedup_v2_str = f"{speedup_v2:.3f}x {'↑' if speedup_v2 > 1 else '↓'}"
                tflops_v2_str = f"{tflops_diff_v2:+.2f}"
            else:
                speedup_v2_str = "N/A"
                tflops_v2_str = "N/A"
            
            if rope_opt_data and rope_opt_data.get('median'):
                speedup_opt = v2_t / rope_opt_data['median']
                tflops_diff_opt = rope_opt_data['tflops'] - v2_tflops
                speedup_opt_str = f"{speedup_opt:.3f}x {'↑' if speedup_opt > 1 else '↓'}"
                tflops_opt_str = f"{tflops_diff_opt:+.2f}"
            else:
                speedup_opt_str = "N/A"
                tflops_opt_str = "N/A"
            
            print(f"{name:<10} | {speedup_v2_str:<18} | {speedup_opt_str:<19} | {tflops_v2_str:<18} | {tflops_opt_str}")
    
    print("="*100)


if __name__ == "__main__":
    main()

