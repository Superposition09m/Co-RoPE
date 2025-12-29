"""
Flash Attention v2 原版 vs Fused RoPE 性能对比
测试三个版本的 FLOPS：
1. Flash Attention v2 原版（无 RoPE）
2. Fused RoPE v2（最小化修改版本）
3. Fused RoPE（过度优化版本）
"""

import torch
import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flash_attn_v2_triton import attention as attention_v2
from flash_attn_rope_v2_triton import attention as attention_rope_v2
from flash_attn_rope_triton import attention as attention_rope_opt
from utils import calc_sim, assert_similar, print_red_warning


def benchmark_kernel(fn, *args, warmup=3, repeat=10):
    """Benchmark a kernel with multiple runs"""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        fn(*args)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    # 返回中位数时间（更稳健）
    times.sort()
    median_time = times[len(times) // 2]
    min_time = min(times)
    max_time = max(times)
    
    return median_time, min_time, max_time, times


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


def test_performance(B, H, N, D, causal=False, mode='fwd', repeat=10):
    """性能测试三个版本"""
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
    print("\n  [1. Flash v2 原版]", end=' ')
    fn_v2 = lambda: attention_v2(q, k, v, causal, sm_scale, False)
    
    try:
        median, min_t, max_t, _ = benchmark_kernel(fn_v2, warmup=3, repeat=repeat)
        tflops = compute_flops(B, H, N, D, median, mode)
        results['v2'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['v2'] = None
    
    # 2. Fused RoPE v2（最小化修改）
    print("  [2. Fused RoPE v2]", end=' ')
    fn_rope_v2 = lambda: attention_rope_v2(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
    
    try:
        median, min_t, max_t, _ = benchmark_kernel(fn_rope_v2, warmup=3, repeat=repeat)
        tflops = compute_flops(B, H, N, D, median, mode)
        results['rope_v2'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['rope_v2'] = None
    
    # 3. Fused RoPE（过度优化）
    print("  [3. Fused RoPE Opt]", end=' ')
    fn_rope_opt = lambda: attention_rope_opt(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
    
    try:
        median, min_t, max_t, _ = benchmark_kernel(fn_rope_opt, warmup=3, repeat=repeat)
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
    print("第二步：性能对比（Forward Pass，每配置测试 10 次取中位数）")
    print("="*80)
    
    all_results = []
    
    for B, H, N, D, causal, name in configs:
        print(f"\n[配置: {name}] B={B}, H={H}, N={N}, D={D}")
        
        # 根据序列长度动态调整测试次数
        if N >= 262144:
            repeat = 1  # 256K+ 只测 1 次（太大了）
        elif N >= 131072:
            repeat = 2  # 128K 测 2 次
        elif N >= 65536:
            repeat = 2  # 64K 测 2 次
        elif N >= 32768:
            repeat = 3  # 32K 测 3 次
        elif N >= 8192:
            repeat = 5  # 8K-16K 测 5 次
        elif N >= 4096:
            repeat = 7  # 4K 测 7 次
        else:
            repeat = 10  # ≤2K 测 10 次
        
        print(f"  (测试 {repeat} 次取中位数)")
        
        try:
            results = test_performance(B, H, N, D, causal, mode='fwd', repeat=repeat)
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

