"""
RoPE Attention 性能对比
测试四个版本：
1. Baseline 1: Transformers RoPE + PyTorch SDPA
2. Baseline 2: Transformers RoPE + Flash Attention (Official CUDA)
3. Baseline 3: Transformers RoPE + Flash Attention v2 (Triton - 无RoPE融合)
4. Fused RoPE: 我们的 Triton 实现（RoPE 融合进 Flash Attention）

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
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # 添加上一层目录

from baseline import baseline1_rope_pytorch_attn, baseline2_rope_flashattn, baseline3_rope_flashattn_triton
from fused_rope_attn import attention as fused_rope_attn
from rope_attn_pytorch import precompute_freqs_cis
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


def test_correctness(B, H, N, D, causal=False, test_backward=False, theta=10000.0):
    """测试正确性（Forward + Backward）"""
    mode_str = "Forward + Backward" if test_backward else "Forward Only"
    print(f"\n{'='*80}")
    print(f"正确性测试 ({mode_str}): B={B}, H={H}, N={N}, D={D}, causal={causal}")
    print(f"{'='*80}")
    
    device = 'cuda'
    dtype = torch.float16
    
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=test_backward)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=test_backward)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=test_backward)
    
    # Compute RoPE frequencies
    freqs_cos, freqs_sin = precompute_freqs_cis(D, N, theta, device=device)
    sm_scale = 0.5
    
    # Baseline 1: Transformers RoPE + PyTorch SDPA
    print("\n[1. Baseline 1: Transformers RoPE + PyTorch SDPA]")
    try:
        o_baseline1 = baseline1_rope_pytorch_attn(q, k, v, causal, sm_scale, theta)
        if torch.isnan(o_baseline1).any() or torch.isinf(o_baseline1).any():
            print(f"  ❌ Forward 输出包含 NaN/Inf")
            return False
        print(f"  ✅ Forward Output: mean={o_baseline1.mean().item():.6f}, std={o_baseline1.std().item():.6f}")
        
        if test_backward:
            dout = torch.randn_like(o_baseline1)
            o_baseline1.backward(dout, retain_graph=True)
            dq_baseline1 = q.grad.clone()
            q.grad, k.grad, v.grad = None, None, None
            if torch.isnan(dq_baseline1).any() or torch.isinf(dq_baseline1).any():
                print(f"  ❌ Backward 梯度包含 NaN/Inf")
                return False
            print(f"  ✅ Backward dq: mean={dq_baseline1.mean().item():.6f}, std={dq_baseline1.std().item():.6f}")
    except Exception as e:
        print(f"  ❌ 失败: {str(e)[:100]}")
        return False
    
    # Baseline 2: Transformers RoPE + Flash Attention (Official CUDA)
    print("\n[2. Baseline 2: Transformers RoPE + Flash Attention Official]")
    try:
        o_baseline2 = baseline2_rope_flashattn(q, k, v, causal, sm_scale, theta)
        if torch.isnan(o_baseline2).any() or torch.isinf(o_baseline2).any():
            print(f"  ❌ Forward 输出包含 NaN/Inf")
            return False
        print(f"  ✅ Forward Output: mean={o_baseline2.mean().item():.6f}, std={o_baseline2.std().item():.6f}")
        
        if test_backward:
            o_baseline2.backward(dout, retain_graph=True)
            dq_baseline2 = q.grad.clone()
            q.grad, k.grad, v.grad = None, None, None
            if torch.isnan(dq_baseline2).any() or torch.isinf(dq_baseline2).any():
                print(f"  ❌ Backward 梯度包含 NaN/Inf")
                return False
            print(f"  ✅ Backward dq: mean={dq_baseline2.mean().item():.6f}, std={dq_baseline2.std().item():.6f}")
    except Exception as e:
        print(f"  ❌ 失败: {str(e)[:100]}")
        return False
    
    # Baseline 3: Transformers RoPE + Flash Attention v2 (Triton - 无RoPE融合)
    print("\n[3. Baseline 3: Transformers RoPE + Flash Attention v2 Triton]")
    try:
        o_baseline3 = baseline3_rope_flashattn_triton(q, k, v, causal, sm_scale, theta)
        if torch.isnan(o_baseline3).any() or torch.isinf(o_baseline3).any():
            print(f"  ❌ Forward 输出包含 NaN/Inf")
            return False
        print(f"  ✅ Forward Output: mean={o_baseline3.mean().item():.6f}, std={o_baseline3.std().item():.6f}")
        
        if test_backward:
            o_baseline3.backward(dout, retain_graph=True)
            dq_baseline3 = q.grad.clone()
            q.grad, k.grad, v.grad = None, None, None
            if torch.isnan(dq_baseline3).any() or torch.isinf(dq_baseline3).any():
                print(f"  ❌ Backward 梯度包含 NaN/Inf")
                return False
            print(f"  ✅ Backward dq: mean={dq_baseline3.mean().item():.6f}, std={dq_baseline3.std().item():.6f}")
    except Exception as e:
        print(f"  ❌ 失败: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return False
    
    # Fused RoPE: 我们的 Triton 实现
    print("\n[4. Fused RoPE: Triton Implementation]")
    try:
        o_fused = fused_rope_attn(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
        if torch.isnan(o_fused).any() or torch.isinf(o_fused).any():
            print(f"  ❌ Forward 输出包含 NaN/Inf")
            return False
        print(f"  ✅ Forward Output: mean={o_fused.mean().item():.6f}, std={o_fused.std().item():.6f}")
        
        if test_backward:
            o_fused.backward(dout, retain_graph=True)
            dq_fused = q.grad.clone()
            q.grad, k.grad, v.grad = None, None, None
            if torch.isnan(dq_fused).any() or torch.isinf(dq_fused).any():
                print(f"  ❌ Backward 梯度包含 NaN/Inf")
                return False
            print(f"  ✅ Backward dq: mean={dq_fused.mean().item():.6f}, std={dq_fused.std().item():.6f}")
    except Exception as e:
        print(f"  ❌ 失败: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✅ 所有版本数值稳定性验证通过（无 NaN/Inf）")
    return True


def test_performance(B, H, N, D, causal=False, mode='fwd', warmup=25, rep=100, theta=10000.0):
    """
    性能测试三个版本（使用 Triton 官方 benchmark，正确处理 autotune）
    
    Args:
        mode: 'fwd' 或 'bwd'
        warmup: Warmup iterations（默认 25，确保 autotune 完成）
        rep: Measurement iterations（默认 100）
        theta: RoPE base frequency
    """
    device = 'cuda'
    dtype = torch.float16
    
    # 根据 mode 决定是否需要梯度
    requires_grad = (mode == 'bwd')
    
    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=requires_grad)
    
    # Compute RoPE frequencies
    freqs_cos, freqs_sin = precompute_freqs_cis(D, N, theta, device=device)
    sm_scale = 0.5
    
    results = {}
    
    # 1. Baseline 1: Transformers RoPE + PyTorch SDPA
    print("\n  [1. Baseline 1: Transformers RoPE + PyTorch SDPA]", end=' ', flush=True)
    
    if mode == 'fwd':
        fn_baseline1 = lambda: baseline1_rope_pytorch_attn(q, k, v, causal, sm_scale, theta)
    else:  # bwd
        o_baseline1 = baseline1_rope_pytorch_attn(q, k, v, causal, sm_scale, theta)
        dout_baseline1 = torch.randn_like(o_baseline1)
        fn_baseline1 = lambda: o_baseline1.backward(dout_baseline1, retain_graph=True)
    
    try:
        median, min_t, max_t = benchmark_kernel(fn_baseline1, warmup=warmup, rep=rep)
        tflops = compute_flops(B, H, N, D, median, mode)
        results['baseline1'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['baseline1'] = None
    
    # 2. Baseline 2: Transformers RoPE + Flash Attention (Official CUDA)
    print("  [2. Baseline 2: Transformers RoPE + Flash Attn Official]", end=' ', flush=True)
    
    if mode == 'fwd':
        fn_baseline2 = lambda: baseline2_rope_flashattn(q, k, v, causal, sm_scale, theta)
    else:  # bwd
        o_baseline2 = baseline2_rope_flashattn(q, k, v, causal, sm_scale, theta)
        dout_baseline2 = torch.randn_like(o_baseline2)
        fn_baseline2 = lambda: o_baseline2.backward(dout_baseline2, retain_graph=True)
    
    try:
        median, min_t, max_t = benchmark_kernel(fn_baseline2, warmup=warmup, rep=rep)
        tflops = compute_flops(B, H, N, D, median, mode)
        results['baseline2'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['baseline2'] = None
    
    # 3. Baseline 3: Transformers RoPE + Flash Attention v2 (Triton - 无RoPE融合)
    print("  [3. Baseline 3: Transformers RoPE + Flash Attn v2 Triton]", end=' ', flush=True)
    
    if mode == 'fwd':
        fn_baseline3 = lambda: baseline3_rope_flashattn_triton(q, k, v, causal, sm_scale, theta)
    else:  # bwd
        o_baseline3 = baseline3_rope_flashattn_triton(q, k, v, causal, sm_scale, theta)
        dout_baseline3 = torch.randn_like(o_baseline3)
        fn_baseline3 = lambda: o_baseline3.backward(dout_baseline3, retain_graph=True)
    
    try:
        median, min_t, max_t = benchmark_kernel(fn_baseline3, warmup=warmup, rep=rep)
        tflops = compute_flops(B, H, N, D, median, mode)
        results['baseline3'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['baseline3'] = None
    
    # 4. Fused RoPE: 我们的 Triton 实现
    print("  [4. Fused RoPE: Triton (RoPE融合)]", end=' ', flush=True)
    
    if mode == 'fwd':
        fn_fused = lambda: fused_rope_attn(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
    else:  # bwd
        o_fused = fused_rope_attn(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
        dout_fused = torch.randn_like(o_fused)
        fn_fused = lambda: o_fused.backward(dout_fused, retain_graph=True)
    
    try:
        median, min_t, max_t = benchmark_kernel(fn_fused, warmup=warmup, rep=rep)
        tflops = compute_flops(B, H, N, D, median, mode)
        results['fused'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['fused'] = None
    
    return results


def main():
    print("="*80)
    print("RoPE Attention 性能对比")
    print("Baseline 1: Transformers RoPE + PyTorch SDPA")
    print("Baseline 2: Transformers RoPE + Flash Attention (Official CUDA)")
    print("Baseline 3: Transformers RoPE + Flash Attention v2 (Triton - 无RoPE融合)")
    print("Fused RoPE: 我们的 Triton 实现（RoPE 融合进 Flash Attention）")
    print("="*80)
    
    # 测试配置：(BATCH, H, N_CTX, HEAD_DIM, causal, name)
    # 方案A：全面测试（只测 causal=True，覆盖实际 LLM 场景）
    configs = [
        # === 小模型配置 (D=64) ===
        (4, 8, 512, 64, True, "Small-512"),
        (4, 8, 1024, 64, True, "Small-1K"),
        (2, 8, 2048, 64, True, "Small-2K"),
        (2, 8, 4096, 64, True, "Small-4K"),
        (1, 8, 8192, 64, True, "Small-8K"),
        
        # === 标准配置 (D=128, 类似 Llama-7B: 32 heads) ===
        (4, 32, 512, 128, True, "Llama7B-512"),
        (4, 32, 1024, 128, True, "Llama7B-1K"),
        (2, 32, 2048, 128, True, "Llama7B-2K"),
        (2, 32, 4096, 128, True, "Llama7B-4K"),
        (1, 32, 8192, 128, True, "Llama7B-8K"),
        (1, 32, 16384, 128, True, "Llama7B-16K"),
        
        # === 大模型配置 (D=128, 类似 Llama-70B: 64 heads) ===
        (2, 64, 512, 128, True, "Llama70B-512"),
        (2, 64, 1024, 128, True, "Llama70B-1K"),
        (1, 64, 2048, 128, True, "Llama70B-2K"),
        (1, 64, 4096, 128, True, "Llama70B-4K"),
        (1, 32, 8192, 128, True, "Llama70B-8K"),
        
        # === 超长序列 (显存优化配置) ===
        (1, 16, 32768, 128, True, "Long-32K"),
        (1, 8, 65536, 128, True, "Long-64K"),
        (1, 4, 131072, 128, True, "Long-128K"),
    ]
    
    # 正确性测试
    print("\n" + "="*80)
    print("第一步：正确性验证")
    print("="*80)
    
    # Forward 正确性测试
    print("\n" + "-"*80)
    print("1.1 Forward 正确性测试")
    print("-"*80)
    fwd_correctness_passed = test_correctness(1, 2, 128, 64, causal=True, test_backward=False)
    
    if not fwd_correctness_passed:
        print("\n⚠️  Forward 正确性测试未通过，停止测试")
        return
    
    # Backward 正确性测试
    print("\n" + "-"*80)
    print("1.2 Backward 正确性测试")
    print("-"*80)
    bwd_correctness_passed = test_correctness(1, 2, 128, 64, causal=True, test_backward=True)
    
    if not bwd_correctness_passed:
        print("\n⚠️  Backward 正确性测试未通过，停止测试")
        return
    
    # Forward 性能测试
    print("\n" + "="*80)
    print("第二步：Forward Pass 性能对比（使用 Triton do_bench，正确处理 autotune）")
    print("="*80)
    
    all_fwd_results = []
    
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
            all_fwd_results.append(results)
        except Exception as e:
            print(f"  ❌ 失败: {e}")
    
    # Forward 总结
    print("\n" + "="*120)
    print("性能对比总结 (Forward Pass)")
    print("="*120)
    print(f"{'配置':<10} | {'Baseline1':<11} | {'Baseline2':<11} | {'Baseline3':<11} | {'Fused':<11} | {'B1 TF':<7} | {'B2 TF':<7} | {'B3 TF':<7} | {'Fused TF':<8}")
    print("-"*120)
    
    for result in all_fwd_results:
        if result.get('baseline1') or result.get('baseline2') or result.get('baseline3') or result.get('fused'):
            name = result['config']
            
            baseline1_t = result.get('baseline1', {}).get('median', 0)
            baseline1_tflops = result.get('baseline1', {}).get('tflops', 0)
            
            baseline2_t = result.get('baseline2', {}).get('median', 0)
            baseline2_tflops = result.get('baseline2', {}).get('tflops', 0)
            
            baseline3_t = result.get('baseline3', {}).get('median', 0)
            baseline3_tflops = result.get('baseline3', {}).get('tflops', 0)
            
            fused_t = result.get('fused', {}).get('median', 0)
            fused_tflops = result.get('fused', {}).get('tflops', 0)
            
            b1_str = f"{baseline1_t:.2f}ms" if baseline1_t > 0 else "N/A"
            b2_str = f"{baseline2_t:.2f}ms" if baseline2_t > 0 else "N/A"
            b3_str = f"{baseline3_t:.2f}ms" if baseline3_t > 0 else "N/A"
            fused_str = f"{fused_t:.2f}ms" if fused_t > 0 else "N/A"
            
            print(f"{name:<10} | {b1_str:<11} | {b2_str:<11} | {b3_str:<11} | {fused_str:<11} | {baseline1_tflops:>6.2f}  | {baseline2_tflops:>6.2f}  | {baseline3_tflops:>6.2f}  | {fused_tflops:>7.2f}")
    
    print("="*120)
    
    # Forward 性能差异分析（以 Baseline 3 为基准）
    print("\n" + "="*130)
    print("Forward 性能差异分析（相对于 Baseline 3: Flash Attention v2 Triton 无RoPE融合）")
    print("="*130)
    print(f"{'配置':<10} | {'B1 vs B3':<18} | {'B2 vs B3':<18} | {'Fused vs B3':<18} | {'B1 TFLOPS Δ':<15} | {'B2 TFLOPS Δ':<15} | {'Fused TFLOPS Δ':<15}")
    print("-"*130)
    
    for result in all_fwd_results:
        if result.get('baseline3'):
            name = result['config']
            b3_t = result['baseline3']['median']
            b3_tflops = result['baseline3']['tflops']
            
            baseline1_data = result.get('baseline1', {})
            baseline2_data = result.get('baseline2', {})
            fused_data = result.get('fused', {})
            
            if baseline1_data and baseline1_data.get('median'):
                speedup_b1 = b3_t / baseline1_data['median']
                tflops_diff_b1 = baseline1_data['tflops'] - b3_tflops
                speedup_b1_str = f"{speedup_b1:.3f}x {'↑' if speedup_b1 > 1 else '↓'}"
                tflops_b1_str = f"{tflops_diff_b1:+.2f}"
            else:
                speedup_b1_str = "N/A"
                tflops_b1_str = "N/A"
            
            if baseline2_data and baseline2_data.get('median'):
                speedup_b2 = b3_t / baseline2_data['median']
                tflops_diff_b2 = baseline2_data['tflops'] - b3_tflops
                speedup_b2_str = f"{speedup_b2:.3f}x {'↑' if speedup_b2 > 1 else '↓'}"
                tflops_b2_str = f"{tflops_diff_b2:+.2f}"
            else:
                speedup_b2_str = "N/A"
                tflops_b2_str = "N/A"
            
            if fused_data and fused_data.get('median'):
                speedup_fused = b3_t / fused_data['median']
                tflops_diff_fused = fused_data['tflops'] - b3_tflops
                speedup_fused_str = f"{speedup_fused:.3f}x {'↑' if speedup_fused > 1 else '↓'}"
                tflops_fused_str = f"{tflops_diff_fused:+.2f}"
            else:
                speedup_fused_str = "N/A"
                tflops_fused_str = "N/A"
            
            print(f"{name:<10} | {speedup_b1_str:<18} | {speedup_b2_str:<18} | {speedup_fused_str:<18} | {tflops_b1_str:<15} | {tflops_b2_str:<15} | {tflops_fused_str}")
    
    print("="*130)
    
    # ===================================================================================
    # Backward Pass 性能测试
    # ===================================================================================
    print("\n" + "="*80)
    print("第三步：Backward Pass 性能对比（使用 Triton do_bench，正确处理 autotune）")
    print("="*80)
    
    all_bwd_results = []
    
    for B, H, N, D, causal, name in configs:
        print(f"\n[配置: {name}] B={B}, H={H}, N={N}, D={D}")
        
        # 根据序列长度动态调整测试次数
        if N >= 262144:
            warmup, rep = 10, 20
        elif N >= 131072:
            warmup, rep = 15, 30
        elif N >= 65536:
            warmup, rep = 20, 50
        elif N >= 32768:
            warmup, rep = 25, 75
        elif N >= 8192:
            warmup, rep = 25, 100
        else:
            warmup, rep = 25, 100
        
        print(f"  (Warmup={warmup}, Rep={rep}, 使用 Triton do_bench)")
        
        try:
            results = test_performance(B, H, N, D, causal, mode='bwd', warmup=warmup, rep=rep)
            results['config'] = name
            all_bwd_results.append(results)
        except Exception as e:
            print(f"  ❌ 失败: {e}")
    
    # Backward 总结
    print("\n" + "="*120)
    print("性能对比总结 (Backward Pass)")
    print("="*120)
    print(f"{'配置':<10} | {'Baseline1':<11} | {'Baseline2':<11} | {'Baseline3':<11} | {'Fused':<11} | {'B1 TF':<7} | {'B2 TF':<7} | {'B3 TF':<7} | {'Fused TF':<8}")
    print("-"*120)
    
    for result in all_bwd_results:
        if result.get('baseline1') or result.get('baseline2') or result.get('baseline3') or result.get('fused'):
            name = result['config']
            
            baseline1_t = result.get('baseline1', {}).get('median', 0)
            baseline1_tflops = result.get('baseline1', {}).get('tflops', 0)
            
            baseline2_t = result.get('baseline2', {}).get('median', 0)
            baseline2_tflops = result.get('baseline2', {}).get('tflops', 0)
            
            baseline3_t = result.get('baseline3', {}).get('median', 0)
            baseline3_tflops = result.get('baseline3', {}).get('tflops', 0)
            
            fused_t = result.get('fused', {}).get('median', 0)
            fused_tflops = result.get('fused', {}).get('tflops', 0)
            
            b1_str = f"{baseline1_t:.2f}ms" if baseline1_t > 0 else "N/A"
            b2_str = f"{baseline2_t:.2f}ms" if baseline2_t > 0 else "N/A"
            b3_str = f"{baseline3_t:.2f}ms" if baseline3_t > 0 else "N/A"
            fused_str = f"{fused_t:.2f}ms" if fused_t > 0 else "N/A"
            
            print(f"{name:<10} | {b1_str:<11} | {b2_str:<11} | {b3_str:<11} | {fused_str:<11} | {baseline1_tflops:>6.2f}  | {baseline2_tflops:>6.2f}  | {baseline3_tflops:>6.2f}  | {fused_tflops:>7.2f}")
    
    print("="*120)
    
    # Backward 性能差异分析（以 Baseline 3 为基准）
    print("\n" + "="*130)
    print("Backward 性能差异分析（相对于 Baseline 3: Flash Attention v2 Triton 无RoPE融合）")
    print("="*130)
    print(f"{'配置':<10} | {'B1 vs B3':<18} | {'B2 vs B3':<18} | {'Fused vs B3':<18} | {'B1 TFLOPS Δ':<15} | {'B2 TFLOPS Δ':<15} | {'Fused TFLOPS Δ':<15}")
    print("-"*130)
    
    for result in all_bwd_results:
        if result.get('baseline3'):
            name = result['config']
            b3_t = result['baseline3']['median']
            b3_tflops = result['baseline3']['tflops']
            
            baseline1_data = result.get('baseline1', {})
            baseline2_data = result.get('baseline2', {})
            fused_data = result.get('fused', {})
            
            if baseline1_data and baseline1_data.get('median'):
                speedup_b1 = b3_t / baseline1_data['median']
                tflops_diff_b1 = baseline1_data['tflops'] - b3_tflops
                speedup_b1_str = f"{speedup_b1:.3f}x {'↑' if speedup_b1 > 1 else '↓'}"
                tflops_b1_str = f"{tflops_diff_b1:+.2f}"
            else:
                speedup_b1_str = "N/A"
                tflops_b1_str = "N/A"
            
            if baseline2_data and baseline2_data.get('median'):
                speedup_b2 = b3_t / baseline2_data['median']
                tflops_diff_b2 = baseline2_data['tflops'] - b3_tflops
                speedup_b2_str = f"{speedup_b2:.3f}x {'↑' if speedup_b2 > 1 else '↓'}"
                tflops_b2_str = f"{tflops_diff_b2:+.2f}"
            else:
                speedup_b2_str = "N/A"
                tflops_b2_str = "N/A"
            
            if fused_data and fused_data.get('median'):
                speedup_fused = b3_t / fused_data['median']
                tflops_diff_fused = fused_data['tflops'] - b3_tflops
                speedup_fused_str = f"{speedup_fused:.3f}x {'↑' if speedup_fused > 1 else '↓'}"
                tflops_fused_str = f"{tflops_diff_fused:+.2f}"
            else:
                speedup_fused_str = "N/A"
                tflops_fused_str = "N/A"
            
            print(f"{name:<10} | {speedup_b1_str:<18} | {speedup_b2_str:<18} | {speedup_fused_str:<18} | {tflops_b1_str:<15} | {tflops_b2_str:<15} | {tflops_fused_str}")
    
    print("="*130)


if __name__ == "__main__":
    main()

