"""
Co-RoPE GQA 性能对比：PyTorch vs Triton
========================================

测试两个版本的 Co-RoPE GQA 实现：
1. PyTorch 参考实现 (corope_attn_gqa_pytorch.py)
2. Triton 优化实现 (flash_attn_co_rope_gqa_triton.py)

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

from flash_attn_co_rope_gqa_triton import attention as attention_triton
from corope_attn_gqa_pytorch import attention_pytorch


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


def compute_flops(B, H_Q, H_KV, N, D, time_ms, mode='fwd'):
    """
    计算 FLOPS（考虑 GQA）
    
    Args:
        B: Batch size
        H_Q: Number of query heads
        H_KV: Number of KV heads
        N: Sequence length
        D: Head dimension
        time_ms: Time in milliseconds
        mode: 'fwd' or 'bwd'
    """
    # Forward: Q@K^T 和 Attn@V
    # Q@K^T: B * H_Q * N * N * D
    # Attn@V: B * H_Q * N * N * D (但 K/V 共享，所以是 H_KV)
    flops_qk = 2.0 * B * H_Q * N * N * D
    flops_attn_v = 2.0 * B * H_Q * N * N * D  # 虽然 V 只有 H_KV 个 head，但广播到 H_Q
    
    # Co-RoPE 额外计算：里程计算、相位校准等
    flops_rope = B * H_KV * N * N * D * 0.5  # 里程计算（leader heads）
    flops_rope += B * H_Q * N * N * D * 0.3  # 相位校准
    
    total_flops = flops_qk + flops_attn_v + flops_rope
    
    if mode == 'bwd':
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    
    tflops = total_flops * 1e-12 / (time_ms * 1e-3)
    return tflops


def test_correctness(B, H_Q, H_KV, N, D, causal=True, test_backward=False, atol=1e-2, rtol=1e-2):
    """
    测试正确性（Forward + Backward）
    
    Args:
        B: Batch size
        H_Q: Number of query heads
        H_KV: Number of KV heads
        N: Sequence length
        D: Head dimension
        causal: Whether to use causal mask
        test_backward: Whether to test backward pass
        atol: Absolute tolerance
        rtol: Relative tolerance
    """
    mode_str = "Forward + Backward" if test_backward else "Forward Only"
    print(f"\n{'='*80}")
    print(f"正确性测试 ({mode_str}): B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, causal={causal}")
    print(f"{'='*80}")
    
    device = 'cuda'
    dtype = torch.float16
    
    torch.manual_seed(42)
    q = torch.randn(B, H_Q, N, D, device=device, dtype=dtype, requires_grad=test_backward)
    k = torch.randn(B, H_KV, N, D, device=device, dtype=dtype, requires_grad=test_backward)
    v = torch.randn(B, H_KV, N, D, device=device, dtype=dtype, requires_grad=test_backward)
    sm_scale = 1.0 / (D ** 0.5)
    theta = 10000.0
    
    # PyTorch 参考实现
    print("\n[1. PyTorch 参考实现]")
    try:
        o_pytorch = attention_pytorch(q, k, v, causal, sm_scale, theta)
        if torch.isnan(o_pytorch).any() or torch.isinf(o_pytorch).any():
            print(f"  ❌ Forward 输出包含 NaN/Inf")
            return False
        print(f"  ✅ Forward Output: shape={o_pytorch.shape}, mean={o_pytorch.mean().item():.6f}, std={o_pytorch.std().item():.6f}")
        
        if test_backward:
            dout = torch.randn_like(o_pytorch)
            o_pytorch.backward(dout, retain_graph=True)
            dq_pytorch = q.grad.clone()
            dk_pytorch = k.grad.clone()
            dv_pytorch = v.grad.clone()
            q.grad, k.grad, v.grad = None, None, None
            
            if torch.isnan(dq_pytorch).any() or torch.isinf(dq_pytorch).any():
                print(f"  ❌ Backward 梯度包含 NaN/Inf")
                return False
            print(f"  ✅ Backward dq: mean={dq_pytorch.mean().item():.6f}, std={dq_pytorch.std().item():.6f}")
            print(f"  ✅ Backward dk: mean={dk_pytorch.mean().item():.6f}, std={dk_pytorch.std().item():.6f}")
            print(f"  ✅ Backward dv: mean={dv_pytorch.mean().item():.6f}, std={dv_pytorch.std().item():.6f}")
    except Exception as e:
        print(f"  ❌ PyTorch 失败: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False
    
    # Triton 实现
    print("\n[2. Triton 优化实现]")
    try:
        o_triton = attention_triton(q, k, v, causal, sm_scale, theta)
        if torch.isnan(o_triton).any() or torch.isinf(o_triton).any():
            print(f"  ❌ Forward 输出包含 NaN/Inf")
            return False
        print(f"  ✅ Forward Output: shape={o_triton.shape}, mean={o_triton.mean().item():.6f}, std={o_triton.std().item():.6f}")
        
        if test_backward:
            o_triton.backward(dout, retain_graph=True)
            dq_triton = q.grad.clone()
            dk_triton = k.grad.clone()
            dv_triton = v.grad.clone()
            q.grad, k.grad, v.grad = None, None, None
            
            if torch.isnan(dq_triton).any() or torch.isinf(dq_triton).any():
                print(f"  ❌ Backward 梯度包含 NaN/Inf")
                return False
            print(f"  ✅ Backward dq: mean={dq_triton.mean().item():.6f}, std={dq_triton.std().item():.6f}")
            print(f"  ✅ Backward dk: mean={dk_triton.mean().item():.6f}, std={dk_triton.std().item():.6f}")
            print(f"  ✅ Backward dv: mean={dv_triton.mean().item():.6f}, std={dv_triton.std().item():.6f}")
    except Exception as e:
        print(f"  ❌ Triton 失败: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False
    
    # 对拍：Forward
    print("\n[对拍结果]")
    fwd_diff = (o_pytorch - o_triton).abs().float()  # 转换为 float32 以支持 quantile
    fwd_max_diff = fwd_diff.max().item()
    fwd_mean_diff = fwd_diff.mean().item()
    fwd_median_diff = fwd_diff.median().item()
    fwd_p95_diff = fwd_diff.quantile(0.95).item()
    fwd_p99_diff = fwd_diff.quantile(0.99).item()
    
    # 相对误差
    fwd_rel_diff = fwd_diff / (o_pytorch.abs().float() + 1e-8)
    fwd_max_rel_diff = fwd_rel_diff.max().item()
    fwd_mean_rel_diff = fwd_rel_diff.mean().item()
    
    print(f"  Forward 绝对误差:")
    print(f"    Max Diff:    {fwd_max_diff:.6f}")
    print(f"    Mean Diff:   {fwd_mean_diff:.6f}")
    print(f"    Median Diff: {fwd_median_diff:.6f}")
    print(f"    P95 Diff:    {fwd_p95_diff:.6f}")
    print(f"    P99 Diff:    {fwd_p99_diff:.6f}")
    print(f"  Forward 相对误差:")
    print(f"    Max Rel Diff:  {fwd_max_rel_diff:.6f}")
    print(f"    Mean Rel Diff: {fwd_mean_rel_diff:.6f}")
    
    # 检查是否通过
    fwd_pass = fwd_max_diff < atol or (fwd_max_rel_diff < rtol and fwd_mean_rel_diff < rtol * 0.1)
    if fwd_pass:
        print(f"  ✅ Forward 对拍通过 (atol={atol}, rtol={rtol})")
    else:
        print(f"  ⚠️  Forward 对拍未完全通过 (atol={atol}, rtol={rtol})")
        # 找出最大差异位置
        max_idx = fwd_diff.argmax()
        coords = torch.unravel_index(max_idx, fwd_diff.shape)
        print(f"    最大差异位置: b={coords[0]}, h={coords[1]}, i={coords[2]}, j={coords[3]}")
        print(f"    PyTorch: {o_pytorch[coords]:.6f}")
        print(f"    Triton:  {o_triton[coords]:.6f}")
        print(f"    Diff:    {fwd_diff[coords]:.6f}")
    
    # 对拍：Backward
    if test_backward:
        dq_diff = (dq_pytorch - dq_triton).abs()
        dk_diff = (dk_pytorch - dk_triton).abs()
        dv_diff = (dv_pytorch - dv_triton).abs()
        
        dq_max_diff = dq_diff.max().item()
        dk_max_diff = dk_diff.max().item()
        dv_max_diff = dv_diff.max().item()
        
        dq_mean_diff = dq_diff.mean().item()
        dk_mean_diff = dk_diff.mean().item()
        dv_mean_diff = dv_diff.mean().item()
        
        print(f"\n  Backward 绝对误差:")
        print(f"    dQ Max Diff:  {dq_max_diff:.6f}, Mean Diff: {dq_mean_diff:.6f}")
        print(f"    dK Max Diff:  {dk_max_diff:.6f}, Mean Diff: {dk_mean_diff:.6f}")
        print(f"    dV Max Diff:  {dv_max_diff:.6f}, Mean Diff: {dv_mean_diff:.6f}")
        
        bwd_pass = (dq_max_diff < atol * 10 or dq_mean_diff < atol) and \
                   (dk_max_diff < atol * 10 or dk_mean_diff < atol) and \
                   (dv_max_diff < atol * 10 or dv_mean_diff < atol)
        
        if bwd_pass:
            print(f"  ✅ Backward 对拍通过 (atol={atol})")
        else:
            print(f"  ⚠️  Backward 对拍未完全通过 (atol={atol})")
            # 找出最大差异位置
            if dq_max_diff > atol * 10:
                max_idx = dq_diff.argmax()
                coords = torch.unravel_index(max_idx, dq_diff.shape)
                print(f"    dQ 最大差异位置: {coords}")
                print(f"      PyTorch: {dq_pytorch[coords]:.6f}")
                print(f"      Triton:  {dq_triton[coords]:.6f}")
    else:
        bwd_pass = True
    
    return fwd_pass and bwd_pass


def test_performance(B, H_Q, H_KV, N, D, causal=True, mode='fwd', warmup=25, rep=100):
    """
    性能测试两个版本（使用 Triton 官方 benchmark，正确处理 autotune）
    
    Args:
        B: Batch size
        H_Q: Number of query heads
        H_KV: Number of KV heads
        N: Sequence length
        D: Head dimension
        causal: Whether to use causal mask
        mode: 'fwd' 或 'bwd'
        warmup: Warmup iterations（默认 25，确保 autotune 完成）
        rep: Measurement iterations（默认 100）
    """
    device = 'cuda'
    dtype = torch.float16
    
    # 根据 mode 决定是否需要梯度
    requires_grad = (mode == 'bwd')
    
    q = torch.randn(B, H_Q, N, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, H_KV, N, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, H_KV, N, D, device=device, dtype=dtype, requires_grad=requires_grad)
    sm_scale = 1.0 / (D ** 0.5)
    theta = 10000.0
    
    results = {}
    
    # 1. PyTorch 参考实现
    print("\n  [1. PyTorch]", end=' ', flush=True)
    
    if mode == 'fwd':
        fn_pytorch = lambda: attention_pytorch(q, k, v, causal, sm_scale, theta)
    else:  # bwd
        # 每次调用都需要重新创建 tensor，避免梯度累积
        def fn_pytorch():
            q_pytorch = q.clone().detach().requires_grad_(True)
            k_pytorch = k.clone().detach().requires_grad_(True)
            v_pytorch = v.clone().detach().requires_grad_(True)
            o_pytorch = attention_pytorch(q_pytorch, k_pytorch, v_pytorch, causal, sm_scale, theta)
            dout_pytorch = torch.randn_like(o_pytorch)
            o_pytorch.backward(dout_pytorch)
            return o_pytorch
    
    try:
        median, min_t, max_t = benchmark_kernel(fn_pytorch, warmup=warmup, rep=rep)
        tflops = compute_flops(B, H_Q, H_KV, N, D, median, mode)
        results['pytorch'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['pytorch'] = None
    
    # 2. Triton 优化实现
    print("  [2. Triton]", end=' ', flush=True)
    
    if mode == 'fwd':
        fn_triton = lambda: attention_triton(q, k, v, causal, sm_scale, theta)
    else:  # bwd
        # 每次调用都需要重新创建 tensor，避免梯度累积
        def fn_triton():
            q_triton = q.clone().detach().requires_grad_(True)
            k_triton = k.clone().detach().requires_grad_(True)
            v_triton = v.clone().detach().requires_grad_(True)
            o_triton = attention_triton(q_triton, k_triton, v_triton, causal, sm_scale, theta)
            dout_triton = torch.randn_like(o_triton)
            o_triton.backward(dout_triton)
            return o_triton
    
    try:
        median, min_t, max_t = benchmark_kernel(fn_triton, warmup=warmup, rep=rep)
        tflops = compute_flops(B, H_Q, H_KV, N, D, median, mode)
        results['triton'] = {'median': median, 'min': min_t, 'max': max_t, 'tflops': tflops}
        print(f"{median:.3f} ms ({tflops:.2f} TFLOPS)")
    except Exception as e:
        print(f"❌ {str(e)[:50]}")
        results['triton'] = None
    
    return results


def main():
    print("="*80)
    print("Co-RoPE GQA 性能对比: PyTorch vs Triton")
    print("="*80)
    
    # 测试配置：(BATCH, H_Q, H_KV, N_CTX, HEAD_DIM, causal, name)
    configs = [
        # 小规模测试
        (1, 4, 2, 32, 64, True, "Small-32"),
        (1, 4, 2, 64, 64, True, "Small-64"),
        (2, 4, 2, 64, 128, True, "Small-64-D128"),
        # 中等规模
        (2, 8, 4, 128, 64, True, "Med-128"),
        (2, 8, 4, 256, 64, True, "Med-256"),
        (2, 8, 4, 512, 64, True, "Med-512"),
        (1, 8, 4, 1024, 64, True, "Med-1K"),
        # 大规模
        (1, 8, 4, 2048, 64, True, "Large-2K"),
        (1, 8, 4, 4096, 64, True, "Large-4K"),
        (1, 8, 4, 8192, 64, True, "Large-8K"),
        (1, 4, 2, 16384, 64, True, "Large-16K"),
        # D=128 配置
        (2, 8, 4, 128, 128, True, "Med-128-D128"),
        (2, 8, 4, 256, 128, True, "Med-256-D128"),
        (1, 8, 4, 512, 128, True, "Med-512-D128"),
        (1, 8, 4, 1024, 128, True, "Med-1K-D128"),
        (1, 8, 4, 2048, 128, True, "Large-2K-D128"),
    ]
    
    # 正确性测试
    print("\n" + "="*80)
    print("第一步：正确性验证")
    print("="*80)
    
    # Forward 正确性测试
    print("\n" + "-"*80)
    print("1.1 Forward 正确性测试")
    print("-"*80)
    fwd_correctness_passed = test_correctness(1, 4, 2, 32, 64, causal=True, test_backward=False)
    
    if not fwd_correctness_passed:
        print("\n⚠️  Forward 正确性测试未通过，停止测试")
        return
    
    # Backward 正确性测试
    print("\n" + "-"*80)
    print("1.2 Backward 正确性测试")
    print("-"*80)
    bwd_correctness_passed = test_correctness(1, 4, 2, 32, 64, causal=True, test_backward=True)
    
    if not bwd_correctness_passed:
        print("\n⚠️  Backward 正确性测试未通过，停止测试")
        return
    
    # Forward 性能测试
    print("\n" + "="*80)
    print("第二步：Forward Pass 性能对比（使用 Triton do_bench，正确处理 autotune）")
    print("="*80)
    
    all_fwd_results = []
    
    for B, H_Q, H_KV, N, D, causal, name in configs:
        print(f"\n[配置: {name}] B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
        
        # 根据序列长度动态调整测试次数（使用 Triton benchmark）
        if N >= 16384:
            warmup, rep = 10, 20   # 16K+ 测试少一点
        elif N >= 8192:
            warmup, rep = 15, 30   # 8K
        elif N >= 4096:
            warmup, rep = 20, 50   # 4K
        elif N >= 2048:
            warmup, rep = 25, 75   # 2K
        elif N >= 1024:
            warmup, rep = 25, 100  # 1K
        else:
            warmup, rep = 25, 100  # ≤512 标准测试
        
        print(f"  (Warmup={warmup}, Rep={rep}, 使用 Triton do_bench)")
        
        try:
            results = test_performance(B, H_Q, H_KV, N, D, causal, mode='fwd', warmup=warmup, rep=rep)
            results['config'] = name
            all_fwd_results.append(results)
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # Forward 总结
    print("\n" + "="*100)
    print("性能对比总结 (Forward Pass)")
    print("="*100)
    print(f"{'配置':<15} | {'PyTorch Time':<15} | {'Triton Time':<15} | {'PyTorch TFLOPS':<15} | {'Triton TFLOPS':<15} | {'Speedup':<10}")
    print("-"*100)
    
    for result in all_fwd_results:
        if result.get('pytorch') and result.get('triton'):
            name = result['config']
            pytorch_t = result['pytorch']['median']
            pytorch_tflops = result['pytorch']['tflops']
            triton_t = result['triton']['median']
            triton_tflops = result['triton']['tflops']
            
            speedup = pytorch_t / triton_t if triton_t > 0 else 0
            
            pytorch_str = f"{pytorch_t:.2f}ms" if pytorch_t > 0 else "N/A"
            triton_str = f"{triton_t:.2f}ms" if triton_t > 0 else "N/A"
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            
            print(f"{name:<15} | {pytorch_str:<15} | {triton_str:<15} | {pytorch_tflops:>14.2f}  | {triton_tflops:>14.2f}  | {speedup_str:<10}")
    
    print("="*100)
    
    # Backward Pass 性能测试
    print("\n" + "="*80)
    print("第三步：Backward Pass 性能对比（使用 Triton do_bench，正确处理 autotune）")
    print("="*80)
    
    all_bwd_results = []
    
    # Backward 测试使用较小的配置（因为更慢）
    bwd_configs = [
        (1, 4, 2, 32, 64, True, "Small-32"),
        (1, 4, 2, 64, 64, True, "Small-64"),
        (2, 4, 2, 64, 128, True, "Small-64-D128"),
        (2, 8, 4, 128, 64, True, "Med-128"),
        (2, 8, 4, 256, 64, True, "Med-256"),
        (1, 8, 4, 512, 64, True, "Med-512"),
        (1, 8, 4, 1024, 64, True, "Med-1K"),
        (1, 8, 4, 2048, 64, True, "Large-2K"),
    ]
    
    for B, H_Q, H_KV, N, D, causal, name in bwd_configs:
        print(f"\n[配置: {name}] B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}")
        
        # 根据序列长度动态调整测试次数
        if N >= 2048:
            warmup, rep = 15, 30
        elif N >= 1024:
            warmup, rep = 20, 50
        elif N >= 512:
            warmup, rep = 25, 75
        else:
            warmup, rep = 25, 100
        
        print(f"  (Warmup={warmup}, Rep={rep}, 使用 Triton do_bench)")
        
        try:
            results = test_performance(B, H_Q, H_KV, N, D, causal, mode='bwd', warmup=warmup, rep=rep)
            results['config'] = name
            all_bwd_results.append(results)
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # Backward 总结
    print("\n" + "="*100)
    print("性能对比总结 (Backward Pass)")
    print("="*100)
    print(f"{'配置':<15} | {'PyTorch Time':<15} | {'Triton Time':<15} | {'PyTorch TFLOPS':<15} | {'Triton TFLOPS':<15} | {'Speedup':<10}")
    print("-"*100)
    
    for result in all_bwd_results:
        if result.get('pytorch') and result.get('triton'):
            name = result['config']
            pytorch_t = result['pytorch']['median']
            pytorch_tflops = result['pytorch']['tflops']
            triton_t = result['triton']['median']
            triton_tflops = result['triton']['tflops']
            
            speedup = pytorch_t / triton_t if triton_t > 0 else 0
            
            pytorch_str = f"{pytorch_t:.2f}ms" if pytorch_t > 0 else "N/A"
            triton_str = f"{triton_t:.2f}ms" if triton_t > 0 else "N/A"
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            
            print(f"{name:<15} | {pytorch_str:<15} | {triton_str:<15} | {pytorch_tflops:>14.2f}  | {triton_tflops:>14.2f}  | {speedup_str:<10}")
    
    print("="*100)


if __name__ == "__main__":
    main()

