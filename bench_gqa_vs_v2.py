"""
GQA vs 原始 V2 性能对比
对比在相同 query heads 数量下，GQA（更少的 KV heads）相比标准 MHA 的性能提升
"""

import torch
import triton
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flash_attn_v2_triton import attention as attention_v2
from flash_attn_co_rope_gqa_triton import attention as attention_gqa


def benchmark_config(B, H_Q, H_KV, N, D, dtype=torch.float16, num_repeats=100):
    """
    对比单个配置下的性能
    
    Args:
        B: batch size
        H_Q: query heads 数量
        H_KV: key/value heads 数量（GQA）
        N: sequence length
        D: head dimension
        dtype: 数据类型
        num_repeats: 重复次数
    """
    device = 'cuda'
    causal = True
    sm_scale = 1.0 / (D ** 0.5)
    warp_specialize = False
    
    # 准备输入
    torch.manual_seed(42)
    
    # V2 版本：所有 heads 都是独立的 (H_Q == H_KV)
    q_v2 = torch.randn(B, H_Q, N, D, device=device, dtype=dtype)
    k_v2 = torch.randn(B, H_Q, N, D, device=device, dtype=dtype)
    v_v2 = torch.randn(B, H_Q, N, D, device=device, dtype=dtype)
    
    # GQA 版本：更少的 KV heads
    q_gqa = torch.randn(B, H_Q, N, D, device=device, dtype=dtype)
    k_gqa = torch.randn(B, H_KV, N, D, device=device, dtype=dtype)
    v_gqa = torch.randn(B, H_KV, N, D, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(5):
        _ = attention_v2(q_v2, k_v2, v_v2, causal, sm_scale, warp_specialize)
        _ = attention_gqa(q_gqa, k_gqa, v_gqa, causal, sm_scale, warp_specialize)
        torch.cuda.synchronize()
    
    # 测速 V2
    fn_v2 = lambda: attention_v2(q_v2, k_v2, v_v2, causal, sm_scale, warp_specialize)
    time_v2 = triton.testing.do_bench(fn_v2, rep=num_repeats)
    
    # 测速 GQA
    fn_gqa = lambda: attention_gqa(q_gqa, k_gqa, v_gqa, causal, sm_scale, warp_specialize)
    time_gqa = triton.testing.do_bench(fn_gqa, rep=num_repeats)
    
    # 计算内存占用（理论值）
    # Q: [B, H_Q, N, D]
    # V2: K,V 都是 [B, H_Q, N, D]
    # GQA: K,V 都是 [B, H_KV, N, D]
    bytes_per_element = 2 if dtype == torch.float16 else 4
    kv_memory_v2 = 2 * B * H_Q * N * D * bytes_per_element / (1024**2)  # MB
    kv_memory_gqa = 2 * B * H_KV * N * D * bytes_per_element / (1024**2)  # MB
    
    return {
        'time_v2': time_v2,
        'time_gqa': time_gqa,
        'speedup': time_v2 / time_gqa if time_gqa > 0 else 0,
        'kv_memory_v2': kv_memory_v2,
        'kv_memory_gqa': kv_memory_gqa,
        'memory_saved': kv_memory_v2 - kv_memory_gqa,
        'memory_ratio': kv_memory_v2 / kv_memory_gqa if kv_memory_gqa > 0 else 0,
    }


def main():
    print("=" * 80)
    print("GQA vs 原始 V2 性能对比")
    print("=" * 80)
    print("\n说明：")
    print("  - V2:  标准 Multi-Head Attention (H_Q == H_KV)")
    print("  - GQA: Grouped-Query Attention (H_Q > H_KV, KV heads 被多个 Q heads 共享)")
    print("  - 测试配置: dtype=FP16, causal=True")
    print("=" * 80)
    
    # 测试配置列表
    configs = [
        # (B, H_Q, H_KV, N, D, 描述)
        (1, 8, 2, 128, 64, "小序列 (N=128)"),
        (1, 8, 2, 512, 64, "中序列 (N=512)"),
        (1, 8, 2, 1024, 64, "长序列 (N=1024)"),
        (1, 8, 2, 2048, 64, "长序列 (N=2048)"),
        (2, 8, 2, 1024, 64, "Batch=2, N=1024"),
        (1, 32, 8, 1024, 64, "大模型配置 (H=32)"),
        (1, 8, 1, 1024, 64, "MQA (H_KV=1)"),
    ]
    
    print()
    results = []
    
    for B, H_Q, H_KV, N, D, desc in configs:
        GROUP_SIZE = H_Q // H_KV
        print(f"\n{'='*80}")
        print(f"配置: {desc}")
        print(f"  B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, GROUP_SIZE={GROUP_SIZE}")
        print(f"{'='*80}")
        
        try:
            result = benchmark_config(B, H_Q, H_KV, N, D)
            results.append((desc, result))
            
            print(f"\n性能:")
            print(f"  V2 (H={H_Q}):        {result['time_v2']:.3f} ms")
            print(f"  GQA (H_Q={H_Q}, H_KV={H_KV}): {result['time_gqa']:.3f} ms")
            print(f"  ⚡ Speedup:          {result['speedup']:.2f}x")
            
            print(f"\nKV Cache 内存:")
            print(f"  V2:                 {result['kv_memory_v2']:.2f} MB")
            print(f"  GQA:                {result['kv_memory_gqa']:.2f} MB")
            print(f"  💾 Memory Saved:    {result['memory_saved']:.2f} MB ({(1 - 1/result['memory_ratio'])*100:.1f}% 减少)")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总表格
    print("\n" + "=" * 80)
    print("汇总表格")
    print("=" * 80)
    print(f"{'配置':<25} {'V2 (ms)':<12} {'GQA (ms)':<12} {'Speedup':<10} {'内存节省':<12}")
    print("-" * 80)
    for desc, result in results:
        speedup_color = "🚀" if result['speedup'] > 1.2 else "  "
        memory_pct = (1 - 1/result['memory_ratio'])*100
        print(f"{desc:<25} {result['time_v2']:>8.3f}     {result['time_gqa']:>8.3f}     "
              f"{speedup_color}{result['speedup']:>5.2f}x    {memory_pct:>5.1f}%")
    
    print("\n" + "=" * 80)
    print("总结：")
    print("  ✅ GQA 在保持相同 query heads 的情况下，通过减少 KV heads 实现：")
    print("     - 计算速度提升")
    print("     - KV cache 内存显著减少（重要！在推理时可以支持更大的 batch size）")
    print("=" * 80)


if __name__ == "__main__":
    main()

