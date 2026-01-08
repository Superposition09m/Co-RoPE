"""
Kernel 诊断工具：检查 register usage, occupancy 等
用于定位长序列性能退化的根本原因
"""

import torch
import triton

from fused_rope_attn import attention as fused_rope_attn
from flash_attn_v2_triton import attention as flash_attn_v2_triton
from rope_attn_pytorch import precompute_freqs_cis


def diagnose_kernel_config(seq_len, head_dim=128, batch=1, n_heads=32, causal=True):
    """
    诊断特定序列长度下的 kernel 配置
    
    Args:
        seq_len: 序列长度
        head_dim: head dimension
        batch: batch size
        n_heads: number of heads
        causal: causal masking
    """
    device = 'cuda'
    dtype = torch.float16
    
    print(f"\n{'='*80}")
    print(f"诊断序列长度: N={seq_len}, D={head_dim}, B={batch}, H={n_heads}")
    print(f"{'='*80}")
    
    # 准备输入
    q = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len, device=device)
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    print("\n[1] Baseline 3: Flash Attention v2 Triton (无RoPE融合)")
    try:
        # Warmup to trigger autotune
        for _ in range(10):
            _ = flash_attn_v2_triton(q, k, v, causal, sm_scale, False)
        torch.cuda.synchronize()
        
        print("  ✓ Kernel compiled successfully")
        # Note: Getting actual register count and occupancy requires nsight compute
        print("  ⚠️  Use `nsys profile` or `ncu` to get detailed metrics:")
        print(f"      ncu --metrics sm__warps_active.avg.pct_of_peak python -c 'your_code'")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n[2] Fused RoPE: Triton (RoPE融合)")
    try:
        # Warmup to trigger autotune
        for _ in range(10):
            _ = fused_rope_attn(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)
        torch.cuda.synchronize()
        
        print("  ✓ Kernel compiled successfully")
        print("  ⚠️  Use `nsys profile` or `ncu` to get detailed metrics")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # 理论分析
    print(f"\n[3] 理论分析")
    print(f"  序列长度: {seq_len}")
    print(f"  FLOPs (causal): {2 * batch * n_heads * seq_len * seq_len * head_dim * 0.5 / 1e9:.2f} GFLOPs")
    print(f"  Memory (Q+K+V): {3 * batch * n_heads * seq_len * head_dim * 2 / 1024**2:.2f} MB")
    print(f"  Memory (O): {batch * n_heads * seq_len * head_dim * 2 / 1024**2:.2f} MB")
    print(f"  Total Memory: {4 * batch * n_heads * seq_len * head_dim * 2 / 1024**2:.2f} MB")
    
    # L2 cache size (H200 has 60MB L2)
    l2_cache_mb = 60
    total_mem_mb = 4 * batch * n_heads * seq_len * head_dim * 2 / 1024**2
    if total_mem_mb > l2_cache_mb:
        print(f"\n  ⚠️  数据量 ({total_mem_mb:.2f} MB) 超过 L2 cache ({l2_cache_mb} MB)")
        print(f"      长序列时会频繁访问 HBM，bandwidth 成为瓶颈")
    
    # Register pressure估计
    print(f"\n[4] Register Pressure 估计")
    print(f"  Fused RoPE 需要额外存储:")
    print(f"    - cos/sin frequencies: ~{seq_len * head_dim // 2 * 4 / 1024:.2f} KB per block")
    print(f"    - Rotated Q/K: 增加中间结果寄存器使用")
    print(f"    - 估计增加 20-30% 寄存器使用")
    
    if seq_len >= 8192:
        print(f"\n  💡 建议:")
        print(f"    1. 减小 BLOCK_M/BLOCK_N 以降低寄存器压力")
        print(f"    2. 使用 multi-stage pipeline 优化内存访问")
        print(f"    3. 考虑长序列时回退到非融合版本")


def run_diagnosis():
    """运行完整诊断"""
    print("="*80)
    print("Fused RoPE Kernel 性能诊断工具")
    print("="*80)
    
    # 诊断不同序列长度
    test_configs = [
        (512, 128, 4, 32),    # 短序列：性能好
        (2048, 128, 2, 32),   # 中等序列：性能好
        (8192, 128, 1, 32),   # 长序列：开始退化
        (32768, 128, 1, 16),  # 超长序列：严重退化
    ]
    
    for seq_len, head_dim, batch, n_heads in test_configs:
        diagnose_kernel_config(seq_len, head_dim, batch, n_heads)
    
    print(f"\n{'='*80}")
    print("诊断完成")
    print(f"{'='*80}")
    
    print("\n💡 下一步:")
    print("1. 运行 nsight compute 获取详细 metrics:")
    print("   ncu --set full -o profile python bench_compare.py")
    print("\n2. 查看关键指标:")
    print("   - sm__warps_active.avg.pct_of_peak (Occupancy)")
    print("   - l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum (Global Load)")
    print("   - smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (Load Efficiency)")


if __name__ == "__main__":
    run_diagnosis()

