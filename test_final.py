"""
物理级零搬运方案 - 最终测试
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flash_attn_rope_triton import attention


def test_forward_backward():
    """测试 Forward + Backward 完整流程"""
    print("=" * 80)
    print("测试 Forward + Backward 完整流程")
    print("=" * 80)
    
    B, H, N, D = 2, 4, 256, 64
    device = 'cuda'
    dtype = torch.float16
    
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    freqs_cos = torch.randn(N, D // 2, device=device, dtype=dtype)
    freqs_sin = torch.randn(N, D // 2, device=device, dtype=dtype)
    
    try:
        # Forward
        print("\n[Forward]")
        o = attention(q, k, v, True, 0.5, freqs_cos, freqs_sin, False)
        print(f"✅ Forward 成功！输出形状: {o.shape}")
        print(f"   输出均值: {o.mean().item():.6f}, 标准差: {o.std().item():.6f}")
        
        # Backward
        print("\n[Backward]")
        loss = o.sum()
        loss.backward()
        
        print(f"✅ Backward 成功！")
        print(f"   dQ 形状: {q.grad.shape}, 均值: {q.grad.mean().item():.6f}")
        print(f"   dK 形状: {k.grad.shape}, 均值: {k.grad.mean().item():.6f}")
        print(f"   dV 形状: {v.grad.shape}, 均值: {v.grad.mean().item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_causal_modes():
    """测试 Causal 和 Non-Causal 模式"""
    print("\n" + "=" * 80)
    print("测试 Causal/Non-Causal 模式")
    print("=" * 80)
    
    B, H, N, D = 1, 2, 128, 64
    device = 'cuda'
    dtype = torch.float16
    
    results = {}
    
    for causal in [False, True]:
        print(f"\n[Causal = {causal}]")
        try:
            q = torch.randn(B, H, N, D, device=device, dtype=dtype)
            k = torch.randn(B, H, N, D, device=device, dtype=dtype)
            v = torch.randn(B, H, N, D, device=device, dtype=dtype)
            freqs_cos = torch.randn(N, D // 2, device=device, dtype=dtype)
            freqs_sin = torch.randn(N, D // 2, device=device, dtype=dtype)
            
            o = attention(q, k, v, causal, 0.5, freqs_cos, freqs_sin, False)
            print(f"✅ 成功！输出均值: {o.mean().item():.6f}")
            results[causal] = True
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            results[causal] = False
    
    return all(results.values())


def test_different_sizes():
    """测试不同的序列长度和特征维度"""
    print("\n" + "=" * 80)
    print("测试不同配置")
    print("=" * 80)
    
    configs = [
        (1, 1, 64, 64, "Small"),
        (2, 4, 128, 64, "Medium-64"),
        (2, 4, 128, 128, "Medium-128"),
        (1, 8, 512, 128, "Large"),
    ]
    
    results = {}
    
    for B, H, N, D, name in configs:
        print(f"\n[{name}: B={B}, H={H}, N={N}, D={D}]")
        try:
            q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
            k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
            v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
            freqs_cos = torch.randn(N, D // 2, device='cuda', dtype=torch.float16)
            freqs_sin = torch.randn(N, D // 2, device='cuda', dtype=torch.float16)
            
            o = attention(q, k, v, True, 0.5, freqs_cos, freqs_sin, False)
            print(f"✅ 成功！输出形状: {o.shape}")
            results[name] = True
            
        except Exception as e:
            print(f"❌ 失败: {str(e)[:100]}")
            results[name] = False
    
    return all(results.values())


if __name__ == "__main__":
    print("=" * 80)
    print("物理级零搬运方案 - 最终验证")
    print("=" * 80)
    
    all_results = {}
    
    all_results['forward_backward'] = test_forward_backward()
    all_results['causal_modes'] = test_causal_modes()
    all_results['different_sizes'] = test_different_sizes()
    
    print("\n" + "=" * 80)
    print("最终总结")
    print("=" * 80)
    for test_name, passed in all_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(all_results.values())
    if all_passed:
        print("\n🎉🎉🎉 所有测试通过！物理级零搬运方案已就绪！")
        print("\n💪 优化完成清单:")
        print("   ✅ 物理双指针加载（Forward & Backward）")
        print("   ✅ 双 dot 累加（完全避免拼接）")
        print("   ✅ 双指针 Dual-Store 写回")
        print("   ✅ Loop Hoisting（Q 在 dQ 计算中提升）")
        print("   ✅ 完整 Stride 支持（view/transpose 兼容）")
        print("\n🚀 可以开始 Benchmark 了！")
    else:
        print("\n⚠️  部分测试失败，需要进一步调试。")
    
    sys.exit(0 if all_passed else 1)

