"""
Co-RoPE PyTorch vs Triton 对拍测试
比较 corope_attn_gqa_pytorch.py 和 flash_attn_co_rope_gqa_triton.py 两个实现

两个实现接口完全一致：
    attention(q, k, v, causal, sm_scale, theta)
    
其中 theta 是 RoPE 基础频率（通常为 10000.0），两个实现内部都会计算 inv_freq。
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from corope_attn_gqa_pytorch import attention_pytorch
from flash_attn_co_rope_gqa_triton import attention as attention_triton


def compare_forward(B, H_Q, H_KV, N, D, dtype=torch.float16, rtol=1e-3, atol=1e-5):
    """
    对拍测试前向传播
    
    Args:
        B: batch size
        H_Q: number of query heads
        H_KV: number of KV heads
        N: sequence length
        D: head dimension
        dtype: data type
        rtol: relative tolerance
        atol: absolute tolerance
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  CUDA not available, skipping test")
        return
    
    torch.manual_seed(42)
    
    # 准备输入
    q = torch.randn(B, H_Q, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H_KV, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H_KV, N, D, device=device, dtype=dtype)
    
    sm_scale = D ** -0.5
    theta = 10000.0
    causal = True
    
    # PyTorch 版本
    try:
        out_pytorch = attention_pytorch(q, k, v, causal, sm_scale, theta)
    except Exception as e:
        print(f"❌ PyTorch forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Triton 版本（现在接口一致，都用 theta）
    try:
        out_triton = attention_triton(q, k, v, causal, sm_scale, theta)
    except Exception as e:
        print(f"❌ Triton forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 比较结果
    max_diff = (out_pytorch - out_triton).abs().max().item()
    mean_diff = (out_pytorch - out_triton).abs().mean().item()
    
    print(f"  Forward Max Diff: {max_diff:.2e}, Mean Diff: {mean_diff:.2e}")
    
    if torch.allclose(out_pytorch, out_triton, rtol=rtol, atol=atol):
        print(f"  ✅ Forward PASS (rtol={rtol}, atol={atol})")
        return True
    else:
        print(f"  ❌ Forward FAIL (rtol={rtol}, atol={atol})")
        print(f"     PyTorch output: mean={out_pytorch.mean().item():.6f}, std={out_pytorch.std().item():.6f}")
        print(f"     Triton output: mean={out_triton.mean().item():.6f}, std={out_triton.std().item():.6f}")
        return False


def compare_backward(B, H_Q, H_KV, N, D, dtype=torch.float16, rtol=1e-2, atol=1e-4):
    """
    对拍测试反向传播
    
    Args:
        B: batch size
        H_Q: number of query heads
        H_KV: number of KV heads
        N: sequence length
        D: head dimension
        dtype: data type
        rtol: relative tolerance
        atol: absolute tolerance
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  CUDA not available, skipping test")
        return
    
    torch.manual_seed(42)
    
    # 准备输入（需要梯度）
    q_pytorch = torch.randn(B, H_Q, N, D, device=device, dtype=dtype, requires_grad=True)
    k_pytorch = torch.randn(B, H_KV, N, D, device=device, dtype=dtype, requires_grad=True)
    v_pytorch = torch.randn(B, H_KV, N, D, device=device, dtype=dtype, requires_grad=True)
    
    q_triton = q_pytorch.clone().detach().requires_grad_(True)
    k_triton = k_pytorch.clone().detach().requires_grad_(True)
    v_triton = v_pytorch.clone().detach().requires_grad_(True)
    
    sm_scale = D ** -0.5
    theta = 10000.0
    causal = True
    
    # PyTorch 版本
    try:
        out_pytorch = attention_pytorch(q_pytorch, k_pytorch, v_pytorch, causal, sm_scale, theta)
        loss_pytorch = out_pytorch.sum()
        loss_pytorch.backward()
        dq_pytorch = q_pytorch.grad.clone()
        dk_pytorch = k_pytorch.grad.clone()
        dv_pytorch = v_pytorch.grad.clone()
    except Exception as e:
        print(f"❌ PyTorch backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Triton 版本（现在接口一致，都用 theta）
    try:
        out_triton = attention_triton(q_triton, k_triton, v_triton, causal, sm_scale, theta)
        loss_triton = out_triton.sum()
        loss_triton.backward()
        dq_triton = q_triton.grad.clone()
        dk_triton = k_triton.grad.clone()
        dv_triton = v_triton.grad.clone()
    except Exception as e:
        print(f"❌ Triton backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 比较梯度
    dq_diff = (dq_pytorch - dq_triton).abs()
    dk_diff = (dk_pytorch - dk_triton).abs()
    dv_diff = (dv_pytorch - dv_triton).abs()
    
    dq_max = dq_diff.max().item()
    dq_mean = dq_diff.mean().item()
    dk_max = dk_diff.max().item()
    dk_mean = dk_diff.mean().item()
    dv_max = dv_diff.max().item()
    dv_mean = dv_diff.mean().item()
    
    print(f"  Backward dQ: Max Diff={dq_max:.2e}, Mean Diff={dq_mean:.2e}")
    print(f"  Backward dK: Max Diff={dk_max:.2e}, Mean Diff={dk_mean:.2e}")
    print(f"  Backward dV: Max Diff={dv_max:.2e}, Mean Diff={dv_mean:.2e}")
    
    all_close = (
        torch.allclose(dq_pytorch, dq_triton, rtol=rtol, atol=atol) and
        torch.allclose(dk_pytorch, dk_triton, rtol=rtol, atol=atol) and
        torch.allclose(dv_pytorch, dv_triton, rtol=rtol, atol=atol)
    )
    
    if all_close:
        print(f"  ✅ Backward PASS (rtol={rtol}, atol={atol})")
        return True
    else:
        print(f"  ❌ Backward FAIL (rtol={rtol}, atol={atol})")
        return False


def test_non_causal_rejection():
    """测试两个实现是否都拒绝 noncausal"""
    print("\n" + "="*80)
    print("测试 noncausal 拒绝")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  CUDA not available, skipping test")
        return
    
    B, H_Q, H_KV, N, D = 1, 4, 2, 32, 64
    dtype = torch.float16
    
    q = torch.randn(B, H_Q, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H_KV, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H_KV, N, D, device=device, dtype=dtype)
    
    sm_scale = D ** -0.5
    theta = 10000.0
    causal = False
    
    # 测试 PyTorch 版本
    pytorch_rejected = False
    try:
        attention_pytorch(q, k, v, causal, sm_scale, theta)
    except ValueError as e:
        if "causal" in str(e).lower():
            pytorch_rejected = True
            print(f"  ✅ PyTorch correctly rejects noncausal: {e}")
        else:
            print(f"  ❌ PyTorch raised unexpected error: {e}")
    except Exception as e:
        print(f"  ❌ PyTorch raised unexpected exception: {e}")
    
    if not pytorch_rejected:
        print("  ❌ PyTorch did NOT reject noncausal!")
    
    # 测试 Triton 版本
    triton_rejected = False
    try:
        attention_triton(q, k, v, causal, sm_scale, theta)
    except ValueError as e:
        if "causal" in str(e).lower():
            triton_rejected = True
            print(f"  ✅ Triton correctly rejects noncausal: {e}")
        else:
            print(f"  ❌ Triton raised unexpected error: {e}")
    except Exception as e:
        print(f"  ❌ Triton raised unexpected exception: {e}")
    
    if not triton_rejected:
        print("  ❌ Triton did NOT reject noncausal!")
    
    return pytorch_rejected and triton_rejected


def run_all_tests():
    """运行所有测试"""
    print("="*80)
    print("Co-RoPE PyTorch vs Triton 对拍测试")
    print("="*80)
    
    # 测试 noncausal 拒绝
    test_non_causal_rejection()
    
    # 测试配置列表
    test_configs = [
        # (B, H_Q, H_KV, N, D, dtype)
        (1, 4, 2, 64, 64, torch.float16),   # GQA: group_size=2
        (2, 8, 4, 128, 64, torch.float16),  # GQA: group_size=2
        (1, 8, 2, 64, 64, torch.float16),   # GQA: group_size=4
        (1, 4, 4, 64, 64, torch.float16),   # MHA: group_size=1
        (1, 4, 2, 32, 32, torch.float16),   # 小尺寸测试
    ]
    
    print("\n" + "="*80)
    print("前向传播测试")
    print("="*80)
    
    forward_passed = 0
    forward_total = 0
    
    for B, H_Q, H_KV, N, D, dtype in test_configs:
        print(f"\n[B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, dtype={dtype}]")
        forward_total += 1
        if compare_forward(B, H_Q, H_KV, N, D, dtype):
            forward_passed += 1
    
    print("\n" + "="*80)
    print("反向传播测试")
    print("="*80)
    
    backward_passed = 0
    backward_total = 0
    
    for B, H_Q, H_KV, N, D, dtype in test_configs:
        print(f"\n[B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, dtype={dtype}]")
        backward_total += 1
        if compare_backward(B, H_Q, H_KV, N, D, dtype):
            backward_passed += 1
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"前向传播: {forward_passed}/{forward_total} passed")
    print(f"反向传播: {backward_passed}/{backward_total} passed")
    
    if forward_passed == forward_total and backward_passed == backward_total:
        print("\n✨ 所有测试通过！")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查输出")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

