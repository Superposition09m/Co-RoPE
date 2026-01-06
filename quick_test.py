"""
快速测试：输出所有偏差信息，不因错误终止
"""

import torch
from flash_attn_co_rope_gqa_triton import attention as attention_triton
from corope_attn_gqa_pytorch import attention_pytorch

DEVICE = 'cuda'

def test_forward(B, H_Q, H_KV, N, D):
    """测试前向传播"""
    print(f"\n{'='*80}")
    print(f"前向测试: [B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}]")
    print(f"{'='*80}")
    
    torch.manual_seed(42)
    q = torch.randn(B, H_Q, N, D, device=DEVICE, dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H_KV, N, D, device=DEVICE, dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H_KV, N, D, device=DEVICE, dtype=torch.float16, requires_grad=True)
    
    theta = 10000.0
    sm_scale = 1.0
    
    try:
        print("  计算 PyTorch 参考...")
        out_pytorch = attention_pytorch(q, k, v, True, sm_scale, theta)
        print(f"    PyTorch output: shape={out_pytorch.shape}, mean={out_pytorch.mean():.6f}, std={out_pytorch.std():.6f}")
    except Exception as e:
        print(f"    ❌ PyTorch 失败: {e}")
        return
    
    try:
        print("  计算 Triton 实现...")
        out_triton = attention_triton(q, k, v, True, sm_scale, theta)
        print(f"    Triton output:  shape={out_triton.shape}, mean={out_triton.mean():.6f}, std={out_triton.std():.6f}")
    except Exception as e:
        print(f"    ❌ Triton 失败: {e}")
        return
    
    # 详细偏差分析
    diff = (out_pytorch - out_triton).abs().float()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    median_diff = diff.median().item()
    p95_diff = diff.quantile(0.95).item()
    p99_diff = diff.quantile(0.99).item()
    
    print(f"\n  偏差统计:")
    print(f"    Max Diff:    {max_diff:.6f}")
    print(f"    Mean Diff:   {mean_diff:.6f}")
    print(f"    Median Diff: {median_diff:.6f}")
    print(f"    P95 Diff:    {p95_diff:.6f}")
    print(f"    P99 Diff:    {p99_diff:.6f}")
    
    # 最大差异位置
    max_idx = diff.argmax()
    coords = torch.unravel_index(max_idx, diff.shape)
    print(f"\n  最大差异位置: b={coords[0]}, h={coords[1]}, i={coords[2]}, j={coords[3]}")
    print(f"    PyTorch: {out_pytorch[coords]:.6f}")
    print(f"    Triton:  {out_triton[coords]:.6f}")
    print(f"    Diff:    {diff[coords]:.6f}")
    
    # 相对误差
    rel_diff = diff / (out_pytorch.abs().float() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    print(f"\n  相对误差:")
    print(f"    Max Rel Diff:  {max_rel_diff:.6f}")
    print(f"    Mean Rel Diff: {mean_rel_diff:.6f}")
    
    return q, k, v, out_pytorch, out_triton

def test_backward(q, k, v, out_pytorch, out_triton):
    """测试反向传播"""
    print(f"\n{'='*80}")
    print(f"反向测试")
    print(f"{'='*80}")
    
    # 创建梯度
    grad_out = torch.randn_like(out_pytorch)
    
    # PyTorch backward
    try:
        print("  计算 PyTorch backward...")
        q_pytorch = q.clone().detach().requires_grad_(True)
        k_pytorch = k.clone().detach().requires_grad_(True)
        v_pytorch = v.clone().detach().requires_grad_(True)
        
        out_pytorch_grad = attention_pytorch(q_pytorch, k_pytorch, v_pytorch, True, 1.0, 10000.0)
        out_pytorch_grad.backward(grad_out)
        
        dq_pytorch = q_pytorch.grad
        dk_pytorch = k_pytorch.grad
        dv_pytorch = v_pytorch.grad
        
        print(f"    dQ: mean={dq_pytorch.mean():.6f}, std={dq_pytorch.std():.6f}")
        print(f"    dK: mean={dk_pytorch.mean():.6f}, std={dk_pytorch.std():.6f}")
        print(f"    dV: mean={dv_pytorch.mean():.6f}, std={dv_pytorch.std():.6f}")
    except Exception as e:
        print(f"    ❌ PyTorch backward 失败: {e}")
        return
    
    # Triton backward
    try:
        print("  计算 Triton backward...")
        q_triton = q.clone().detach().requires_grad_(True)
        k_triton = k.clone().detach().requires_grad_(True)
        v_triton = v.clone().detach().requires_grad_(True)
        
        out_triton_grad = attention_triton(q_triton, k_triton, v_triton, True, 1.0, 10000.0)
        out_triton_grad.backward(grad_out)
        
        dq_triton = q_triton.grad
        dk_triton = k_triton.grad
        dv_triton = v_triton.grad
        
        print(f"    dQ: mean={dq_triton.mean():.6f}, std={dq_triton.std():.6f}")
        print(f"    dK: mean={dk_triton.mean():.6f}, std={dk_triton.std():.6f}")
        print(f"    dV: mean={dv_triton.mean():.6f}, std={dv_triton.std():.6f}")
    except Exception as e:
        print(f"    ❌ Triton backward 失败: {e}")
        return
    
    # 梯度偏差分析
    print(f"\n  梯度偏差统计:")
    
    for name, grad_pytorch, grad_triton in [("dQ", dq_pytorch, dq_triton),
                                             ("dK", dk_pytorch, dk_triton),
                                             ("dV", dv_pytorch, dv_triton)]:
        if grad_pytorch is None or grad_triton is None:
            print(f"    {name}: 梯度为 None，跳过")
            continue
            
        diff_grad = (grad_pytorch - grad_triton).abs()
        max_diff_grad = diff_grad.max().item()
        mean_diff_grad = diff_grad.mean().item()
        
        print(f"    {name}:")
        print(f"      Max Diff:  {max_diff_grad:.6f}")
        print(f"      Mean Diff: {mean_diff_grad:.6f}")
        
        if max_diff_grad > 0.001:
            max_idx_grad = diff_grad.argmax()
            coords_grad = torch.unravel_index(max_idx_grad, diff_grad.shape)
            print(f"      最大差异位置: {coords_grad}")
            print(f"        PyTorch: {grad_pytorch[coords_grad]:.6f}")
            print(f"        Triton:  {grad_triton[coords_grad]:.6f}")

if __name__ == "__main__":
    # 小规模测试
    B, H_Q, H_KV, N, D = 1, 4, 2, 32, 64
    
    result = test_forward(B, H_Q, H_KV, N, D)
    if result:
        q, k, v, out_pytorch, out_triton = result
        test_backward(q, k, v, out_pytorch, out_triton)

