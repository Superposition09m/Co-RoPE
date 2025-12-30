"""
CoRoPE 梯度正确性测试

验证手动实现的 backward 和 PyTorch autograd 的结果一致
"""

import torch
import sys
from corope_attn_gqa_pytorch import attention_pytorch


def numerical_gradient(func, inputs, eps=1e-4):
    """
    使用有限差分法计算数值梯度
    
    Args:
        func: 输出标量的函数
        inputs: 输入张量列表
        eps: 扰动大小
    
    Returns:
        gradients: 数值梯度列表
    """
    gradients = []
    
    for input_tensor in inputs:
        if not input_tensor.requires_grad:
            gradients.append(None)
            continue
        
        grad = torch.zeros_like(input_tensor)
        
        # 遍历每个元素
        it = torch.nditer(input_tensor.cpu().numpy(), flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            
            # f(x + eps)
            input_tensor.data[idx] += eps
            loss_plus = func()
            
            # f(x - eps)
            input_tensor.data[idx] -= 2 * eps
            loss_minus = func()
            
            # f(x) 恢复
            input_tensor.data[idx] += eps
            
            # 中心差分：(f(x+eps) - f(x-eps)) / (2*eps)
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            
            it.iternext()
        
        gradients.append(grad)
    
    return gradients


def test_corop_gradient_simple(B=1, H_q=4, H_kv=2, N=32, D=64, causal=True):
    """
    简化的梯度测试：使用 torch.autograd.gradcheck
    """
    print('='*80)
    print(f'CoRoPE 梯度测试（torch.autograd.gradcheck）')
    print('='*80)
    print(f'配置: B={B}, H_q={H_q}, H_kv={H_kv}, N={N}, D={D}, causal={causal}')
    print(f'GQA group_size: {H_q // H_kv}')
    
    device = 'cuda'
    theta = 10000.0
    sm_scale = 1.0 / (D ** 0.5)
    
    # 创建小规模输入（gradcheck 对大张量很慢）
    q = torch.randn(B, H_q, N, D, device=device, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float64, requires_grad=True)
    
    # 定义测试函数
    def func(q_in, k_in, v_in):
        return attention_pytorch(q_in, k_in, v_in, causal, sm_scale, theta)
    
    print('\n执行 gradcheck（这可能需要几分钟...）')
    
    try:
        # PyTorch 的 gradcheck 使用有限差分验证梯度
        # eps: 扰动大小
        # atol: 绝对误差容忍度
        result = torch.autograd.gradcheck(
            func,
            (q, k, v),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-2,
            raise_exception=False,
        )
        
        if result:
            print('✅ gradcheck PASSED: 手动梯度与数值梯度一致！')
            return True
        else:
            print('❌ gradcheck FAILED: 梯度不匹配')
            return False
            
    except Exception as e:
        print(f'❌ gradcheck 出错: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_corope_gradient_manual(B=2, H_q=4, H_kv=2, N=64, D=64, causal=True):
    """
    手动梯度对比测试：对比自动求导和手动实现
    """
    print('\n' + '='*80)
    print('CoRoPE 梯度手动验证（对比 autograd）')
    print('='*80)
    print(f'配置: B={B}, H_q={H_q}, H_kv={H_kv}, N={N}, D={D}, causal={causal}')
    print(f'GQA group_size: {H_q // H_kv}')
    
    device = 'cuda'
    dtype = torch.float32
    theta = 10000.0
    sm_scale = 1.0 / (D ** 0.5)
    
    # ========== 方法 1: 使用我们的手动 backward ==========
    print('\n[方法 1: 手动 Backward]')
    q1 = torch.randn(B, H_q, N, D, device=device, dtype=dtype, requires_grad=True)
    k1 = torch.randn(B, H_kv, N, D, device=device, dtype=dtype, requires_grad=True)
    v1 = torch.randn(B, H_kv, N, D, device=device, dtype=dtype, requires_grad=True)
    
    output1 = attention_pytorch(q1, k1, v1, causal, sm_scale, theta)
    
    # 随机梯度
    grad_out = torch.randn_like(output1)
    output1.backward(grad_out)
    
    dq1 = q1.grad.clone()
    dk1 = k1.grad.clone()
    dv1 = v1.grad.clone()
    
    print(f'  dQ: mean={dq1.mean().item():.6e}, std={dq1.std().item():.6e}')
    print(f'  dK: mean={dk1.mean().item():.6e}, std={dk1.std().item():.6e}')
    print(f'  dV: mean={dv1.mean().item():.6e}, std={dv1.std().item():.6e}')
    
    # ========== 方法 2: 使用 PyTorch autograd（不调用手动 backward）==========
    print('\n[方法 2: 纯 Autograd（作为参考）]')
    
    # 重新实现一个简化版本，让 PyTorch 自动求导
    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)
    
    # 使用相同的 forward 逻辑，但让 PyTorch 自动求导
    # 复制 forward 的核心计算（不使用自定义 Function）
    with torch.enable_grad():
        # GQA expansion
        if H_q == H_kv:
            k_exp2 = k2
            v_exp2 = v2
        else:
            group_size = H_q // H_kv
            k_exp2 = k2.view(B, H_kv, 1, N, D).expand(B, H_kv, group_size, N, D).reshape(B, H_q, N, D)
            v_exp2 = v2.view(B, H_kv, 1, N, D).expand(B, H_kv, group_size, N, D).reshape(B, H_q, N, D)
        
        # 计算步长能量
        z_scores2 = torch.einsum('bhqd,bhkd->bhqk', q2, k_exp2) * sm_scale
        z2 = torch.sigmoid(z_scores2)
        
        # 计算累积里程
        a_q2 = torch.cumsum(z2, dim=-1)
        a_q_total2 = torch.diagonal(a_q2, dim1=-2, dim2=-1)
        z_avg2 = z2.mean(dim=2, keepdim=True)
        a_k2 = torch.cumsum(z_avg2.squeeze(2), dim=-1)
        
        # 里程差
        mileage_diff2 = a_k2.unsqueeze(2) - a_q_total2.unsqueeze(3)
        
        # 频率
        inv_freq2 = 1.0 / (theta ** (torch.arange(0, D, 2, device=device).float() / D))
        
        # 旋转角度
        angles2 = mileage_diff2.unsqueeze(-1) * inv_freq2
        cos_m2 = torch.cos(angles2)
        sin_m2 = torch.sin(angles2)
        
        # Split layout
        half_dim = D // 2
        q1_2, q2_2 = q2[..., :half_dim], q2[..., half_dim:]
        k1_2, k2_2 = k_exp2[..., :half_dim], k_exp2[..., half_dim:]
        
        # 扩展维度
        q1_e = q1_2.unsqueeze(3)
        q2_e = q2_2.unsqueeze(3)
        k1_e = k1_2.unsqueeze(2)
        k2_e = k2_2.unsqueeze(2)
        
        # 旋转点积
        real2 = q1_e * k1_e + q2_e * k2_e
        imag2 = q2_e * k1_e - q1_e * k2_e
        rotated2 = real2 * cos_m2 - imag2 * sin_m2
        attn_scores2 = rotated2.sum(dim=-1)
        
        # Causal mask
        if causal:
            mask2 = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
            attn_scores2 = attn_scores2.masked_fill(mask2.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax
        attn_weights2 = torch.nn.functional.softmax(attn_scores2, dim=-1, dtype=torch.float32).to(dtype)
        
        # Output
        output2 = torch.einsum('bhqk,bhkd->bhqd', attn_weights2, v_exp2)
    
    # Backward
    output2.backward(grad_out)
    
    dq2 = q2.grad.clone()
    dk2 = k2.grad.clone()
    dv2 = v2.grad.clone()
    
    print(f'  dQ: mean={dq2.mean().item():.6e}, std={dq2.std().item():.6e}')
    print(f'  dK: mean={dk2.mean().item():.6e}, std={dk2.std().item():.6e}')
    print(f'  dV: mean={dv2.mean().item():.6e}, std={dv2.std().item():.6e}')
    
    # ========== 对比梯度 ==========
    print('\n' + '='*80)
    print('梯度对比')
    print('='*80)
    
    def compare_gradients(g1, g2, name):
        abs_diff = torch.abs(g1 - g2)
        rel_diff = abs_diff / (torch.abs(g2) + 1e-8)
        
        print(f'\n{name}:')
        print(f'  Max Abs Error:  {abs_diff.max().item():.6e}')
        print(f'  Mean Abs Error: {abs_diff.mean().item():.6e}')
        print(f'  Max Rel Error:  {rel_diff.max().item():.6e}')
        print(f'  Mean Rel Error: {rel_diff.mean().item():.6e}')
        
        # 判断是否通过
        passed = abs_diff.max().item() < 1e-3 and rel_diff.max().item() < 0.1
        if passed:
            print(f'  ✅ {name} 梯度一致')
        else:
            print(f'  ❌ {name} 梯度不匹配')
        
        return passed
    
    dq_pass = compare_gradients(dq1, dq2, 'dQ')
    dk_pass = compare_gradients(dk1, dk2, 'dK')
    dv_pass = compare_gradients(dv1, dv2, 'dV')
    
    all_passed = dq_pass and dk_pass and dv_pass
    
    print('\n' + '='*80)
    if all_passed:
        print('🎉 所有梯度测试通过！手动 backward 实现正确！')
    else:
        print('❌ 部分梯度测试失败，需要检查 backward 实现')
    print('='*80)
    
    return all_passed


def test_corope_output_consistency():
    """
    测试两种方式的输出是否一致（作为前置检查）
    """
    print('\n' + '='*80)
    print('前置检查：Forward 输出一致性')
    print('='*80)
    
    B, H_q, H_kv, N, D = 2, 4, 2, 64, 64
    device = 'cuda'
    dtype = torch.float32
    theta = 10000.0
    sm_scale = 1.0 / (D ** 0.5)
    
    torch.manual_seed(42)
    q = torch.randn(B, H_q, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H_kv, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H_kv, N, D, device=device, dtype=dtype)
    
    # 方法1：使用自定义 Function
    output1 = attention_pytorch(q, k, v, True, sm_scale, theta)
    
    # 方法2：应该得到相同结果（因为用的是同一套 forward 逻辑）
    q2 = q.clone()
    k2 = k.clone()
    v2 = v.clone()
    output2 = attention_pytorch(q2, k2, v2, True, sm_scale, theta)
    
    diff = torch.abs(output1 - output2)
    print(f'  Output max diff: {diff.max().item():.6e}')
    print(f'  Output mean diff: {diff.mean().item():.6e}')
    
    if diff.max().item() < 1e-6:
        print('  ✅ Forward 输出一致')
        return True
    else:
        print('  ❌ Forward 输出不一致（这不应该发生）')
        return False


def main():
    """运行所有梯度测试"""
    print('='*80)
    print('CoRoPE GQA 梯度正确性测试套件')
    print('='*80)
    
    # 前置检查
    if not test_corope_output_consistency():
        print('\n❌ Forward 输出不一致，停止测试')
        sys.exit(1)
    
    # 测试不同配置
    configs = [
        # (B, H_q, H_kv, N, D, causal, name)
        (1, 4, 4, 32, 64, True, 'MHA-Small'),
        (2, 4, 2, 32, 64, True, 'GQA-Small'),
        (1, 8, 2, 64, 64, True, 'GQA-Medium'),
        (1, 4, 4, 32, 64, False, 'MHA-No-Causal'),
    ]
    
    all_passed = True
    
    for B, H_q, H_kv, N, D, causal, name in configs:
        print(f'\n{"#"*80}')
        print(f'测试配置: {name}')
        print(f'{"#"*80}')
        
        try:
            passed = test_corope_gradient_manual(B, H_q, H_kv, N, D, causal)
            if not passed:
                all_passed = False
        except Exception as e:
            print(f'❌ 配置 {name} 测试失败: {e}')
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # 最终总结
    print('\n' + '='*80)
    print('最终结果')
    print('='*80)
    
    if all_passed:
        print('🎉🎉🎉 所有梯度测试通过！')
        print('CoRoPE 的手动 backward 实现完全正确！')
        return 0
    else:
        print('❌ 部分测试失败，需要修复 backward 实现')
        return 1


if __name__ == '__main__':
    sys.exit(main())

