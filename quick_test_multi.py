"""
快速测试多个配置
"""

import torch
from flash_attn_co_rope_gqa_triton import attention as attention_triton
from corope_attn_gqa_pytorch import attention_pytorch

DEVICE = 'cuda'

def test_case(B, H_Q, H_KV, N, D):
    print(f"[B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}]", end=" ")
    
    torch.manual_seed(42)
    q = torch.randn(B, H_Q, N, D, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, H_KV, N, D, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, H_KV, N, D, device=DEVICE, dtype=torch.float16)
    
    theta = 10000.0
    sm_scale = 1.0
    
    try:
        out_pytorch = attention_pytorch(q, k, v, True, sm_scale, theta)
        out_triton = attention_triton(q, k, v, True, sm_scale, theta)
        
        diff = (out_pytorch - out_triton).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        if max_diff < 0.01:
            print(f"✅ max={max_diff:.4f}, mean={mean_diff:.6f}")
        else:
            print(f"❌ max={max_diff:.4f}, mean={mean_diff:.6f}")
    except Exception as e:
        print(f"❌ Error: {str(e)[:80]}")

if __name__ == "__main__":
    print("快速多case测试:")
    print("-" * 80)
    
    # 不同配置
    test_case(1, 2, 2, 32, 64)  # GROUP_SIZE=1
    test_case(1, 4, 2, 32, 64)  # GROUP_SIZE=2
    test_case(1, 8, 2, 32, 64)  # GROUP_SIZE=4
    test_case(1, 4, 4, 32, 64)  # GROUP_SIZE=1
    test_case(1, 8, 4, 32, 64)  # GROUP_SIZE=2
    test_case(1, 4, 2, 64, 64)  # 更长序列

