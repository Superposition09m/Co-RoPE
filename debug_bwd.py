"""
调试 Backward Pass 的 NaN 问题
"""

import torch
from flash_attn_rope_triton import attention

B, H, N, D = 1, 1, 64, 64
device = 'cuda'

torch.manual_seed(42)
q = torch.randn(B, H, N, D, device=device, dtype=torch.float16, requires_grad=True)
k = torch.randn(B, H, N, D, device=device, dtype=torch.float16, requires_grad=True)
v = torch.randn(B, H, N, D, device=device, dtype=torch.float16, requires_grad=True)
freqs_cos = torch.randn(N, D // 2, device=device, dtype=torch.float16)
freqs_sin = torch.randn(N, D // 2, device=device, dtype=torch.float16)

print("Forward...")
o = attention(q, k, v, False, 0.5, freqs_cos, freqs_sin, False)
print(f"Output: mean={o.mean().item():.6f}, std={o.std().item():.6f}, has_nan={torch.isnan(o).any().item()}")

print("\nBackward...")
loss = o.sum()
loss.backward()

print(f"\ndQ: mean={q.grad.mean().item() if q.grad is not None else 'None'}, has_nan={torch.isnan(q.grad).any().item() if q.grad is not None else 'N/A'}")
print(f"dK: mean={k.grad.mean().item() if k.grad is not None else 'None'}, has_nan={torch.isnan(k.grad).any().item() if k.grad is not None else 'N/A'}")
print(f"dV: mean={v.grad.mean().item() if v.grad is not None else 'None'}, has_nan={torch.isnan(v.grad).any().item() if v.grad is not None else 'N/A'}")

if q.grad is not None:
    print(f"\ndQ sample: {q.grad[0, 0, :5, :5]}")
if k.grad is not None:
    print(f"dK sample: {k.grad[0, 0, :5, :5]}")

