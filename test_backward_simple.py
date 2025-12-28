"""
Simple test for backward pass without warp specialization
"""

import torch
from rope_attn_pytorch import precompute_freqs_cis, attention_pytorch
from flash_attn_rope_triton import attention as attention_triton

# Test configuration
BATCH, HEADS, SEQ_LEN, HEAD_DIM = 2, 4, 128, 64
THETA = 10000.0
CAUSAL = True
device = 'cuda'

torch.manual_seed(42)

# Create inputs
q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16, requires_grad=True)
k = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16, requires_grad=True)
v = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float16, requires_grad=True)

# Clone for separate tests
q_pt, k_pt, v_pt = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)
q_tr, k_tr, v_tr = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True)

# Precompute RoPE frequencies (only for PyTorch version)
freqs_cos, freqs_sin = precompute_freqs_cis(HEAD_DIM, SEQ_LEN, THETA, device)

sm_scale = 1.0 / (HEAD_DIM ** 0.5)

# Forward pass
print("\nRunning forward passes...")
out_pt = attention_pytorch(q_pt, k_pt, v_pt, CAUSAL, sm_scale, freqs_cos, freqs_sin)
out_tr = attention_triton(q_tr, k_tr, v_tr, CAUSAL, sm_scale, THETA, False)

# Create same gradient
grad_output = torch.randn_like(out_pt)

# Backward pass
print("Running backward passes...")
out_pt.backward(grad_output)
out_tr.backward(grad_output)

# Compare gradients
print(f"\nBackward Pass Results:")

def compare_grads(name, grad_pt, grad_tr):
    # Convert to same dtype
    grad_pt = grad_pt.to(torch.float32)
    grad_tr = grad_tr.to(torch.float32)
    
    abs_diff = torch.abs(grad_pt - grad_tr)
    rel_diff = abs_diff / (torch.abs(grad_pt) + 1e-8)
    print(f"\n  {name}:")
    print(f"    Max Absolute Error: {abs_diff.max().item():.2e}")
    print(f"    Mean Absolute Error: {abs_diff.mean().item():.2e}")
    print(f"    Max Relative Error: {rel_diff.max().item():.2e}")
    print(f"    Mean Relative Error: {rel_diff.mean().item():.2e}")
    return abs_diff.max().item(), rel_diff.max().item()

max_abs_dq, max_rel_dq = compare_grads("dQ", q_pt.grad, q_tr.grad)
max_abs_dk, max_rel_dk = compare_grads("dK", k_pt.grad, k_tr.grad)
max_abs_dv, max_rel_dv = compare_grads("dV", v_pt.grad, v_tr.grad)

# Check if errors are within tolerance
atol, rtol = 1e-2, 5e-2
passed = (
    torch.allclose(q_pt.grad.to(torch.float32), q_tr.grad.to(torch.float32), atol=atol, rtol=rtol) and
    torch.allclose(k_pt.grad.to(torch.float32), k_tr.grad.to(torch.float32), atol=atol, rtol=rtol) and
    torch.allclose(v_pt.grad.to(torch.float32), v_tr.grad.to(torch.float32), atol=atol, rtol=rtol)
)

if passed:
    print(f"\n✓ BACKWARD PASS PASSED (atol={atol}, rtol={rtol})")
else:
    print(f"\n✗ BACKWARD PASS FAILED")
    print(f"  Some gradients differ by more than tolerance")

