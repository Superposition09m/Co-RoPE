"""
Complete test for both forward and backward pass
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

print("\n" + "="*60)
print("RoPE Fused Kernel Test (warp_specialize=False)")
print("="*60)

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

# ========== FORWARD PASS ==========
print("\n" + "-"*60)
print("FORWARD PASS")
print("-"*60)

print("Running PyTorch version...")
out_pt = attention_pytorch(q_pt, k_pt, v_pt, CAUSAL, sm_scale, freqs_cos, freqs_sin)

print("Running Triton version...")
out_tr = attention_triton(q_tr, k_tr, v_tr, CAUSAL, sm_scale, THETA, False)

# Compare outputs (ensure same dtype)
out_pt_f32 = out_pt.to(torch.float32)
out_tr_f32 = out_tr.to(torch.float32)

abs_diff = torch.abs(out_pt_f32 - out_tr_f32)
rel_diff = abs_diff / (torch.abs(out_pt_f32) + 1e-8)

print(f"\nForward Results:")
print(f"  Max Absolute Error: {abs_diff.max().item():.2e}")
print(f"  Mean Absolute Error: {abs_diff.mean().item():.2e}")
print(f"  Max Relative Error: {rel_diff.max().item():.2e}")
print(f"  Mean Relative Error: {rel_diff.mean().item():.2e}")

atol, rtol = 1e-2, 1e-2
fwd_passed = torch.allclose(out_pt_f32, out_tr_f32, atol=atol, rtol=rtol)

if fwd_passed:
    print(f"✓ FORWARD PASSED (atol={atol}, rtol={rtol})")
else:
    print(f"✗ FORWARD FAILED")

# ========== BACKWARD PASS ==========
print("\n" + "-"*60)
print("BACKWARD PASS")
print("-"*60)

# Create same gradient
grad_output = torch.randn_like(out_pt)

# Backward pass
print("Running backward passes...")
out_pt.backward(grad_output)
out_tr.backward(grad_output)

# Compare gradients
def compare_grads(name, grad_pt, grad_tr):
    grad_pt_f32 = grad_pt.to(torch.float32)
    grad_tr_f32 = grad_tr.to(torch.float32)
    
    abs_diff = torch.abs(grad_pt_f32 - grad_tr_f32)
    rel_diff = abs_diff / (torch.abs(grad_pt_f32) + 1e-8)
    print(f"\n  {name}:")
    print(f"    Max Abs Error: {abs_diff.max().item():.2e}")
    print(f"    Mean Abs Error: {abs_diff.mean().item():.2e}")
    print(f"    Max Rel Error: {rel_diff.max().item():.2e}")
    print(f"    Mean Rel Error: {rel_diff.mean().item():.2e}")
    return abs_diff, rel_diff

print("\nBackward Results:")
abs_diff_dq, rel_diff_dq = compare_grads("dQ", q_pt.grad, q_tr.grad)
abs_diff_dk, rel_diff_dk = compare_grads("dK", k_pt.grad, k_tr.grad)
abs_diff_dv, rel_diff_dv = compare_grads("dV", v_pt.grad, v_tr.grad)

# Check if errors are within tolerance
atol_bwd, rtol_bwd = 1e-2, 5e-2
bwd_passed = (
    torch.allclose(q_pt.grad.to(torch.float32), q_tr.grad.to(torch.float32), atol=atol_bwd, rtol=rtol_bwd) and
    torch.allclose(k_pt.grad.to(torch.float32), k_tr.grad.to(torch.float32), atol=atol_bwd, rtol=rtol_bwd) and
    torch.allclose(v_pt.grad.to(torch.float32), v_tr.grad.to(torch.float32), atol=atol_bwd, rtol=rtol_bwd)
)

if bwd_passed:
    print(f"\n✓ BACKWARD PASSED (atol={atol_bwd}, rtol={rtol_bwd})")
else:
    print(f"\n✗ BACKWARD FAILED")

# ========== SUMMARY ==========
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Forward:  {'PASS ✓' if fwd_passed else 'FAIL ✗'}")
print(f"  Backward: {'PASS ✓' if bwd_passed else 'FAIL ✗'}")

if fwd_passed and bwd_passed:
    print("\n🎉 ALL TESTS PASSED!")
else:
    print("\n⚠ SOME TESTS FAILED")

print("="*60 + "\n")

