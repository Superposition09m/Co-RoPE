"""
Simple test without warp specialization
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

# PyTorch version
print("\nRunning PyTorch version...")
out_pt = attention_pytorch(q_pt, k_pt, v_pt, CAUSAL, sm_scale, freqs_cos, freqs_sin)

# Triton version (computes RoPE on-the-fly with theta!) - DISABLE warp_specialize
print("Running Triton version (warp_specialize=False)...")
out_tr = attention_triton(q_tr, k_tr, v_tr, CAUSAL, sm_scale, THETA, False)

# Compare outputs (ensure same dtype)
out_pt = out_pt.to(torch.float32)
out_tr = out_tr.to(torch.float32)

abs_diff = torch.abs(out_pt - out_tr)
rel_diff = abs_diff / (torch.abs(out_pt) + 1e-8)

print(f"\nForward Pass Results:")
print(f"  Max Absolute Error: {abs_diff.max().item():.2e}")
print(f"  Mean Absolute Error: {abs_diff.mean().item():.2e}")
print(f"  Max Relative Error: {rel_diff.max().item():.2e}")
print(f"  Mean Relative Error: {rel_diff.mean().item():.2e}")

# Check if errors are within tolerance
atol, rtol = 1e-2, 1e-2
passed = torch.allclose(out_pt, out_tr, atol=atol, rtol=rtol)

if passed:
    print(f"\n✓ FORWARD PASS PASSED (atol={atol}, rtol={rtol})")
else:
    print(f"\n✗ FORWARD PASS FAILED")
    print(f"  Some values differ by more than tolerance")

