"""
Verification script to compare Triton and PyTorch implementations
对拍脚本：验证 Triton 实现和 PyTorch 实现的一致性
"""

import torch
import sys

# Import implementations
from fused_rope_attn import attention as attention_triton
from rope_attn_pytorch import attention_pytorch, precompute_freqs_cis


def test_attention(batch, n_heads, seq_len, head_dim, causal, sm_scale=0.5, theta=10000.0, device='cuda', dtype=torch.float16):
    """
    Test and compare Triton and PyTorch implementations
    
    Args:
        batch: batch size
        n_heads: number of attention heads
        seq_len: sequence length
        head_dim: head dimension
        causal: whether to use causal attention
        sm_scale: attention scale factor
        theta: RoPE base frequency
        device: device to run on
        dtype: data type
    """
    print(f"\n{'='*80}")
    print(f"Testing: batch={batch}, heads={n_heads}, seq_len={seq_len}, head_dim={head_dim}, causal={causal}")
    print(f"{'='*80}")
    
    torch.manual_seed(42)
    
    # Generate random inputs
    q = torch.randn((batch, n_heads, seq_len, head_dim), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((batch, n_heads, seq_len, head_dim), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((batch, n_heads, seq_len, head_dim), dtype=dtype, device=device, requires_grad=True)
    
    # Precompute RoPE frequencies
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len, theta, device=device)
    
    # Clone inputs for PyTorch version
    q_pt = q.clone().detach().requires_grad_(True)
    k_pt = k.clone().detach().requires_grad_(True)
    v_pt = v.clone().detach().requires_grad_(True)
    
    # ========== Forward Pass ==========
    print("\n[Forward Pass]")
    
    # Triton implementation
    out_triton = attention_triton(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, warp_specialize=False)
    
    # PyTorch implementation
    out_pytorch = attention_pytorch(q_pt, k_pt, v_pt, causal, sm_scale, freqs_cos, freqs_sin)
    
    # Compare outputs
    max_diff = (out_triton - out_pytorch).abs().max().item()
    mean_diff = (out_triton - out_pytorch).abs().mean().item()
    relative_diff = ((out_triton - out_pytorch).abs() / (out_pytorch.abs() + 1e-8)).mean().item()
    
    print(f"Output max diff: {max_diff:.6e}")
    print(f"Output mean diff: {mean_diff:.6e}")
    print(f"Output relative diff: {relative_diff:.6e}")
    
    fwd_pass = max_diff < 1e-2
    print(f"Forward pass: {'✓ PASS' if fwd_pass else '✗ FAIL'}")
    
    # ========== Backward Pass ==========
    print("\n[Backward Pass]")
    
    # Generate random gradient
    grad_out = torch.randn_like(out_triton)
    grad_out_pt = grad_out.clone()
    
    # Triton backward
    out_triton.backward(grad_out)
    dq_triton = q.grad.clone()
    dk_triton = k.grad.clone()
    dv_triton = v.grad.clone()
    
    # PyTorch backward
    out_pytorch.backward(grad_out_pt)
    dq_pytorch = q_pt.grad.clone()
    dk_pytorch = k_pt.grad.clone()
    dv_pytorch = v_pt.grad.clone()
    
    # Compare gradients
    def compare_grad(grad_triton, grad_pytorch, name):
        max_diff = (grad_triton - grad_pytorch).abs().max().item()
        mean_diff = (grad_triton - grad_pytorch).abs().mean().item()
        relative_diff = ((grad_triton - grad_pytorch).abs() / (grad_pytorch.abs() + 1e-8)).mean().item()
        
        print(f"\n{name}:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")
        print(f"  Relative diff: {relative_diff:.6e}")
        
        passed = max_diff < 1e-2
        print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
        return passed
    
    dq_pass = compare_grad(dq_triton, dq_pytorch, "dQ")
    dk_pass = compare_grad(dk_triton, dk_pytorch, "dK")
    dv_pass = compare_grad(dv_triton, dv_pytorch, "dV")
    
    bwd_pass = dq_pass and dk_pass and dv_pass
    
    # ========== Summary ==========
    print(f"\n{'='*80}")
    all_pass = fwd_pass and bwd_pass
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print(f"{'='*80}")
    
    return all_pass


def run_all_tests():
    """Run a suite of verification tests"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, running on CPU (may be slow)")
    
    print("\n" + "="*80)
    print("RoPE Attention Verification Suite")
    print("="*80)
    
    test_cases = [
        # (batch, n_heads, seq_len, head_dim, causal)
        (2, 4, 512, 64, False),   # Small non-causal
        (2, 4, 512, 64, True),    # Small causal
        (1, 8, 1024, 128, False), # Medium non-causal
        (1, 8, 1024, 128, True),  # Medium causal
        (4, 16, 2048, 64, False), # Large non-causal
        (4, 16, 2048, 64, True),  # Large causal
    ]
    
    results = []
    for batch, n_heads, seq_len, head_dim, causal in test_cases:
        try:
            passed = test_attention(batch, n_heads, seq_len, head_dim, causal, device=device)
            results.append((batch, n_heads, seq_len, head_dim, causal, passed))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((batch, n_heads, seq_len, head_dim, causal, False))
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    passed_count = sum(1 for r in results if r[-1])
    total_count = len(results)
    
    for batch, n_heads, seq_len, head_dim, causal, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: batch={batch}, heads={n_heads}, seq={seq_len}, dim={head_dim}, causal={causal}")
    
    print(f"\n{passed_count}/{total_count} tests passed")
    print("="*80)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

