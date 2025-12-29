"""
Test script to verify Triton RoPE fused Flash Attention against PyTorch baseline

This test compares the Triton implementation with the PyTorch reference implementation
for both forward and backward passes.
"""

import torch
from rope_attn_pytorch import precompute_freqs_cis, attention_pytorch
from flash_attn_rope_triton import attention as attention_triton


def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)


def test_rope_attention_single_config(BATCH, H, N_CTX, HEAD_DIM, causal, device='cuda', dtype=torch.float16):
    """
    Test a single configuration of RoPE attention

    Args:
        BATCH: batch size
        H: number of heads
        N_CTX: sequence length
        HEAD_DIM: head dimension
        causal: whether to use causal masking
        device: device to run on
        dtype: data type

    Returns:
        bool: True if all tests passed, False otherwise
    """
    print(f"\nTesting: BATCH={BATCH}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}, causal={causal}")
    print_separator('-')

    theta = 10000.0

    # Generate random inputs
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

    # Clone for separate gradient computation
    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)

    # Precompute RoPE frequencies
    freqs_cos, freqs_sin = precompute_freqs_cis(HEAD_DIM, N_CTX, theta, device=device)

    # Scale factor
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    # ========== Forward Pass ==========
    print("\n[Forward Pass]")

    # PyTorch reference
    out_ref = attention_pytorch(q_ref, k_ref, v_ref, causal, sm_scale, freqs_cos, freqs_sin)

    # Triton implementation (warp_specialize=False to avoid compiler bugs)
    out_tri = attention_triton(q, k, v, causal, sm_scale, freqs_cos, freqs_sin, False)

    # Compare outputs
    max_diff = torch.max(torch.abs(out_tri - out_ref)).item()
    mean_diff = torch.mean(torch.abs(out_tri - out_ref)).item()
    max_val = torch.max(torch.abs(out_ref)).item()

    print(f"  Output max diff: {max_diff:.6e}")
    print(f"  Output mean diff: {mean_diff:.6e}")
    print(f"  Output max value: {max_val:.6e}")
    print(f"  Relative error: {max_diff / (max_val + 1e-8):.6e}")

    # Check if forward pass is close
    fwd_passed = False
    try:
        torch.testing.assert_close(out_tri, out_ref, rtol=1e-2, atol=1e-2, check_dtype=False)
        print("  Status: PASSED")
        fwd_passed = True
    except AssertionError as e:
        print(f"  Status: FAILED")
        print(f"  Error: {str(e)[:200]}...")

    if not fwd_passed:
        return False

    # ========== Backward Pass ==========
    print("\n[Backward Pass]")

    # Generate random gradient
    grad_out = torch.randn_like(out_ref)

    # PyTorch reference backward
    out_ref.backward(grad_out)
    dq_ref = q_ref.grad.clone()
    dk_ref = k_ref.grad.clone()
    dv_ref = v_ref.grad.clone()

    # Triton backward
    out_tri.backward(grad_out)
    dq_tri = q.grad.clone()
    dk_tri = k.grad.clone()
    dv_tri = v.grad.clone()

    # Compare gradients
    all_passed = True

    # dQ comparison
    dq_max_diff = torch.max(torch.abs(dq_tri - dq_ref)).item()
    dq_mean_diff = torch.mean(torch.abs(dq_tri - dq_ref)).item()
    dq_max_val = torch.max(torch.abs(dq_ref)).item()

    print(f"\n  dQ:")
    print(f"    max diff: {dq_max_diff:.6e}")
    print(f"    mean diff: {dq_mean_diff:.6e}")
    print(f"    max value: {dq_max_val:.6e}")
    print(f"    relative error: {dq_max_diff / (dq_max_val + 1e-8):.6e}")

    try:
        torch.testing.assert_close(dq_tri, dq_ref, rtol=1e-2, atol=1e-2, check_dtype=False)
        print("    Status: PASSED")
    except AssertionError:
        print("    Status: FAILED")
        all_passed = False

    # dK comparison
    dk_max_diff = torch.max(torch.abs(dk_tri - dk_ref)).item()
    dk_mean_diff = torch.mean(torch.abs(dk_tri - dk_ref)).item()
    dk_max_val = torch.max(torch.abs(dk_ref)).item()

    print(f"\n  dK:")
    print(f"    max diff: {dk_max_diff:.6e}")
    print(f"    mean diff: {dk_mean_diff:.6e}")
    print(f"    max value: {dk_max_val:.6e}")
    print(f"    relative error: {dk_max_diff / (dk_max_val + 1e-8):.6e}")

    try:
        torch.testing.assert_close(dk_tri, dk_ref, rtol=1e-2, atol=1e-2, check_dtype=False)
        print("    Status: PASSED")
    except AssertionError:
        print("    Status: FAILED")
        all_passed = False

    # dV comparison
    dv_max_diff = torch.max(torch.abs(dv_tri - dv_ref)).item()
    dv_mean_diff = torch.mean(torch.abs(dv_tri - dv_ref)).item()
    dv_max_val = torch.max(torch.abs(dv_ref)).item()

    print(f"\n  dV:")
    print(f"    max diff: {dv_max_diff:.6e}")
    print(f"    mean diff: {dv_mean_diff:.6e}")
    print(f"    max value: {dv_max_val:.6e}")
    print(f"    relative error: {dv_max_diff / (dv_max_val + 1e-8):.6e}")

    try:
        torch.testing.assert_close(dv_tri, dv_ref, rtol=1e-2, atol=1e-2, check_dtype=False)
        print("    Status: PASSED")
    except AssertionError:
        print("    Status: FAILED")
        all_passed = False

    return all_passed


def test_rope_attention():
    """Test RoPE attention with multiple configurations"""

    # Test configurations: (BATCH, H, N_CTX, HEAD_DIM, causal)
    configs = [
        (1, 2, 128, 64, True),    # Small test
        (2, 4, 256, 64, False),   # Medium test, non-causal
        (1, 8, 512, 128, True),   # Larger test, causal
    ]

    device = 'cuda'
    dtype = torch.float16

    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print_separator()
    print("Triton RoPE Fused Flash Attention - Correctness Test")
    print_separator()

    passed_count = 0
    failed_count = 0

    for config in configs:
        BATCH, H, N_CTX, HEAD_DIM, causal = config

        try:
            passed = test_rope_attention_single_config(
                BATCH, H, N_CTX, HEAD_DIM, causal, device, dtype
            )

            if passed:
                print("\n  OVERALL: ALL TESTS PASSED")
                passed_count += 1
            else:
                print("\n  OVERALL: SOME TESTS FAILED")
                failed_count += 1

        except Exception as e:
            print(f"\n  OVERALL: EXCEPTION OCCURRED")
            print(f"  Error: {e}")
            failed_count += 1

        print_separator()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total configurations: {len(configs)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")

    if failed_count == 0:
        print("\nALL TESTS PASSED!")
    else:
        print(f"\nSOME TESTS FAILED ({failed_count}/{len(configs)})")

    print("="*80)

    return failed_count == 0


if __name__ == "__main__":
    success = test_rope_attention()
