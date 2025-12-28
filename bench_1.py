"""
Benchmark script to compare PyTorch and Triton implementations of attention

Usage:
    python bench_1.py --gpu 0  # Use GPU 0 (default)
    python bench_1.py --gpu 1  # Use GPU 1

Note:
    - PyTorch implementation uses naive attention (full attention matrix)
    - Memory usage: O(BATCH * N_HEADS * N_CTX^2)
    - Triton implementation uses tiled/fused attention (memory efficient)
    - For large sequence lengths (>4096), consider using Triton-only benchmarks
"""

import torch
import triton
import triton.testing
import argparse
from attn_pytorch import attention_pytorch
from flash_attn_v2_triton import attention as attention_triton
from utils import assert_similar

# Parse command line arguments
parser = argparse.ArgumentParser(description='Benchmark PyTorch vs Triton Attention')
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')
args = parser.parse_args()

# Set GPU device
GPU_ID = args.gpu
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)
    print(f"Using GPU {GPU_ID}: {torch.cuda.get_device_name(GPU_ID)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(GPU_ID).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA is not available!")

DEVICE = triton.runtime.driver.active.get_active_torch_device()
# Reduced default values to avoid OOM with naive PyTorch implementation
# For larger scales, use Triton implementation only
BATCH, N_HEADS = 2, 16


def test_correctness():
    """
    Test correctness of PyTorch vs Triton implementations
    Verifies both forward and backward passes
    """
    print("=" * 60)
    print("Testing Correctness: PyTorch vs Triton")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {"BATCH": 2, "H": 4, "N_CTX": 128, "HEAD_DIM": 64, "causal": True},
        {"BATCH": 2, "H": 4, "N_CTX": 128, "HEAD_DIM": 64, "causal": False},
        {"BATCH": 4, "H": 8, "N_CTX": 256, "HEAD_DIM": 128, "causal": True},
        {"BATCH": 4, "H": 8, "N_CTX": 256, "HEAD_DIM": 128, "causal": False},
    ]
    
    for i, config in enumerate(test_configs):
        BATCH = config["BATCH"]
        H = config["H"]
        N_CTX = config["N_CTX"]
        HEAD_DIM = config["HEAD_DIM"]
        causal = config["causal"]
        
        print(f"\n--- Test {i+1}/{len(test_configs)}: B={BATCH}, H={H}, N_CTX={N_CTX}, "
              f"HEAD_DIM={HEAD_DIM}, causal={causal} ---")
        
        # Set random seed for reproducibility
        torch.manual_seed(42 + i)
        
        dtype = torch.float16
        sm_scale = 1.3
        
        # Create input tensors (shared for both implementations)
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE, requires_grad=True)
        
        # PyTorch forward
        q_pt = q.detach().clone().requires_grad_(True)
        k_pt = k.detach().clone().requires_grad_(True)
        v_pt = v.detach().clone().requires_grad_(True)
        out_pt = attention_pytorch(q_pt, k_pt, v_pt, causal, sm_scale)
        
        # Triton forward
        q_tr = q.detach().clone().requires_grad_(True)
        k_tr = k.detach().clone().requires_grad_(True)
        v_tr = v.detach().clone().requires_grad_(True)
        out_tr = attention_triton(q_tr, k_tr, v_tr, causal, sm_scale, False)  # warp_specialize=False
        
        # Check forward pass
        assert_similar(out_pt, out_tr, eps=1e-2, name=f"forward output (config {i+1})")
        print(f"✓ Forward pass: PASSED")
        
        # Backward pass
        grad_output = torch.randn_like(out_pt)
        
        # PyTorch backward
        out_pt.backward(grad_output)
        dq_pt = q_pt.grad.clone()
        dk_pt = k_pt.grad.clone()
        dv_pt = v_pt.grad.clone()
        
        # Triton backward
        out_tr.backward(grad_output)
        dq_tr = q_tr.grad.clone()
        dk_tr = k_tr.grad.clone()
        dv_tr = v_tr.grad.clone()
        
        # Check backward pass
        assert_similar(dq_pt, dq_tr, eps=1e-2, name=f"dq (config {i+1})")
        assert_similar(dk_pt, dk_tr, eps=1e-2, name=f"dk (config {i+1})")
        assert_similar(dv_pt, dv_tr, eps=1e-2, name=f"dv (config {i+1})")
        print(f"✓ Backward pass: PASSED")
    
    print("\n" + "=" * 60)
    print("All correctness tests PASSED! ✓")
    print("=" * 60)
    print()


configs = []
for HEAD_DIM in [64, 128]:
    for mode in ["fwd", "bwd"]:
        for causal in [True, False]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    # Adjusted range: 2^10 to 2^13 (1024 to 8192)
                    # PyTorch naive implementation allocates full attention matrix
                    x_vals=[2**i for i in range(10, 14)],
                    line_arg="provider",
                    line_vals=["pytorch", "triton"],
                    line_names=["PyTorch", "Triton"],
                    styles=[("blue", "-"), ("red", "-")],
                    ylabel="Time (ms)",
                    plot_name=
                    f"attention-comparison-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "HEAD_DIM": HEAD_DIM,
                        "mode": mode,
                        "causal": causal,
                    },
                ))


@triton.testing.perf_report(configs)
def bench_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device=DEVICE):
    """
    Benchmark function to compare PyTorch and Triton attention implementations
    
    Args:
        BATCH: batch size
        H: number of heads
        N_CTX: sequence length
        HEAD_DIM: dimension of each head
        causal: whether to use causal masking
        mode: "fwd" or "bwd"
        provider: "pytorch" or "triton"
    
    Returns:
        Median runtime in milliseconds (50th percentile)
    """
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    
    # Create input tensors
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    sm_scale = 1.3
    
    # Select the appropriate function
    if provider == "pytorch":
        fn = lambda: attention_pytorch(q, k, v, causal, sm_scale)
    elif provider == "triton":
        fn = lambda: attention_triton(q, k, v, causal, sm_scale, False)  # warp_specialize=False
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Backward mode: need to compute gradient
    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
    
    # Benchmark: measure runtime in milliseconds using median (50th percentile)
    # quantiles parameter ensures we use median instead of mean
    # This avoids making assumptions about PyTorch internals
    ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    
    # Return median runtime (50th percentile) - most reliable metric
    return ms[0]


if __name__ == "__main__":
    # First, test correctness
    test_correctness()
    
    # Then, run benchmarks
    print("=" * 60)
    print("Benchmarking PyTorch vs Triton Attention")
    print("=" * 60)
    bench_attention.run(save_path="./eval_data", print_data=True)

