import torch
import torch.nn.functional as F
import math

from utils import print_red_warning, calc_sim, assert_similar

class CoRoPEAttention(torch.autograd.Function):
    """
    CoRoPE Attention Function with manual backward pass
    当前实现：Plain Attention（后续会加 RoPE）
    """
    @staticmethod
    def forward(ctx, q, k, v, scale_factor, causal_mask):
        """
        Args:
            q: (B, n_head, T, head_dim)
            k: (B, n_head, T, head_dim)
            v: (B, n_head, T, head_dim)
            scale_factor: float
            causal_mask: (1, 1, T, T) or None
        Returns:
            output: (B, n_head, T, head_dim)
        """
        # Compute attention scores: Q @ K^T * scale_factor
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale_factor
        
        # Apply causal mask if needed
        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply mask to weights (set masked positions to 0)
        if causal_mask is not None:
            attn_weights = attn_weights.masked_fill(causal_mask, 0.0)
        
        # Attention output: attn_weights @ V
        output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        # Save for backward
        ctx.save_for_backward(q, k, v, attn_weights)
        ctx.scale_factor = scale_factor
        ctx.causal_mask = causal_mask
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: (B, n_head, T, head_dim) - gradient w.r.t. output
        Returns:
            dq, dk, dv: (B, n_head, T, head_dim)
            dscale_factor: None
            dcausal_mask: None
        """
        q, k, v, attn_weights = ctx.saved_tensors
        scale_factor = ctx.scale_factor
        causal_mask = ctx.causal_mask
        
        # Step 1: dV = attn_weights^T @ grad_output
        # attn_weights: (B, H, Q, K)
        # grad_output: (B, H, Q, D)
        # dv: (B, H, K, D)
        dv = torch.einsum('bhqk,bhqd->bhkd', attn_weights, grad_output)
        
        # Step 2: dS = grad_output @ V^T (gradient w.r.t. attn_weights)
        # grad_output: (B, H, Q, D)
        # v: (B, H, K, D)
        # ds: (B, H, Q, K)
        ds = torch.einsum('bhqd,bhkd->bhqk', grad_output, v)
        
        # Step 3: Softmax backward
        # Softmax backward formula: dp[i] = s[i] * (ds[i] - sum_j(s[j] * ds[j]))
        # where s = attn_weights, ds = gradient w.r.t. attn_weights
        d_softmax_sum = torch.sum(ds * attn_weights, dim=-1, keepdim=True)  # (B, H, Q, 1)
        dp = attn_weights * (ds - d_softmax_sum)  # (B, H, Q, K)
        
        # Apply mask to dp (gradient w.r.t. attn_scores)
        if causal_mask is not None:
            dp = dp.masked_fill(causal_mask, 0.0)
        
        # Step 4: dQ = dp @ K * scale_factor
        # dp: (B, H, Q, K)
        # k: (B, H, K, D)
        # dq: (B, H, Q, D)
        dq = torch.einsum('bhqk,bhkd->bhqd', dp, k) * scale_factor
        
        # Step 5: dK = dp^T @ Q * scale_factor
        # dp: (B, H, Q, K)
        # q: (B, H, Q, D)
        # dk: (B, H, K, D)
        dk = torch.einsum('bhqk,bhqd->bhkd', dp, q) * scale_factor
        
        return dq, dk, dv, None, None

class CoRoPE(torch.nn.Module):
    """
    Co-RoPE Attention Block:

    x: (B, T, n_embd), B is batch size, T is sequence length, n_embd is embedding dimension
    q: (B, T, n_head, head_dim), then transpose to (B, n_head, T, head_dim) for attention computation
    k: (B, T, n_kv_head, head_dim), then transpose to (B, n_kv_head, T, head_dim) for attention computation
    v: (B, T, n_kv_head, head_dim), then transpose to (B, n_kv_head, T, head_dim) for attention computation
    
    """
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int = None):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head  # Default MHA, also supports GQA
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        # Q projection: n_head heads
        self.q_proj = torch.nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        
        # K/V projection: n_kv_head heads (possibly fewer)
        self.k_proj = torch.nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        
        # Output projection: input is n_head * head_dim (because GQA will broadcast)
        self.o_proj = torch.nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, causal: bool = True):
        B, T, _ = x.shape

        # Q projection: (B, T, n_embd) -> (B, T, n_head * head_dim)
        q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.head_dim)
        q = q.transpose(1, 2)  # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        
        # K/V projection: (B, T, n_embd) -> (B, T, n_kv_head * head_dim)
        k = self.k_proj(x)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        k = k.transpose(1, 2)  # (B, T, n_kv_head, head_dim) -> (B, n_kv_head, T, head_dim)
        
        v = self.v_proj(x)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
        v = v.transpose(1, 2)  # (B, T, n_kv_head, head_dim) -> (B, n_kv_head, T, head_dim)
        
        # GQA broadcast k and v to match q
        if self.n_kv_head < self.n_head:
            num_groups = self.n_head // self.n_kv_head
            k = k.repeat_interleave(num_groups, dim=1)  # (B, n_head, T, head_dim)
            v = v.repeat_interleave(num_groups, dim=1)  # (B, n_head, T, head_dim)

        # Create causal mask if needed
        causal_mask = None
        if causal:
            mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
            causal_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Use CoRoPEAttention with manual backward
        y = CoRoPEAttention.apply(q, k, v, self.scale_factor, causal_mask)

        # Reshape back: (B, n_head, T, head_dim) -> (B, T, n_head * head_dim)
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, -1)
        y = self.o_proj(y)
        return y



def simple_test1():
    """Test the attention block forward pass
    """
    print("=" * 60)
    print("Testing CoRoPE Attention Block Forward Pass")
    print("=" * 60)
    
    # Test parameters
    B, T, n_embd = 2, 8, 64
    n_head = 4
    
    # Test 1: Standard MHA (n_kv_head = n_head)
    print("\n--- Test 1: Standard Multi-Head Attention (MHA) ---")
    model_mha = CoRoPE(n_embd=n_embd, n_head=n_head, n_kv_head=None)
    x = torch.randn(B, T, n_embd)
    
    output = model_mha(x, causal=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (B, T, n_embd), f"Expected output shape {(B, T, n_embd)}, got {output.shape}"
    print("✓ MHA forward pass: PASSED")
    
    # Test 2: GQA (n_kv_head < n_head)
    print("\n--- Test 2: Group-Query Attention (GQA) ---")
    n_kv_head = 2
    model_gqa = CoRoPE(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
    output_gqa = model_gqa(x, causal=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_gqa.shape}")
    assert output_gqa.shape == (B, T, n_embd), f"Expected output shape {(B, T, n_embd)}, got {output_gqa.shape}"
    print("✓ GQA forward pass: PASSED")
    
    # Test 3: Non-causal attention
    print("\n--- Test 3: Non-causal Attention ---")
    output_nc = model_mha(x, causal=False)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_nc.shape}")
    assert output_nc.shape == (B, T, n_embd), f"Expected output shape {(B, T, n_embd)}, got {output_nc.shape}"
    print("✓ Non-causal forward pass: PASSED")
    
    # Test 4: Check output is not NaN or Inf
    print("\n--- Test 4: Output Validity Check ---")
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"
    print("✓ Output validity: PASSED")
    
    # Test 5: Different sequence lengths
    print("\n--- Test 5: Different Sequence Lengths ---")
    T2 = 16
    x2 = torch.randn(B, T2, n_embd)
    output2 = model_mha(x2, causal=True)
    print(f"Input shape: {x2.shape}")
    print(f"Output shape: {output2.shape}")
    assert output2.shape == (B, T2, n_embd), f"Expected output shape {(B, T2, n_embd)}, got {output2.shape}"
    print("✓ Variable sequence length: PASSED")
    
    print("\n" + "=" * 60)
    print("All forward pass tests PASSED! ✓")
    print("=" * 60)
    


def simple_test2():
    """
    Test the handwritten backward pass against PyTorch autograd
    """
    print("=" * 60)
    print("Testing Handwritten Backward Pass")
    print("=" * 60)
    
    torch.manual_seed(42)
    B, T, n_embd = 2, 8, 64
    n_head = 4
    n_kv_head = 2  # Test GQA
    
    print(f"\nTest configuration: B={B}, T={T}, n_embd={n_embd}, n_head={n_head}, n_kv_head={n_kv_head}")
    
    # Test 1: Forward pass consistency
    print("\n--- Test 1: Forward Pass Consistency ---")
    q_test = torch.randn(B, n_head, T, n_embd // n_head, requires_grad=True, dtype=torch.float64)
    k_test = torch.randn(B, n_head, T, n_embd // n_head, requires_grad=True, dtype=torch.float64)
    v_test = torch.randn(B, n_head, T, n_embd // n_head, requires_grad=True, dtype=torch.float64)
    scale_factor = 1.0 / math.sqrt(n_embd // n_head)
    
    # Create causal mask
    mask = torch.triu(torch.ones(T, T, device=q_test.device, dtype=torch.bool), diagonal=1)
    causal_mask = mask.unsqueeze(0).unsqueeze(0)
    
    # Reference: using PyTorch autograd (manual computation)
    attn_scores_ref = torch.einsum('bhqd,bhkd->bhqk', q_test, k_test) * scale_factor
    attn_scores_ref = attn_scores_ref.masked_fill(causal_mask, float('-inf'))
    attn_weights_ref = F.softmax(attn_scores_ref, dim=-1)
    attn_weights_ref = attn_weights_ref.masked_fill(causal_mask, 0.0)
    output_ref = torch.einsum('bhqk,bhkd->bhqd', attn_weights_ref, v_test)
    
    # Our manual forward (using CoRoPEAttention)
    q_manual = q_test.detach().clone().requires_grad_(True)
    k_manual = k_test.detach().clone().requires_grad_(True)
    v_manual = v_test.detach().clone().requires_grad_(True)
    output_manual = CoRoPEAttention.apply(q_manual, k_manual, v_manual, scale_factor, causal_mask)
    
    # Check forward
    assert_similar(output_ref, output_manual, eps=1e-8, name="forward output")
    
    # Test 2: Backward pass consistency
    print("\n--- Test 2: Backward Pass Consistency ---")
    grad_output = torch.randn_like(output_ref)
    
    # Reference backward
    output_ref.backward(grad_output)
    dq_ref = q_test.grad.clone()
    dk_ref = k_test.grad.clone()
    dv_ref = v_test.grad.clone()
    
    # Manual backward
    output_manual.backward(grad_output)
    dq_manual = q_manual.grad.clone()
    dk_manual = k_manual.grad.clone()
    dv_manual = v_manual.grad.clone()
    
    # Check backward
    assert_similar(dq_ref, dq_manual, eps=1e-6, name="dq")
    assert_similar(dk_ref, dk_manual, eps=1e-6, name="dk")
    assert_similar(dv_ref, dv_manual, eps=1e-6, name="dv")
    
    # Test 3: Full model backward (including Linear layers)
    print("\n--- Test 3: Full Model Backward Pass ---")
    model = CoRoPE(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head).double()
    x = torch.randn(B, T, n_embd, dtype=torch.float64, requires_grad=True)
    
    output = model(x, causal=True)
    loss = (output * torch.randn_like(output)).sum()
    loss.backward()
    
    # Check all gradients are finite
    assert x.grad is not None, "Input gradient is None!"
    assert torch.isfinite(x.grad).all(), "Input gradient contains NaN or Inf!"
    print("✓ Input gradient is finite")
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None!"
        assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf!"
    print("✓ All parameter gradients are finite")
    
    print("\n" + "=" * 60)
    print("All backward pass tests PASSED! ✓")
    print("=" * 60)

if __name__ == "__main__":
    simple_test1()
    print("\n")
    simple_test2()
