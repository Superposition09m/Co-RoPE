"""
PyTorch implementation of RoPE attention for benchmarking
This is used as a baseline to compare with Triton implementation
"""

import torch
import torch.nn.functional as F

class _attention_pytorch(torch.autograd.Function):
    """
    Plain PyTorch Attention with manual backward pass
    Interface compatible with flash_attn_v2_triton.py
    """
    
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, theta):
        """
        Args:
            q: (BATCH, H, N_CTX, HEAD_DIM)
            k: (BATCH, H, N_CTX, HEAD_DIM)
            v: (BATCH, H, N_CTX, HEAD_DIM)
            causal: bool
            sm_scale: float, scaling factor for attention scores
            theta: float, RoPE base (e.g., 10000.0)
        Returns:
            output: (BATCH, H, N_CTX, HEAD_DIM)
        """
        if not causal:
            raise ValueError("CoRoPE PyTorch implementation supports causal=True only.")
        
        B, n_heads_q, N_CTX, HEAD_DIM = q.shape
        device = q.device

        # Compute RoPE frequencies dynamically
        inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))

        n_heads_kv = k.shape[1]
        if n_heads_q == n_heads_kv:
            group_size = 1
            k_expanded = k
            v_expanded = v
        else:
            if n_heads_q % n_heads_kv != 0:
                raise ValueError(
                    f"Number of Q heads ({n_heads_q}) must be divisible by KV heads ({n_heads_kv})."
                )
            group_size = n_heads_q // n_heads_kv
            B, _, N_CTX, HEAD_DIM = k.shape
            k_expanded = k.view(B, n_heads_kv, 1, N_CTX, HEAD_DIM).expand(
                B, n_heads_kv, group_size, N_CTX, HEAD_DIM
            ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
            v_expanded = v.view(B, n_heads_kv, 1, N_CTX, HEAD_DIM).expand(
                B, n_heads_kv, group_size, N_CTX, HEAD_DIM
            ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
        
        # --- Co-RoPE Leader-Driven Logic: BEGIN ---
        # 1. 采样领航头并计算组共享里程
        q_leaders = q[:, ::group_size, :, :]
        z_leader = torch.sigmoid(torch.einsum('bhqd,bhkd->bhqk', q_leaders, k) * sm_scale)
        a_leader = torch.cumsum(z_leader, dim=-1)
        
        # 2. 广播里程分布并计算里程差 delta_a
        a = a_leader.repeat_interleave(group_size, dim=1)
        delta_a = torch.diagonal(a, dim1=-2, dim2=-1).unsqueeze(-1) - a
        # --- Co-RoPE Leader-Driven Logic: END ---
        
        d_half = HEAD_DIM // 2
        q1, q2 = q[..., :d_half], q[..., d_half:]
        k1, k2 = k_expanded[..., :d_half], k_expanded[..., d_half:]
        
        E_A = q1.unsqueeze(3) * k1.unsqueeze(2) + q2.unsqueeze(3) * k2.unsqueeze(2)
        E_B = q2.unsqueeze(3) * k1.unsqueeze(2) - q1.unsqueeze(3) * k2.unsqueeze(2)
        
        phi = delta_a.unsqueeze(-1) * inv_freq.view(1, 1, 1, 1, -1)
        
        # Compute attention scores: Q @ K^T * sm_scale
        # Use fp32 for dot product accumulation to avoid precision loss in long sequences
        attn_scores = (E_A * torch.cos(phi) - E_B * torch.sin(phi)).sum(dim=-1) * sm_scale
        
        # Apply causal mask if needed
        if causal:
            N_CTX = q.shape[2]
            mask = torch.triu(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool), diagonal=1)
            causal_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N_CTX, N_CTX)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Safe Softmax: manually implement to ensure fp32 precision
        # Step 1: Find row-wise maximum for numerical stability
        attn_scores_max = torch.max(attn_scores, dim=-1, keepdim=True).values
        # Handle -inf case (all masked) to avoid NaN in subtraction
        if causal:
            attn_scores_max = torch.where(
                torch.isinf(attn_scores_max), 
                torch.zeros_like(attn_scores_max), 
                attn_scores_max
            )
        
        # Step 2: Subtract max (safe softmax trick)
        attn_scores_shifted = attn_scores - attn_scores_max
        
        # Step 3: Exponentiate (all in fp32)
        attn_scores_exp = torch.exp(attn_scores_shifted)
        
        # Step 4: Apply mask to exp scores (set masked positions to 0)
        if causal:
            attn_scores_exp = attn_scores_exp.masked_fill(causal_mask, 0.0)
        
        # Step 5: Sum exponentials (in fp32 to avoid overflow)
        attn_scores_sum = torch.sum(attn_scores_exp, dim=-1, keepdim=True)
        
        # Step 6: Normalize to get attention weights
        attn_weights = attn_scores_exp / attn_scores_sum
        
        # Apply mask to weights (set masked positions to 0)
        if causal:
            attn_weights = attn_weights.masked_fill(causal_mask, 0.0)
        
        # Attention output: attn_weights @ V
        # Use fp32 for weighted sum accumulation to avoid precision loss in long sequences
        output = torch.einsum(
            'bhqk,bhkd->bhqd',
            attn_weights.to(torch.float32),
            v_expanded.to(torch.float32)
        ).to(q.dtype)

        if ctx is not None:
            ctx.save_for_backward(q, k, v, attn_weights)
            ctx.sm_scale = sm_scale
            ctx.causal = causal
            ctx.group_size = group_size
            ctx.n_kv_heads = n_heads_kv
            ctx.theta = theta

        return output

    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: (BATCH, H, N_CTX, HEAD_DIM) - gradient w.r.t. output
        Returns:
            dq, dk, dv: (BATCH, H, N_CTX, HEAD_DIM)
            dcausal: None
            dsm_scale: None
            dfreqs_cos: None
            dfreqs_sin: None
        """
        q, k, v, attn_weights = ctx.saved_tensors
        needs_dq, needs_dk, needs_dv, _, _, _ = ctx.needs_input_grad

        if not (needs_dq or needs_dk or needs_dv):
            return None, None, None, None, None, None

        sm_scale = ctx.sm_scale
        causal = ctx.causal
        group_size = ctx.group_size
        n_kv_heads = ctx.n_kv_heads
        theta = ctx.theta

        B, n_heads_q, N_CTX, HEAD_DIM = q.shape
        device = q.device
        d_half = HEAD_DIM // 2

        grad_output_fp32 = grad_output.to(torch.float32)
        attn_weights_fp32 = attn_weights.to(torch.float32)
        q_fp32 = q.to(torch.float32)
        k_fp32 = k.to(torch.float32)
        v_fp32 = v.to(torch.float32)

        if group_size == 1:
            k_expanded_fp32 = k_fp32
            v_expanded_fp32 = v_fp32
        else:
            k_expanded_fp32 = k_fp32.view(B, n_kv_heads, 1, N_CTX, HEAD_DIM).expand(
                B, n_kv_heads, group_size, N_CTX, HEAD_DIM
            ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
            v_expanded_fp32 = v_fp32.view(B, n_kv_heads, 1, N_CTX, HEAD_DIM).expand(
                B, n_kv_heads, group_size, N_CTX, HEAD_DIM
            ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)

        d_attn_weights = torch.einsum(
            "bhqd,bhkd->bhqk", grad_output_fp32, v_expanded_fp32
        )

        if needs_dv:
            dv_expanded_fp32 = torch.einsum(
                "bhqk,bhqd->bhkd", attn_weights_fp32, grad_output_fp32
            )

        d_softmax_sum = torch.sum(d_attn_weights * attn_weights_fp32, dim=-1, keepdim=True)
        dp = attn_weights_fp32 * (d_attn_weights - d_softmax_sum)

        causal_mask = None
        if causal:
            mask = torch.triu(
                torch.ones(N_CTX, N_CTX, device=device, dtype=torch.bool), diagonal=1
            )
            causal_mask = mask.unsqueeze(0).unsqueeze(0)
            dp = dp.masked_fill(causal_mask, 0.0)

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM)
        )

        # --- Leader-Driven Co-RoPE: 重新计算里程（与 forward 对齐）---
        q_leaders_fp32 = q_fp32[:, ::group_size, :, :]  # (B, n_kv_heads, N, D)
        raw_dot_leader = torch.einsum("bhqd,bhkd->bhqk", q_leaders_fp32, k_fp32)
        z_leader = torch.sigmoid(raw_dot_leader * sm_scale)
        a_leader = torch.cumsum(z_leader, dim=-1)
        a = a_leader.repeat_interleave(group_size, dim=1)  # 广播到全组
        diag_a = torch.diagonal(a, dim1=-2, dim2=-1)
        delta_a = diag_a.unsqueeze(-1) - a

        q1_fp32, q2_fp32 = q_fp32[..., :d_half], q_fp32[..., d_half:]
        k1_fp32, k2_fp32 = k_expanded_fp32[..., :d_half], k_expanded_fp32[..., d_half:]

        q1_e = q1_fp32.unsqueeze(3)
        q2_e = q2_fp32.unsqueeze(3)
        k1_e = k1_fp32.unsqueeze(2)
        k2_e = k2_fp32.unsqueeze(2)

        E_A = q1_e * k1_e + q2_e * k2_e
        E_B = q2_e * k1_e - q1_e * k2_e

        phi = delta_a.unsqueeze(-1) * inv_freq.view(1, 1, 1, 1, -1)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        g_T = dp * sm_scale
        g_T_expanded = g_T.unsqueeze(-1)

        g_E_A = g_T_expanded * cos_phi
        g_E_B = g_T_expanded * (-sin_phi)
        g_phi = g_T_expanded * (-E_A * sin_phi - E_B * cos_phi)

        dq1_from_EA = torch.einsum("bhqkd,bhkd->bhqd", g_E_A, k1_fp32)
        dq2_from_EA = torch.einsum("bhqkd,bhkd->bhqd", g_E_A, k2_fp32)
        dk1_from_EA = torch.einsum("bhqkd,bhqd->bhkd", g_E_A, q1_fp32)
        dk2_from_EA = torch.einsum("bhqkd,bhqd->bhkd", g_E_A, q2_fp32)

        dq2_from_EB = torch.einsum("bhqkd,bhkd->bhqd", g_E_B, k1_fp32)
        dk1_from_EB = torch.einsum("bhqkd,bhqd->bhkd", g_E_B, q2_fp32)
        dq1_from_EB = torch.einsum("bhqkd,bhkd->bhqd", -g_E_B, k2_fp32)
        dk2_from_EB = torch.einsum("bhqkd,bhqd->bhkd", -g_E_B, q1_fp32)

        g_delta = torch.sum(
            g_phi * inv_freq.view(1, 1, 1, 1, -1), dim=-1
        )

        g_diag_a = g_delta.sum(dim=-1)
        g_a = torch.diag_embed(g_diag_a) - g_delta

        # --- Leader-Driven Gradient: 聚合组内梯度到领航头 ---
        g_a_leader = g_a.view(B, n_kv_heads, group_size, N_CTX, N_CTX).sum(dim=2)
        g_z_leader = torch.cumsum(g_a_leader.flip(-1), dim=-1).flip(-1)
        g_pre_sigmoid_leader = g_z_leader * z_leader * (1.0 - z_leader)
        g_raw_dot_leader = g_pre_sigmoid_leader * sm_scale

        # 梯度传回领航头 Q 和原始 K
        dq_from_raw = torch.zeros_like(q_fp32)
        dq_from_raw[:, ::group_size, :, :] = torch.einsum("bhqk,bhkd->bhqd", g_raw_dot_leader, k_fp32)
        dk_from_raw = torch.einsum("bhqk,bhqd->bhkd", g_raw_dot_leader, q_leaders_fp32)

        dq_fp32 = dq_from_raw
        dq_fp32[..., :d_half] += dq1_from_EA + dq1_from_EB
        dq_fp32[..., d_half:] += dq2_from_EA + dq2_from_EB

        # dk_from_raw 现在是 (B, n_kv_heads, N, D)，需要先累加能量场梯度
        dk_expanded_fp32 = torch.zeros_like(k_expanded_fp32)
        dk_expanded_fp32[..., :d_half] = dk1_from_EA + dk1_from_EB
        dk_expanded_fp32[..., d_half:] = dk2_from_EA + dk2_from_EB

        if needs_dq:
            dq = dq_fp32.to(q.dtype)
        else:
            dq = None

        if needs_dk:
            # 聚合能量场梯度（从 k_expanded 回到原始 k）
            dk_from_energy = dk_expanded_fp32.view(B, n_kv_heads, group_size, N_CTX, HEAD_DIM).sum(dim=2)
            # 加上里程梯度
            dk = (dk_from_raw + dk_from_energy).contiguous().to(k.dtype)
        else:
            dk = None

        if needs_dv:
            if group_size == 1:
                dv = dv_expanded_fp32.to(v.dtype)
            else:
                dv = (
                    dv_expanded_fp32.view(B, n_kv_heads, group_size, N_CTX, HEAD_DIM)
                    .sum(dim=2)
                    .contiguous()
                    .to(v.dtype)
                )
        else:
            dv = None

        return dq, dk, dv, None, None, None


attention_pytorch = _attention_pytorch.apply

if __name__ == "__main__":
    import torch
    torch.manual_seed(42)
    
    B, H, N, D = 2, 4, 64, 32
    sm_scale = D ** -0.5
    theta = 10000.0
    causal = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 # 先用 fp32 排除精度干扰

    print(f"🔬 Starting Co-RoPE Bwd vs Autograd Verification")
    
    # 准备输入
    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H // 2, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H // 2, N, D, device=device, dtype=dtype, requires_grad=True)

    # 1. 前向验证（两者应该完全相同，因为 forward 逻辑相同）
    out_manual = attention_pytorch(q, k, v, causal, sm_scale, theta)
    # 直接复用 forward，绕过 custom backward（ctx=None 让 Autograd 接管）
    out_ref = _attention_pytorch.forward(None, q, k, v, causal, sm_scale, theta)
    
    fwd_diff = (out_manual - out_ref).abs().max().item()
    print(f"✅ Forward Max Diff: {fwd_diff:.2e}")

    # 2. 梯度验证
    # 计算参考梯度 (Autograd 真理)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    grad_q_ref, grad_k_ref, grad_v_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
    
    q.grad.zero_(); k.grad.zero_(); v.grad.zero_()

    # 计算手动梯度 (Manual Backward)
    try:
        loss_manual = out_manual.sum()
        loss_manual.backward()
        grad_q_manual, grad_k_manual, grad_v_manual = q.grad.clone(), k.grad.clone(), v.grad.clone()
        
        # 对拍
        diff_q = (grad_q_manual - grad_q_ref).abs().max().item()
        diff_k = (grad_k_manual - grad_k_ref).abs().max().item()
        diff_v = (grad_v_manual - grad_v_ref).abs().max().item()

        print(f"\n🔍 Gradient Comparison:")
        print(f"dQ Max Diff: {diff_q:.2e}")
        print(f"dK Max Diff: {diff_k:.2e}")
        print(f"dV Max Diff: {diff_v:.2e}")
        
        if diff_q < 1e-5 and diff_k < 1e-5 and diff_v < 1e-5:
            print("\n✨ [PASS] Manual backward matches Autograd!")
        else:
            print("\n⚠️ [WARN] Gradients differ, check backward logic.")
        
    except Exception as e:
        print(f"\n❌ Manual Backward Failed: {e}")
        import traceback
        traceback.print_exc()
