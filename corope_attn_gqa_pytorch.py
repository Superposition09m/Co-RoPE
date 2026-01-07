import torch
import torch.nn.functional as F

def attention_pytorch_block_shared(q, k, v, causal, sm_scale, theta, block_m=64):
    """
    PyTorch implementation of Co-RoPE attention with Block-Shared Leader strategy.
    
    This function uses PyTorch Autograd for backward pass, which is the 
    Ground Truth for verifying the Triton kernel.
    
    Args:
        q: (BATCH, H, N_CTX, HEAD_DIM)
        k: (BATCH, H, N_CTX, HEAD_DIM)
        v: (BATCH, H, N_CTX, HEAD_DIM)
        causal: bool
        sm_scale: float
        theta: float
        block_m: int, MUST match the BLOCK_M used in Triton kernel
    """
    if not causal:
        raise ValueError("Current verification only supports causal=True")

    B, n_heads_q, N_CTX, HEAD_DIM = q.shape
    n_heads_kv = k.shape[1]
    device = q.device
    
    # 确保 N_CTX 能被 block_m 整除，方便 reshape/repeat
    # 如果 Triton Kernel 支持非整除，这里需要做 padding 处理
    if N_CTX % block_m != 0:
        print(f"Warning: N_CTX({N_CTX}) is not divisible by block_m({block_m}). "
              "Padding for verification logic.")
        # 简单处理：截断或报错，这里为了演示逻辑假设整除
        pass

    # Compute RoPE frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))

    # Handle GQA (Group Query Attention)
    group_size = n_heads_q // n_heads_kv
    
    # Expand K, V for broadcasting interactions
    # shape: (B, H_Q, N, D)
    if group_size > 1:
        # 这里的 expand 逻辑是为了让 Q 和 K 能点积
        k_expanded = k.view(B, n_heads_kv, 1, N_CTX, HEAD_DIM).expand(
            B, n_heads_kv, group_size, N_CTX, HEAD_DIM
        ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
        v_expanded = v.view(B, n_heads_kv, 1, N_CTX, HEAD_DIM).expand(
            B, n_heads_kv, group_size, N_CTX, HEAD_DIM
        ).reshape(B, n_heads_q, N_CTX, HEAD_DIM)
    else:
        k_expanded = k
        v_expanded = v

    # ============================================================
    # Co-RoPE Block-Shared Logic
    # ============================================================
    
    # 1. 提取 Leaders (Block Shared & GQA Shared)
    # 物理含义：每个 Block (block_m) 只有 1 个 Leader，每个 GQA Group 只有 1 个 Leader
    # Shape: [B, H_KV, N_BLOCKS, D]
    # 注意：这里取的是原始 k (H_KV)，所以 Q 也只取 H_KV 个
    q_leaders = q[:, ::group_size, 0::block_m, :]
    
    N_BLOCKS = q_leaders.shape[2]
    
    # 2. 计算 Block Leader 的里程 (Discovery Pass)
    # q_leaders: [B, H_KV, N_BLOCKS, D]
    # k:         [B, H_KV, N_CTX,    D]
    # raw_dot:   [B, H_KV, N_BLOCKS, N_CTX]
    raw_dot = torch.einsum('bhqd,bhkd->bhqk', q_leaders, k)
    
    # Sigmoid
    z_dist = torch.sigmoid(raw_dot * sm_scale)
    
    # Masking: Leader at block i can only accumulate mileage from k < block_start_i
    # (assuming the leader is the first token of the block)
    leader_indices = torch.arange(0, N_CTX, block_m, device=device) # [N_BLOCKS]
    k_indices = torch.arange(N_CTX, device=device)                  # [N_CTX]
    
    # mask: [1, 1, N_BLOCKS, N_CTX]
    # Leader i 只能看到 K index <= Leader index 的部分 (Causal)
    mileage_mask = leader_indices[:, None] >= k_indices[None, :]
    z_dist = torch.where(mileage_mask[None, None, :, :], z_dist, 0.0)

    # 3. 计算 Running Mileage 和 Total Mileage
    # a_running[..., n] 表示 Leader 遇到 Key n 时的累计里程
    a_running = torch.cumsum(z_dist, dim=-1) # [B, H_KV, N_BLOCKS, N_CTX]
    
    # a_total[..., b] 表示 Leader b 在 mask 范围内的总里程
    # 由于我们 mask 了后续的 K，取最后一个值即为总里程
    a_total = a_running[..., -1].unsqueeze(-1) # [B, H_KV, N_BLOCKS, 1]
    
    # 4. 广播 (Broadcast) 回全分辨率
    # 将 N_BLOCKS 维度拉伸回 N_CTX
    # [B, H_KV, N_CTX, N_CTX]
    a_running_expanded = a_running.repeat_interleave(block_m, dim=2)
    # [B, H_KV, N_CTX, 1]
    a_total_expanded = a_total.repeat_interleave(block_m, dim=2)
    
    # 处理截断问题：如果 N_CTX 不是 block_m 的整数倍，repeat 后会多出来，需要切片
    if a_running_expanded.shape[2] > N_CTX:
        a_running_expanded = a_running_expanded[:, :, :N_CTX, :]
        a_total_expanded = a_total_expanded[:, :, :N_CTX, :]
        
    # GQA Broadcast: 复制给组内的所有 Heads
    # [B, H_Q, N_CTX, N_CTX]
    a_running_final = a_running_expanded.repeat_interleave(group_size, dim=1)
    # [B, H_Q, N_CTX, 1]
    a_total_final = a_total_expanded.repeat_interleave(group_size, dim=1)
    
    # 5. 计算 Delta A (Phase)
    # Co-RoPE Phase: (a_query_total - a_key_current)
    # 注意维度广播: [B, H, N, 1] - [B, H, N, N] -> [B, H, N, N]
    delta_a = a_total_final - a_running_final
    
    # ============================================================
    # Standard Attention with Rotated Score
    # ============================================================

    # Split Q, K into halves for RoPE pairs
    d_half = HEAD_DIM // 2
    q1, q2 = q[..., :d_half], q[..., d_half:]
    k1, k2 = k_expanded[..., :d_half], k_expanded[..., d_half:]
    
    # Pre-rotation dot products (EA, EB)
    # Dimensions: [B, H, N, N, D/2]
    # 注意这里显存占用巨大，N_CTX 较大时可能会 OOM，仅供对拍小序列使用
    E_A = q1.unsqueeze(3) * k1.unsqueeze(2) + q2.unsqueeze(3) * k2.unsqueeze(2)
    E_B = q2.unsqueeze(3) * k1.unsqueeze(2) - q1.unsqueeze(3) * k2.unsqueeze(2)
    
    # Calculate Phase
    # [B, H, N, N, D/2]
    phi = delta_a.unsqueeze(-1) * inv_freq.view(1, 1, 1, 1, -1)
    
    # Apply Rotation
    # score = E_A * cos(phi) - E_B * sin(phi)
    # Sum over head_dim halves
    attn_scores = (E_A * torch.cos(phi) - E_B * torch.sin(phi)).sum(dim=-1) * sm_scale
    
    # Standard Causal Masking (for Attention, distinct from mileage mask)
    mask = torch.triu(torch.ones(N_CTX, N_CTX, device=device, dtype=torch.bool), diagonal=1)
    attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float('-inf'))
    
    # Softmax
    p = torch.softmax(attn_scores.float(), dim=-1).to(q.dtype)
    
    # Output projection
    output = torch.matmul(p, v_expanded)
    
    return output


# ==========================================
# 验证脚本
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 设置参数，确保 N_CTX 是 BLOCK_M 的倍数
    BLOCK_M_VAL = 64
    B, H, N, D = 2, 4, 128, 64 
    sm_scale = D ** -0.5
    theta = 10000.0
    causal = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 # 使用 fp16 测试

    print(f"🔬 Starting Block-Shared Co-RoPE Verification")
    print(f"   Config: B={B}, H={H}, N={N}, D={D}, BLOCK_M={BLOCK_M_VAL}")

    # 准备输入
    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

    # -------------------------------------------------------
    # 1. 运行 PyTorch Reference (Autograd Ground Truth)
    # -------------------------------------------------------
    ref_out = attention_pytorch_block_shared(
        q, k, v, 
        causal=causal, 
        sm_scale=sm_scale, 
        theta=theta, 
        block_m=BLOCK_M_VAL
    )

    # 计算 Reference Backward
    loss_ref = ref_out.sum()
    loss_ref.backward()
    grad_q_ref = q.grad.clone()
    grad_k_ref = k.grad.clone()
    grad_v_ref = v.grad.clone()

    print("✅ PyTorch Reference computed.")

    # -------------------------------------------------------
    # 2. 运行 Triton Kernel (此处调用你的 Triton 函数)
    # -------------------------------------------------------
    # 清空梯度
    q.grad = None
    k.grad = None
    v.grad = None
    
    print("🚀 Running Triton Kernel...")
    try:
        # 假设你的 Triton 封装函数叫做 attention
        # 务必确保 Triton 内部使用的 BLOCK_M 与 BLOCK_M_VAL 一致 (64)
        # 并且 Autotune 已关闭或被限制为仅使用 BLOCK_M=64
        tri_out = attention(q, k, v, causal, sm_scale, warp_specialize=False)
        
        # Triton Backward
        loss_tri = tri_out.sum()
        loss_tri.backward()
        
        # -------------------------------------------------------
        # 3. 结果对比
        # -------------------------------------------------------
        print("\n🔍 Comparison Results:")
        
        # Forward 对比
        fwd_diff = (ref_out - tri_out).abs().max().item()
        print(f"Forward Max Diff: {fwd_diff:.4e}")
        
        # Backward 对比
        # 注意：由于 float16 的累加误差，Triton 和 PyTorch 的差距可能在 1e-3 左右
        dq_diff = (grad_q_ref - q.grad).abs().max().item()
        dk_diff = (grad_k_ref - k.grad).abs().max().item()
        dv_diff = (grad_v_ref - v.grad).abs().max().item()
        
        print(f"dQ Max Diff:      {dq_diff:.4e}")
        print(f"dK Max Diff:      {dk_diff:.4e}")
        print(f"dV Max Diff:      {dv_diff:.4e}")
        
        # 简单判断
        tol = 1e-2 if dtype == torch.float16 else 1e-4
        if fwd_diff < tol and dq_diff < tol:
            print("\n✨ Match! The Triton kernel correctly implements Block-Shared Logic.")
        else:
            print("\n⚠️ Mismatch! Check leader selection logic or mask boundaries.")

    except NameError:
        print("\n⚠️ Triton function 'attention' not found. skipping triton run.")
    except Exception as e:
        print(f"\n❌ Triton Run Failed: {e}")
        import traceback
        traceback.print_exc()