"""
测试 Triton 是否支持在 kernel 内部对张量进行切片操作
特别是验证类似 q[:, :d_half] 这样的操作是否可行
"""

import torch
import triton
import triton.language as tl


@triton.jit
def test_slice_kernel(
    Q,
    Out1,
    Out2,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_o1b, stride_o1h, stride_o1m, stride_o1d,
    stride_o2b, stride_o2h, stride_o2m, stride_o2d,
    BATCH: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    D_HALF: tl.constexpr,  # 必须作为 constexpr 参数传入
):
    """
    尝试在 kernel 内部对 Q 进行切片操作
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # 计算偏移量
    start_m = pid_m * BLOCK_M
    
    # 加载 Q 块: [BLOCK_M, HEAD_DIM]
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    mask_m = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=mask_m, other=0.0)
    
    # 方法1: 直接切片 (这是用户代码中的方式)
    q1 = q[:, :D_HALF]
    q2 = q[:, D_HALF:]
    
    # 存储结果
    out1_ptrs = Out1 + pid_b * stride_o1b + pid_h * stride_o1h + offs_m[:, None] * stride_o1m + tl.arange(0, D_HALF)[None, :] * stride_o1d
    out2_ptrs = Out2 + pid_b * stride_o2b + pid_h * stride_o2h + offs_m[:, None] * stride_o2m + tl.arange(0, D_HALF)[None, :] * stride_o2d
    
    tl.store(out1_ptrs, q1, mask=mask_m)
    tl.store(out2_ptrs, q2, mask=mask_m)


@triton.jit
def test_slice_kernel_v2(
    Q,
    Out1,
    Out2,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_o1b, stride_o1h, stride_o1m, stride_o1d,
    stride_o2b, stride_o2h, stride_o2m, stride_o2d,
    BATCH: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    D_HALF: tl.constexpr,  # 必须作为 constexpr 参数传入
):
    """
    使用索引方式来实现切片（更安全的方式）
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    
    # 分别加载前半部分和后半部分
    offs_d1 = tl.arange(0, D_HALF)
    offs_d2 = D_HALF + tl.arange(0, D_HALF)
    
    q1_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d1[None, :] * stride_qd
    q2_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d2[None, :] * stride_qd
    
    mask_m = offs_m[:, None] < N_CTX
    q1 = tl.load(q1_ptrs, mask=mask_m, other=0.0)
    q2 = tl.load(q2_ptrs, mask=mask_m, other=0.0)
    
    # 存储结果
    out1_ptrs = Out1 + pid_b * stride_o1b + pid_h * stride_o1h + offs_m[:, None] * stride_o1m + tl.arange(0, D_HALF)[None, :] * stride_o1d
    out2_ptrs = Out2 + pid_b * stride_o2b + pid_h * stride_o2h + offs_m[:, None] * stride_o2m + tl.arange(0, D_HALF)[None, :] * stride_o2d
    
    tl.store(out1_ptrs, q1, mask=mask_m)
    tl.store(out2_ptrs, q2, mask=mask_m)


def test_triton_slice_support():
    """测试 Triton 是否支持张量切片"""
    BATCH = 2
    N_HEADS = 4
    N_CTX = 128
    HEAD_DIM = 64
    BLOCK_M = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("需要 CUDA 设备来运行 Triton 测试")
        return
    
    # 创建测试数据
    Q = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=torch.float32, device=device)
    Out1 = torch.zeros((BATCH, N_HEADS, N_CTX, HEAD_DIM // 2), dtype=torch.float32, device=device)
    Out2 = torch.zeros((BATCH, N_HEADS, N_CTX, HEAD_DIM // 2), dtype=torch.float32, device=device)
    
    # 参考实现（PyTorch）
    Q_ref1 = Q[:, :, :, :HEAD_DIM // 2]
    Q_ref2 = Q[:, :, :, HEAD_DIM // 2:]
    
    print("=" * 80)
    print("测试 1: 尝试在 Triton kernel 内部直接使用 q[:, :d_half] 切片")
    print("=" * 80)
    
    grid = (BATCH, N_HEADS, triton.cdiv(N_CTX, BLOCK_M))
    D_HALF = HEAD_DIM // 2
    
    try:
        test_slice_kernel[grid](
            Q, Out1, Out2,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            Out1.stride(0), Out1.stride(1), Out1.stride(2), Out1.stride(3),
            Out2.stride(0), Out2.stride(1), Out2.stride(2), Out2.stride(3),
            BATCH=BATCH,
            N_HEADS=N_HEADS,
            N_CTX=N_CTX,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M,
            D_HALF=D_HALF,
        )
        
        # 检查结果
        error1 = torch.abs(Out1 - Q_ref1).max().item()
        error2 = torch.abs(Out2 - Q_ref2).max().item()
        
        print(f"✓ 测试通过！直接切片方式可以工作")
        print(f"  前半部分最大误差: {error1:.6e}")
        print(f"  后半部分最大误差: {error2:.6e}")
        
    except Exception as e:
        print(f"✗ 测试失败！直接切片方式不支持")
        print(f"  错误信息: {str(e)}")
        print(f"  错误类型: {type(e).__name__}")
    
    print()
    print("=" * 80)
    print("测试 2: 使用索引方式来实现切片（推荐方式）")
    print("=" * 80)
    
    Out1.zero_()
    Out2.zero_()
    
    try:
        test_slice_kernel_v2[grid](
            Q, Out1, Out2,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            Out1.stride(0), Out1.stride(1), Out1.stride(2), Out1.stride(3),
            Out2.stride(0), Out2.stride(1), Out2.stride(2), Out2.stride(3),
            BATCH=BATCH,
            N_HEADS=N_HEADS,
            N_CTX=N_CTX,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M,
            D_HALF=D_HALF,
        )
        
        # 检查结果
        error1 = torch.abs(Out1 - Q_ref1).max().item()
        error2 = torch.abs(Out2 - Q_ref2).max().item()
        
        print(f"✓ 测试通过！索引方式可以工作")
        print(f"  前半部分最大误差: {error1:.6e}")
        print(f"  后半部分最大误差: {error2:.6e}")
        
    except Exception as e:
        print(f"✗ 测试失败！索引方式也不支持")
        print(f"  错误信息: {str(e)}")
        print(f"  错误类型: {type(e).__name__}")
    
    print()
    print("=" * 80)
    print("结论:")
    print("=" * 80)
    print("Triton 在 kernel 内部对已经加载到寄存器的张量进行切片（如 q[:, :d_half]）")
    print("这种操作通常是不支持的。正确的做法是：")
    print("1. 在加载数据时就分别加载不同的部分")
    print("2. 使用不同的偏移量来计算指针地址")
    print("3. 不要依赖 Python 风格的切片语法")


if __name__ == "__main__":
    test_triton_slice_support()

