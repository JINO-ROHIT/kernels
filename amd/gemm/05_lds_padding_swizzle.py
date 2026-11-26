import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def v3_lds_swizzling(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
):

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    gLoadLayoutA: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [4, 1], [1, 0])
    gLoadLayoutB: gl.constexpr = gl.BlockedLayout([2, 1], [16, 4], [1, 4], [0, 1])
    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(2, 4, 8, order=[1, 0])
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(2, 4, 8, order=[0, 1])

    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], sharedLayoutB)

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=4)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=4)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    max_iter = gl.cdiv(K, BLOCK_K)
    gl.assume(max_iter > 0)

    for k in range(0, max_iter):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA, a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB, b_base, b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        a = smemA.load(dotOpLayoutA)
        b = smemB.load(dotOpLayoutB)

        acc = gl.amd.cdna3.mfma(a, b, acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    c = acc.to(a_ptr.dtype.element_ty)

    gStoreLayoutC: gl.constexpr = mfmaLayout
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c, mask=c_mask)


@gluon.jit
def v3_lds_padding(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
):

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # gfx942 copies one dword (two 16-bit elements) per lane into LDS.
    gLoadLayoutA: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [4, 1], [1, 0])
    gLoadLayoutB: gl.constexpr = gl.BlockedLayout([2, 1], [16, 4], [1, 4], [0, 1])
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [BLOCK_K, BLOCK_N], [0, 1]
    )

    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], sharedLayoutB)

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=4)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=4)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    for k in range(0, gl.cdiv(K, BLOCK_K)):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA, a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB, b_base, b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        a = smemA.load(dotOpLayoutA)
        b = smemB.load(dotOpLayoutB)

        acc = gl.amd.cdna3.mfma(a, b, acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    c = acc.to(a_ptr.dtype.element_ty)

    gStoreLayoutC: gl.constexpr = mfmaLayout
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c, mask=c_mask)


def matmul(a, b, c=None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    # The direct-to-LDS loads cover whole tiles without tail masks.
    if M % BLOCK_M or N % BLOCK_N or K % BLOCK_K or K == 0:
        raise ValueError("Expected whole 128 x 128 x 32 tiles with positive K")
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    ## Change the following to v3_lds_swizzling to run the kernel with swizzling layout
    v3_lds_padding[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return c