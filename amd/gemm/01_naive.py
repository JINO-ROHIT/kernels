import torch  
import triton 
from triton.experimental import gluon 
from triton.experimental.gluon import language as gl 


@gluon.jit  # Each program computes one 16-by-16 tile of C.
def v0_naive(
    a_ptr,  # Base address of A[M, K], shape [4096, 512].
    b_ptr,  # Base address of B[K, N], shape [512, 4096].
    c_ptr,  # Base address of C[M, N], shape [4096, 4096].
    M,  # Number of A and C rows: 4096.
    N,  # Number of B and C columns: 4096.
    K,  # Number of products summed for each output element: 512.
    stride_am,  # Step down one row of A, in elements: 512.
    stride_ak,  # Step along K in A: 1.
    stride_bk,  # Step along K in B: 1 because the benchmark passes a transposed view.
    stride_bn,  # Step to the next column of B: 512.
    stride_cm,  # Step down one row of C: 4096 for the default contiguous output.
    stride_cn,  # Step to the next column of C: 1.
    BLOCK_M: gl.constexpr,  # Output rows per tile: 16.
    BLOCK_N: gl.constexpr,  # Output columns per tile: 16.
    BLOCK_K: gl.constexpr,  # K positions processed per loop iteration: 32.
):
    # Follow program 258 as an example. IDs start at 0 and advance across columns first.
    pid = gl.program_id(axis=0)  # One scalar tile ID per program.
    num_pid_n = gl.cdiv(N, BLOCK_N)  # ceil(4096 / 16) = 256 tile columns.
    pid_m = pid // num_pid_n  # Tile row: 258 // 256 = 1.
    pid_n = pid % num_pid_n  # Tile column: 258 % 256 = 2.

    # The product tile has axes [m, n, k] and shape [16, 16, 32].
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1],  # One element per lane in the smallest repeating block.
        [1, 4, 16],  # One 64-lane wave spans 1 row, 4 columns, and 16 K positions.
        [2, 2, 1],  # Four waves per program, arranged 2 along M and 2 along N.
        [2, 1, 0],  # Assign consecutive lanes along K first, then N, then M.
    )
    a_layout: gl.constexpr = gl.SliceLayout(1, layout)  # Drop N: A tile uses [m, k].
    b_layout: gl.constexpr = gl.SliceLayout(0, layout)  # Drop M: B tile uses [n, k].
    c_layout: gl.constexpr = gl.SliceLayout(2, layout)  # Drop K: output uses [m, n].

    # Shape [16]. Program 258 owns global A and C rows 16 through 31.
    rows = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, a_layout))
    # Shape [16]. The same program owns global B and C columns 32 through 47.
    cols = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, b_layout))
    a_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_layout))  # [32], values 0..31.
    b_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, b_layout))  # Same offsets in B's layout.
    # [16, 1] rows + [1, 32] K offsets broadcast to [16, 32] addresses of A[row, k].
    a_ptrs = a_ptr + rows[:, None] * stride_am + a_k[None, :] * stride_ak
    # [16, 32] addresses indexed by [column, k]. Each points to B[k, column].
    b_ptrs = b_ptr + cols[:, None] * stride_bn + b_k[None, :] * stride_bk

    acc = gl.full((BLOCK_M, BLOCK_N), 0, gl.float32, layout=c_layout)  # [16, 16] FP32 zeros.
    for k in range(0, gl.cdiv(K, BLOCK_K)):  # 512 / 32 = 16 iterations, indexed 0..15.
        remaining = K - k * BLOCK_K  # 512, 480, ..., 32 valid K positions still to process.
        # Load A as [16, 32]. Use zero for positions outside M or K on a partial tile.
        a = gl.load(a_ptrs, (rows[:, None] < M) & (a_k[None, :] < remaining), other=0)
        # Load B as [16, 32], with K last locally. Mask positions outside N or K.
        b = gl.load(b_ptrs, (cols[:, None] < N) & (b_k[None, :] < remaining), other=0)
        # [16, 1, 32] * [1, 16, 32] -> [16, 16, 32] FP32 products.
        # On iteration 0, products[0, 0, 3] = A[16, 3] * B[3, 32] for program 258.
        products = a.to(gl.float32)[:, None, :] * b.to(gl.float32)[None, :, :]
        acc += gl.sum(products, axis=2)  # Sum 32 products per output and add the [16, 16] result.
        a_ptrs += BLOCK_K * stride_ak  # Advance every A pointer by 32 K positions.
        b_ptrs += BLOCK_K * stride_bk  # Advance every B pointer by the same 32 K positions.

    # Keep the index values and distribute them across lanes according to the output layout.
    c_rows = gl.convert_layout(rows, gl.SliceLayout(1, c_layout))  # [16], still 16..31.
    c_cols = gl.convert_layout(cols, gl.SliceLayout(0, c_layout))  # [16], still 32..47.
    c_ptrs = c_ptr + c_rows[:, None] * stride_cm + c_cols[None, :] * stride_cn  # [16, 16] addresses.
    # Write C[16:32, 32:48] in the example. The store casts FP32 to C's dtype and masks edge tiles.
    gl.store(c_ptrs, acc, (c_rows[:, None] < M) & (c_cols[None, :] < N))


def matmul(a, b, c=None):  # Launch the kernel from Python; the caller can supply the output.
    M, K = a.shape  # A is [4096, 512] in the example.
    K, N = b.shape  # B is [512, 4096] and shares A's K dimension.
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 32  # One tile computes 16 x 16 outputs, 32 terms at a time.
    num_warps = 4  # Four 64-lane waves per program on the MI300X.
    if c is None:  # Allocate C when the caller omits it.
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)  # [4096, 4096], same dtype as A.
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)  # 256 x 256 = 65,536 programs.
    grid = (GRID_MN, 1)  # Grid axis 0 enumerates all output tiles.
    v0_naive[grid](  # Launch one program per tile with the following arguments.
        a,  # Storage for input A.
        b,  # Storage for input B.
        c,  # Storage for output C.
        M,  # 4096 rows.
        N,  # 4096 columns.
        K,  # 512 reduction terms.
        a.stride(0),  # A row stride: 512.
        a.stride(1),  # A K stride: 1.
        b.stride(0),  # B K stride: 1.
        b.stride(1),  # B column stride: 512.
        c.stride(0),  # C row stride: 4096.
        c.stride(1),  # C column stride: 1.
        BLOCK_M=BLOCK_M,  # Compile for 16 rows per tile.
        BLOCK_N=BLOCK_N,  # Compile for 16 columns per tile.
        BLOCK_K=BLOCK_K,  # Compile for 32 K positions per iteration.
        num_warps=num_warps,  # Launch with the four waves used by the layout.
    )
    return c  # Return the [M, N] result.
