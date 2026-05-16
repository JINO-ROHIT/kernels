import time
import torch
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

VERBOSE = True
LOG = "[Info]"

gemm_dtype = cutlass.Float32

num_stages = 3
threads = 256 # one CTA has 256 threads ie 256/32 = 8 warps
copy_bits = 128
mma_permute = 4 # groups data in chunks of 4 fp32 values
cta_tiler = (128, 128, 8) # one block computes 128x128 of C, consuming K in chunks of 8.
warp_tiler = (32, 64, 8) # one warp computes 32x64 inside that CTA tile.
"""
CTA tile says the blocks total output region.
Warp tile says how that work is divided among warps.
128x128 CTA split by 32x64 warp tile gives 4 x 2 = 8 warp tiles, which exactly matches 8 warps.
"""
wmma_tiler = (4, 8)

tile_m, tile_n, tile_k = cta_tiler
warp_m, warp_n, warp_k = warp_tiler
wmma_m, wmma_n = wmma_tiler
num_warps_m = tile_m // warp_m
num_warps_n = tile_n // warp_n
num_warps = num_warps_m * num_warps_n
vl = copy_bits // gemm_dtype.width
bytes_alignment = copy_bits // 8

assert mma_permute == vl
assert wmma_m * wmma_n == 32
assert wmma_m * wmma_n * num_warps == threads
assert tile_m % (warp_m) == 0
assert tile_n % (warp_n) == 0
assert warp_m % (mma_permute * wmma_m) == 0
assert warp_n % (mma_permute * wmma_n) == 0
assert tile_m % bytes_alignment == 0
assert tile_n % bytes_alignment == 0
assert bytes_alignment % (copy_bits // 8) == 0


def make_tiled_copy_AB(cta_tiler: Tuple[int, int]):
    order = (1, 0)
    thr_tiler_0 = threads // cta_tiler[1]
    thr_tiler_1 = cta_tiler[1]
    val_tiler = (mma_permute, 1)
    num_bits_per_copy = gemm_dtype.width

    thr_layout = cute.make_ordered_layout((thr_tiler_0, thr_tiler_1), order=order)
    val_layout = cute.make_ordered_layout(val_tiler, order=order)
    copy_atom_AB = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        gemm_dtype,
        num_bits_per_copy=num_bits_per_copy,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom_AB, thr_layout, val_layout)
    return tiled_copy


def make_tiled_copy_C():
    order = (1, 0)
    thr_tiler_1 = tile_n // vl
    thr_tiler_0 = threads * vl // tile_n
    val_tiler = (1, vl)

    thr_layout = cute.make_ordered_layout((thr_tiler_0, thr_tiler_1), order=order)
    val_layout = cute.make_ordered_layout(val_tiler, order=order)
    copy_atom_C = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gemm_dtype,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom_C, thr_layout, val_layout)
    return tiled_copy


@cute.kernel
def sgemm_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    smem_layout_A: cute.Layout,
    smem_layout_B: cute.Layout,
    smem_layout_C: cute.Layout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_copy_C: cute.TiledCopy,
    tiled_warp_mma: cute.TiledMma,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    thr_idx = cute.arch.thread_idx()[0]
    blk_idx, blk_idy = cute.arch.block_idx()[:2]
    wrp_idx = cute.arch.warp_idx()
    lne_idx = cute.arch.lane_idx() # between 0 to 31


    cta_coord = (blk_idx, blk_idy, None)

    gA = cute.local_tile(A, cta_tiler, cta_coord, proj=(1, None, 1))
    gB = cute.local_tile(B, cta_tiler, cta_coord, proj=(None, 1, 1))
    gC = cute.local_tile(C, cta_tiler, cta_coord, proj=(1, 1, None))

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(gemm_dtype, smem_layout_A, 16)
    sB = smem.allocate_tensor(gemm_dtype, smem_layout_B, 16)
    sC = smem.allocate_tensor(gemm_dtype, smem_layout_C, 16)


    thr_copy_A = tiled_copy_A.get_slice(thr_idx)
    thr_copy_B = tiled_copy_B.get_slice(thr_idx)
    thr_copy_C = tiled_copy_C.get_slice(thr_idx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)
    tCsC_copy = thr_copy_C.partition_S(sC)
    tCgC_copy = thr_copy_C.partition_D(gC)


    # =============================== Prefetch Prologue ===============================
    # ---------------------------------------------------------------------------------
    tAsA.fill(0)
    tBsB.fill(0)
    cute.arch.sync_threads()

    # Then start async loads and fill the 0 to num_stages-1 pipes
    num_k_tiles = cute.size(tAgA, mode=[3])
    k_tile_index_gmem = cutlass.Int32(0)
    k_tile_index_smem = cutlass.Int32(0)
    for _ in range(0, num_stages - 1):
        if k_tile_index_smem < num_k_tiles:
            coord_gmem = (None, None, None, k_tile_index_gmem)
            coord_smem = (None, None, None, k_tile_index_smem)
            cute.copy(tiled_copy_A, tAgA[coord_gmem], tAsA[coord_smem])
            cute.copy(tiled_copy_B, tBgB[coord_gmem], tBsB[coord_smem])
            k_tile_index_gmem += 1
            k_tile_index_smem += 1
        cute.arch.cp_async_commit_group()


    wrp_idm, wrp_idn = wrp_idx // num_warps_n, wrp_idx % num_warps_n
    warp_coord_A = ((None, None), wrp_idm, None, None)
    warp_coord_B = ((None, None), wrp_idn, None, None)
    warp_coord_C = ((None, None), wrp_idm, wrp_idn)
    wCsA = cute.tiled_divide(sA, (warp_m, warp_k))[warp_coord_A]
    wCsB = cute.tiled_divide(sB, (warp_n, warp_k))[warp_coord_B]
    wCsC_mma = cute.tiled_divide(sC, (warp_m, warp_n))[warp_coord_C]
    wCgC_mma = cute.tiled_divide(gC, (warp_m, warp_n))[warp_coord_C]

    thr_mma = tiled_warp_mma.get_slice(lne_idx)
    tCsA = thr_mma.partition_A(wCsA)
    tCsB = thr_mma.partition_B(wCsB)
    tCsC_mma = thr_mma.partition_C(wCsC_mma)
    tCgC_mma = thr_mma.partition_C(wCgC_mma)

    tCrA = tiled_warp_mma.make_fragment_A(tCsA[None, None, None, None, 0])
    tCrB = tiled_warp_mma.make_fragment_B(tCsB[None, None, None, None, 0])
    tCrC = tiled_warp_mma.make_fragment_C(tCgC_mma)
    tCrC.fill(0.0)

    # Prefetch SMEM->RMEM for the first MMA tile
    smem_pipe_read = cutlass.Int32(0)
    smem_pipe_write = cutlass.Int32(k_tile_index_smem)
    gmem_pipe_read = k_tile_index_gmem

    tCsA_p = tCsA[None, None, None, None, smem_pipe_read]
    tCsB_p = tCsB[None, None, None, None, smem_pipe_read]

    num_k_frags = cute.size(tCrA, mode=[2])
    cta_sync_barrier = cutlass.pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads,
    )
    if num_k_frags > 1:
        # Wait until the first prefetched tile is loaded in
        cute.arch.cp_async_wait_group(num_stages - 2)
        cta_sync_barrier.arrive_and_wait()
        # Prefetch the first rmem from the first k-tile
        coord_frag = (None, None, 0, None)
        cute.autovec_copy(tCsA_p[coord_frag], tCrA[coord_frag])
        cute.autovec_copy(tCsB_p[coord_frag], tCrB[coord_frag])

    if cutlass.const_expr(VERBOSE):
        print(f"{LOG} gA {gA}")
        print(f"{LOG} gB {gB}")
        print(f"{LOG} gC {gC}")
        print(f"{LOG} sA {sA}")
        print(f"{LOG} sB {sB}")
        print(f"{LOG} tAgA {tAgA}")
        print(f"{LOG} tAsA {tAsA}")
        print(f"{LOG} tBgB {tBgB}")
        print(f"{LOG} tBsB {tBsB}")
        print(f"{LOG} tCsC_copy {tCsC_copy}")
        print(f"{LOG} tCgC_copy {tCgC_copy}")
        print(f"{LOG} wCsA {wCsA}")
        print(f"{LOG} wCsB {wCsB}")
        print(f"{LOG} wCsC_mma {wCsC_mma}")
        print(f"{LOG} wCgC_mma {wCgC_mma}")
        print(f"{LOG} tCsA {tCsA}")
        print(f"{LOG} tCsB {tCsB}")
        print(f"{LOG} tCsC_mma {tCsC_mma}")
        print(f"{LOG} tCgC_mma {tCgC_mma}")
        print(f"{LOG} tCrA {tCrA}")
        print(f"{LOG} tCrB {tCrB}")
        print(f"{LOG} tCrC {tCrC}")


    for _ in range(num_k_tiles):

        if gmem_pipe_read < num_k_tiles:
            coord_g = (None, None, None, gmem_pipe_read)
            coord_s = (None, None, None, smem_pipe_write)
            cute.copy(tiled_copy_A, tAgA[coord_g], tAsA[coord_s])
            cute.copy(tiled_copy_B, tBgB[coord_g], tBsB[coord_s])
            # Update meta pointers of gmem/smem pipes
            gmem_pipe_read += 1
            smem_pipe_write = (smem_pipe_write + 1) % num_stages
        # Always commit new cp.async and always move smem_pipe_read ptr
        cute.arch.cp_async_commit_group()
        smem_pipe_read = (smem_pipe_read + 1) % num_stages
        for k_frag_index in cutlass.range(num_k_frags, unroll_full=True):

            if k_frag_index == num_k_frags - 1:
                coord_smem_next = (None, None, None, None, smem_pipe_read)
                tCsA_p = tCsA[coord_smem_next]
                tCsB_p = tCsB[coord_smem_next]
                cute.arch.cp_async_wait_group(num_stages - 2)
                cta_sync_barrier.arrive_and_wait()
            # - Then fetch next frag SMEM->RMEM
            coord_frag_next = (None, None, (k_frag_index + 1) % num_k_frags, None)
            cute.autovec_copy(tCsA_p[coord_frag_next], tCrA[coord_frag_next])
            cute.autovec_copy(tCsB_p[coord_frag_next], tCrB[coord_frag_next])
            # - Finally perform mma on current frag
            coord_mma = (None, None, k_frag_index, None)
            cute.gemm(tiled_warp_mma, tCrC, tCrA[coord_mma], tCrB[coord_mma], tCrC)

    cute.arch.cp_async_wait_group(0)
    cta_sync_barrier.arrive_and_wait()


    tCrC.store(epilogue_op(tCrC.load()))
    cute.autovec_copy(tCrC, tCsC_mma)
    cute.arch.sync_threads()
    tCrC_copy = cute.make_rmem_tensor_like(tCsC_copy)
    cute.autovec_copy(tCsC_copy, tCrC_copy)
    cute.copy(tiled_copy_C, tCrC_copy, tCgC_copy)


@cute.jit
def sgemm_host(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):

    major_mode_A = LayoutEnum.from_tensor(A)
    major_mode_B = LayoutEnum.from_tensor(B)

    padding_A = mma_permute if major_mode_A == LayoutEnum.ROW_MAJOR else 0
    padding_B = mma_permute if major_mode_B == LayoutEnum.ROW_MAJOR else 0

    smem_layout_A = cute.make_layout(
        (tile_m, tile_k, num_stages), stride=(1, tile_m + padding_A, tile_k * (tile_m + padding_A))
    )
    smem_layout_B = cute.make_layout(
        (tile_n, tile_k, num_stages), stride=(1, tile_n + padding_B, tile_k * (tile_n + padding_B))
    )
    smem_layout_C = cute.make_layout((tile_m, tile_n), stride=(tile_n, 1))
    smem_size = sum(
        [cute.size_in_bytes(gemm_dtype, lo) for lo in [smem_layout_A, smem_layout_B, smem_layout_C]]
    )

    tiled_copy_A = make_tiled_copy_AB((tile_m, tile_k))
    tiled_copy_B = make_tiled_copy_AB((tile_n, tile_k))
    tiled_copy_C = make_tiled_copy_C()


    warp_mma_atom_layout = cute.make_layout((wmma_m, wmma_n, 1), stride=(wmma_n, 1, 0))
    permutation_m = cute.make_ordered_layout((wmma_m, mma_permute), order=(1, 0))
    permutation_n = cute.make_ordered_layout((wmma_n, mma_permute), order=(1, 0))
    tiled_warp_mma = cute.make_tiled_mma(
        cute.nvgpu.MmaUniversalOp(gemm_dtype),
        atom_layout_mnk=warp_mma_atom_layout,
        permutation_mnk=(permutation_m, permutation_n, None),
    )

    grid_dim = [*cute.ceil_div(C.shape, (tile_m, tile_n)), 1]
    block_dim = [threads, 1, 1]

    if cutlass.const_expr(VERBOSE):
        print(f"{LOG} Tensor A {A}")
        print(f"{LOG} Tensor B {B}")
        print(f"{LOG} Tensor C {C}")
        print(f"{LOG} Major mode A {major_mode_A}")
        print(f"{LOG} Major mode B {major_mode_B}")
        print(f"{LOG} Smem layout A {smem_layout_A}")
        print(f"{LOG} Smem layout B {smem_layout_B}")
        print(f"{LOG} Copy layout A {tiled_copy_A}")
        print(f"{LOG} Copy layout B {tiled_copy_B}")
        print(f"{LOG} Copy layout C {tiled_copy_C}")
        print(f"{LOG} Mma layout (Warp level) {tiled_warp_mma}")
        print(f"{LOG} Gemm tile size {cta_tiler}")
        print(f"{LOG} Sgemm grid {grid_dim}")
        print(f"{LOG} Sgemm block {block_dim}")

    sgemm_kernel(
        A,
        B,
        C,
        smem_layout_A,
        smem_layout_B,
        smem_layout_C,
        tiled_copy_A,
        tiled_copy_B,
        tiled_copy_C,
        tiled_warp_mma,
        epilogue_op,
    ).launch(grid=grid_dim, block=block_dim, smem=smem_size)


import time

def run_sgemm(
    M: int = 2048,
    K: int = 2048,
    N: int = 2048,
    verify: bool = True,
    warmup_iterations: int = 10,
    iterations: int = 100,
):
    print(f"running sgemm with M={M}, N={N}, K={K}")

    def tensor_generator(return_torch_tensor: bool = False):
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

        b_kernel = b.transpose(0, 1).contiguous()

        a_tensor = from_dlpack(a, assumed_align=16)
        b_tensor = from_dlpack(b_kernel, assumed_align=16)
        c_tensor = from_dlpack(c, assumed_align=16)

        if return_torch_tensor:
            return a, b, c, a_tensor, b_tensor, c_tensor
        return a_tensor, b_tensor, c_tensor

    workspace_generator = lambda: testing.JitArguments(*tensor_generator())

    _a, _b, _c, _a_tensor, _b_tensor, _c_tensor = tensor_generator(return_torch_tensor=True)

    compile_tic = time.perf_counter()
    matmul = cute.compile(sgemm_host, _a_tensor, _b_tensor, _c_tensor)
    #matmul = cute.compile(sgemm_host, _a_tensor, _b_tensor, _c_tensor, options="--generate-line-info")
    print(f"kernel compiled in {time.perf_counter() - compile_tic:.4f} seconds")

    if verify:
        matmul(_a_tensor, _b_tensor, _c_tensor)
        torch.cuda.synchronize()
        torch.testing.assert_close(_c, torch.matmul(_a, _b), atol=1e-3, rtol=1e-3)
        print("verification passed!")
    else:
        print("verification skipped...")
    
    torch.cuda.empty_cache()
    for _ in range(warmup_iterations):
        _ = torch.matmul(_a, _b)
    torch.cuda.synchronize()
    torch_tic = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(_a, _b)
    torch.cuda.synchronize()
    torch_avg_time_us = (time.perf_counter() - torch_tic) * 1e6 / iterations
    print(f"torch kernel execution time: {torch_avg_time_us / 1e3:.2f} ms")
    print(f"torch achieved TFLOPS: {(2 * M * N * K) / torch_avg_time_us / 1e6:.2f}")

    torch.cuda.empty_cache()
    workspace_bytes = (M * K + N * K + M * N) * 4
    workspace_count = testing.get_workspace_count(workspace_bytes, warmup_iterations, iterations)
    avg_time_us = testing.benchmark(
        matmul,
        workspace_generator=workspace_generator,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=False,
    )
    print(f"cute kernel execution time: {avg_time_us / 1e3:.2f} ms")
    print(f"cute achieved TFLOPS: {(2 * M * N * K) / avg_time_us / 1e6:.2f}")


if __name__ == "__main__":
    run_sgemm()