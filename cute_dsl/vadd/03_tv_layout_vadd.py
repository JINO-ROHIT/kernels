import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch

@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # --------------------------------
    # slice for thread-block level view
    # --------------------------------
    blk_coord = ((None, None), bidx)

    # logical coord -> address
    blkA = gA[blk_coord]  # (TileM, TileN) -> physical address
    blkB = gB[blk_coord]  # (TileM, TileN) -> physical address
    blkC = gC[blk_coord]  # (TileM, TileN) -> physical address

    # --------------------------------
    # compose for thread-index & value-index to physical mapping
    # --------------------------------
    # blockA:    (TileM, TileN) -> physical address
    # tv_layout: (tid, vid)     -> (TileM, TileN)
    # tidfrgA = blkA o tv_layout
    # tidfrgA:   (tid, vid) -> physical address
    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    print(f"Composed with TV layout:")
    print(f"  tidfrgA: {tidfrgA.type}")

    # --------------------------------
    # slice for thread-level view
    # --------------------------------
    # `None` represent slice of the entire per-thread data
    thr_coord = (tidx, None)

    # slice for threads: vid -> address
    thrA = tidfrgA[thr_coord]  # (V) -> physical address
    thrB = tidfrgB[thr_coord]  # (V) -> physical address
    thrC = tidfrgC[thr_coord]  # (V) -> physical address

    print(f"thrA/B/C:")
    print(f"  thrA: {thrA.type}")
    print(f"  thrB: {thrB.type}")
    print(f"  thrC: {thrC.type}")

    thrC[None] = thrA.load() + thrB.load()

@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    # mA layout: (M, N):(N, 1)
    # TV layout map thread & value index to (16, 256) logical tile
    #  - contiguous thread index maps to mode-1 because input layout is contiguous on
    #     mode-1 for coalesced load-store
    #  - each thread load 8 contiguous element each row and load 4 rows
    thr_layout = cute.make_layout((4, 32), stride=(32, 1)) # we have 4 * 32 threads = 128
    val_layout = cute.make_layout((4, 8), stride=(8, 1)) # each thread owns 32 values
    #  each CTA (thread block) processes a 16 x 256 sub-matrix.
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}")
    print(f"TV Layout: {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    print(f"Tiled Input Tensors:")
    print(f"  gA: {gA.type}")
    print(f"  gB: {gB.type}")
    print(f"  gC: {gC.type}")

    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


M, N = 16384, 8192 

a = torch.randn(M, N, device = "cuda", dtype = torch.float16)  
b = torch.randn(M, N, device = "cuda", dtype = torch.float16)   
c = torch.zeros(M, N, device = "cuda", dtype = torch.float16)

num_elements = sum([a.numel(), b.numel(), c.numel()])

a_ = from_dlpack(a, assumed_align = 16, enable_tvm_ffi=True)  
b_ = from_dlpack(b, assumed_align = 16, enable_tvm_ffi=True)  
c_ = from_dlpack(c, assumed_align = 16, enable_tvm_ffi=True)

naive_elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_, options="--enable-tvm-ffi")
naive_elementwise_add_(a_, b_, c_) 

# torch.testing.assert_close(c, a + b) comment out to not mess the profiling hehehehe

'''
c = a + b
2 loads + 1 store
'''

def benchmark(callable, a_, b_, c_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )

    dtype = a_.element_type

    bytes_per_element = dtype.width // 8
    total_bytes = num_elements * bytes_per_element

    achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s

    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")

benchmark(naive_elementwise_add_, a_, b_, c_) #comment out to not mess the profiling