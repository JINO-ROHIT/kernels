import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch

@cute.kernel
def vadd_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, tv_layout: cute.Layout):
    bidx, tidx = cute.arch.block_idx()[0], cute.arch.thread_idx()[0] 

    blk_coord = ((None, None), bidx)

    blkA = mA[blk_coord]  # (TileM, TileN) -> physical address
    blkB = mB[blk_coord]  # (TileM, TileN) -> physical address
    blkC = mC[blk_coord]

    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    thr_coord = (tidx, None)
    thrA = tidfrgA[thr_coord]  
    thrB = tidfrgB[thr_coord] 
    thrC = tidfrgC[thr_coord] 

    thrC[None] = thrA.load() + thrB.load()

@cute.jit
def vadd_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):

    coalesced_ldst_bytes = 16

    # print(mA._dtype)

    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0)) 
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] thread layout: {thr_layout}")
    print(f"[DSL INFO] value layout: {val_layout}")
    print(f"[DSL INFO] Tiler: {tiler_mn}")
    print(f"[DSL INFO] TV Layout: {tv_layout}")

    tiled_a = cute.zipped_divide(mA, tiler_mn)
    tiled_b = cute.zipped_divide(mB, tiler_mn)
    tiled_c = cute.zipped_divide(mC, tiler_mn)

    print(f"[DSL INFO] tiled A: {tiled_a}")

    kernel = vadd_kernel(tiled_a, tiled_b, tiled_c, tv_layout)
    print(f"blocks launched : {cute.size(tiled_c, mode=[1])}")
    print(f"threads launched : {cute.size(tv_layout, mode = [0])}")
    kernel.launch(
        grid = cute.size(tiled_c, mode = [1]),
        block = cute.size(tv_layout, mode = [0])
    )


M, N = 16384, 8192 

a = torch.randn(M, N, device = "cuda", dtype = torch.float16)  
b = torch.randn(M, N, device = "cuda", dtype = torch.float16)   
c = torch.zeros(M, N, device = "cuda", dtype = torch.float16)

num_elements = sum([a.numel(), b.numel(), c.numel()])

a_ = from_dlpack(a, assumed_align = 16)  
b_ = from_dlpack(b, assumed_align = 16)  
c_ = from_dlpack(c, assumed_align = 16)

naive_elementwise_add_ = cute.compile(vadd_host, a_, b_, c_)
naive_elementwise_add_(a_, b_, c_) 

torch.testing.assert_close(c, a + b) 

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

benchmark(naive_elementwise_add_, a_, b_, c_)


'''
lets trace this real quick

thread_layout = (4, 64) : (64, 1)
we have 4 threads alone M(row), 64 thread along N(col)

val_layout = (16, 16) : (16, 1)
we have 16 values alone M(row), 16 values along N(col)

tiler = (64, 1024)
tv_layout = ((64,4), (16,16)) : ((1024,16),(64,1))

we have 256 threads each with 256 values

tv layout says where in tiler you land up
(check by doing size(tiler) = size(tv_layout))

now we tile the tensors
tensor A = (64,1024),(256,8)):((8192,1),(524288,1024))

the tile T (64, 1024) repeated
256 times down the M direction (rows)
8 times across the N direction (cols)

so how many blocks and threads we need?
blocks = 2048 (T needs 256 * 8 blocks) 1 block = 1 tile
threads = 256 (16 * 16)

'''