import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

DEBUG = False

@cute.kernel
def elementwise_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout, Crd: cute.Tensor, N: cute.Uint32, shape):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    block_coord = (None, bidx)

    block_A = gA[block_coord] 
    block_B = gB[block_coord]
    block_C = gC[block_coord]

    #print(block_A) #

    block_A = cute.composition(block_A, tv_layout)
    block_B = cute.composition(block_B, tv_layout)
    block_C = cute.composition(block_C, tv_layout)

    #print(block_A) # 
    
    # slice for thread view
    thr_crd = (tidx, None) 
    tAgA = block_A[thr_crd] 
    tBgB = block_B[thr_crd]
    tCgC = block_C[thr_crd]
    
    # creating register space
    tArA = cute.make_fragment_like(tAgA)
    tBrB = cute.make_fragment_like(tBgB)
    tCrC = cute.make_fragment_like(tCgC)

    Crd = Crd[block_coord]
    Crd = cute.composition(Crd, tv_layout)
    tCrd = Crd[thr_crd]
    tCrdPred = cute.make_fragment_like(tCrd, cutlass.Boolean)

    for i in cutlass.range(cute.size(tCrdPred), unroll=1):
        tCrdPred[i] = cute.elem_less(tCrd[i], shape)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)

    cute.copy(copy_atom, tAgA, tArA, pred=tCrdPred)
    cute.copy(copy_atom, tBgB, tBrB, pred=tCrdPred)

    res = tArA.load() + tBrB.load()
    tCrC.store(res)

    cute.copy(copy_atom, tCrC, tCgC, pred=tCrdPred)



@cute.jit
def host(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_layout((4,))
    tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(A, tiler)
    gB = cute.zipped_divide(B, tiler)
    gC = cute.zipped_divide(C, tiler)
    
    id_tensor = cute.make_identity_tensor(C.shape)
    Crd = cute.zipped_divide(id_tensor, tiler)

    grid_size = [cute.size(gC, mode=[1]), 1, 1]
    block_size = [cute.size(128), 1, 1]

    elementwise_add_kernel(gA, gB, gC, tv_layout, Crd, N, C.shape).launch( # we need to pass C shape to do the out of bounds checking
        grid=grid_size,
        block=block_size,
    )


M, N = 16384, 8192 

a = torch.randn(M, N, device = "cuda", dtype = torch.float16)  
b = torch.randn(M, N, device = "cuda", dtype = torch.float16)   
c = torch.zeros(M, N, device = "cuda", dtype = torch.float16)

total_elements = a.numel()
num_elements = sum([a.numel(), b.numel(), c.numel()])

a_ = from_dlpack(a, assumed_align = 16, enable_tvm_ffi=True)  
b_ = from_dlpack(b, assumed_align = 16, enable_tvm_ffi=True)  
c_ = from_dlpack(c, assumed_align = 16, enable_tvm_ffi=True)

naive_elementwise_add_ = cute.compile(host, a_, b_, c_, total_elements, options="--enable-tvm-ffi --generate-line-info")
naive_elementwise_add_(a_, b_, c_, total_elements) 

torch.testing.assert_close(c, a + b) #comment out to not mess the profiling hehehehe

'''
c = a + b
2 loads + 1 store
'''

def benchmark(callable, a_, b_, c_, n):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_, n),
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

benchmark(naive_elementwise_add_, a_, b_, c_, total_elements) # comment out to not mess the profiling

