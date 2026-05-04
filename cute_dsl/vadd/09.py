import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

DEBUG = False

@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    block_coord = ((None, None), bidx)

    block_A = gA[block_coord] 
    block_B = gB[block_coord]
    block_C = gC[block_coord]
    block_Crd = cC[block_coord]

    #print(block_A) # (16,256):(8192,1)

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)

    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)


    tAgA = thr_copy_A.partition_S(block_A)
    tBgB = thr_copy_B.partition_S(block_B)
    tCgC = thr_copy_C.partition_S(block_C)
    
    # creating register space
    tArA = cute.make_fragment_like(tAgA)
    tBrB = cute.make_fragment_like(tBgB)
    tCrC = cute.make_fragment_like(tCgC)

    tCrd = thr_copy_C.partition_S(block_Crd)
    tCrdPred = cute.make_rmem_tensor(tCrd.shape, cutlass.Boolean)
    for i in range(0, cute.size(tCrdPred), 1):
        tCrdPred[i] = cute.elem_less(tCrd[i], shape)

    cute.copy(copy_atom_load, tAgA, tArA, pred=tCrdPred)
    cute.copy(copy_atom_load, tBgB, tBrB, pred=tCrdPred)

    res = tArA.load() + tBrB.load()
    tCrC.store(res)

    cute.copy(copy_atom_load, tCrC, tCgC, pred=tCrdPred)



@cute.jit
def host(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    copy_bits = 128
    dtype = A.element_type
    vector_size = copy_bits // dtype.width

    #print(dtype) # float16

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(val_layout) # (4, 8): (8, 1)

    gA = cute.zipped_divide(A, tiler)
    gB = cute.zipped_divide(B, tiler)
    gC = cute.zipped_divide(C, tiler)
    
    id_tensor = cute.make_identity_tensor(C.shape)
    Crd = cute.zipped_divide(id_tensor, tiler)

    grid_size = [cute.size(gC, mode=[1]), 1, 1]
    block_size = [cute.size(tv_layout, mode=[0]), 1, 1]

    elementwise_add_kernel(gA, gB, gC, Crd, C.shape, thr_layout, val_layout).launch(
        grid=grid_size,
        block=block_size,
    )


M, N = 16384, 8192 

a = torch.randn(M, N, device = "cuda", dtype = torch.float16)  
b = torch.randn(M, N, device = "cuda", dtype = torch.float16)   
c = torch.zeros(M, N, device = "cuda", dtype = torch.float16)

num_elements = sum([a.numel(), b.numel(), c.numel()])

a_ = from_dlpack(a, assumed_align = 16, enable_tvm_ffi=True)  
b_ = from_dlpack(b, assumed_align = 16, enable_tvm_ffi=True)  
c_ = from_dlpack(c, assumed_align = 16, enable_tvm_ffi=True)

naive_elementwise_add_ = cute.compile(host, a_, b_, c_, options="--enable-tvm-ffi --generate-line-info")
naive_elementwise_add_(a_, b_, c_) 

torch.testing.assert_close(c, a + b) #comment out to not mess the profiling hehehehe

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

benchmark(naive_elementwise_add_, a_, b_, c_) # comment out to not mess the profiling


# def test():

#     N = 1 << 15
#     #N = 100
#     a = torch.randn(N, device="cuda", dtype=torch.float32)
#     b = torch.randn(N, device="cuda", dtype=torch.float32)
#     c = torch.empty_like(a)

#     A = cute.runtime.from_dlpack(a)
#     B = cute.runtime.from_dlpack(b)
#     C = cute.runtime.from_dlpack(c)

#     compiled_host = cute.compile(host, A, B, C, N)
#     compiled_host(A, B, C, N)

#     torch.testing.assert_close(a + b, c)

# if __name__ == "__main__":
#     test()
