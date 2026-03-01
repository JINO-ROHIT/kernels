import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch

@cute.kernel
def vadd_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    bidx, bdim, tidx = cute.arch.block_idx()[0], cute.arch.block_dim()[0], cute.arch.thread_idx()[0] 

    curr_idx = bidx * bdim + tidx

    _, n = mA.shape[1] # to get the second mode

    row = curr_idx // n
    col = curr_idx % n

    a_val = mA[(None, (row, col))].load()
    b_val = mB[(None, (row, col))].load()

    print(f"[DSL INFO] sliced gA = {mA[(None, (row, col))]}")
    print(f"[DSL INFO] sliced gB = {mB[(None, (row, col))]}")

    mC[(None, (row, col))] = a_val + b_val

@cute.jit
def vadd_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):

    tiled_a = cute.zipped_divide(mA, (1, 8))
    tiled_b = cute.zipped_divide(mB, (1, 8))
    tiled_c = cute.zipped_divide(mC, (1, 8))

    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   tiled A = {tiled_a}")
    print(f"[DSL INFO]   tiled B = {tiled_b}")
    print(f"[DSL INFO]   tiled C = {tiled_c}")

    kernel = vadd_kernel(tiled_a, tiled_b, tiled_c)
    # print(cute.size(tiled_c, mode=[1]))
    kernel.launch(
        grid = cute.size(tiled_c, mode=[1]) // 1024,
        block = 1024
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
