"""
software pipelining is an optimization technique for hiding the latencies of operations that execute asynchronously with respect to each other. 
ff we prefetch the loads of the next operands before the current add, we can overlap it with the add and store. 
This requires multi-buffering shared memory, so it can be used by both the load and the add at the same time.

Based on the relative latencies of the operations, we can determine the “pipeline depth”. 
This is the number of prefetched loads in-flight. For example, if a load takes 3 times as long as the add, we should pipeline with depth 3 so each load has time to complete 
before the operands are needed.

think of it like , 3 loads per add, when the first tile starts add, we already have 3 loads.

"""
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

@gluon.jit
def issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    # Masking the loads by yoffs < ynumel will handle the case where there
    # are fewer blocks to copy than `num_buffers-1`.
    yoffs = copy_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    cp.async_load(a_smem.index(copy_idx % num_buffers),  #
                  a_ptrs + ystride_a * yoffs[None, :], mask)
    cp.async_load(b_smem.index(copy_idx % num_buffers),  #
                  b_ptrs + ystride_b * yoffs[None, :], mask)
    cp.commit_group()
    return copy_idx + 1


@gluon.jit
def perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr, layout: gl.constexpr):
    a_val = a_smem.index(read_idx % num_buffers).load(layout)
    b_val = b_smem.index(read_idx % num_buffers).load(layout)
    c_val = a_val + b_val
    yoffs = read_idx * YBLOCK + y_idx
    mask = xmask & (yoffs < ynumel)[None, :]
    gl.store(c_ptrs + ystride_c * yoffs[None, :], c_val, mask=mask)
    return read_idx + 1


@gluon.jit
def elementwise_add_pipelined_kernel(  #
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,  #
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,  #
        smem_layout: gl.constexpr, num_buffers: gl.constexpr,  #
):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoffs = pid * XBLOCK + gl.arange(0, XBLOCK, gl.SliceLayout(1, layout))
    a_ptrs = a_ptr + xstride_a * xoffs[:, None]
    b_ptrs = b_ptr + xstride_b * xoffs[:, None]
    c_ptrs = c_ptr + xstride_c * xoffs[:, None]

    y_idx = gl.arange(0, YBLOCK, gl.SliceLayout(0, layout))
    xmask = (xoffs < xnumel)[:, None]

    # New: declare multi-buffered shared memory by adding a pipelining dimension
    # to the descriptors.
    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    copy_idx = 0
    read_idx = 0

    # Peel the `num_buffers-1` iterations from the inner loop to prefetch the
    # first set of copies, filling our pipeline.
    for _ in gl.static_range(num_buffers - 1):
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                               YBLOCK, num_buffers)

    # Inner loop iterations with overlapped copies and compute. This is the
    # steady state of the pipeline.
    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        # Issue the overlapped copy.
        copy_idx = issue_loads(copy_idx, a_smem, b_smem, a_ptrs, ystride_a, b_ptrs, xmask, ynumel, y_idx, ystride_b,
                               YBLOCK, num_buffers)

        # Wait for `num_buffers-1` copies to complete, which is the last issued
        # copy. We can process that buffer.
        cp.wait_group(num_buffers - 1)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK, num_buffers,
                               layout)

    # Peeled iterations to drain the pipeline.
    for i in gl.static_range(num_buffers - 1):
        cp.wait_group(num_buffers - 2 - i)
        read_idx = perform_add(read_idx, a_smem, b_smem, c_ptrs, ynumel, ystride_c, y_idx, xmask, YBLOCK, num_buffers,
                               layout)


def elementwise_add_pipelined(A, B, C, XBLOCK=32, YBLOCK=64, num_buffers=2):
    assert A.shape == B.shape == C.shape
    xnumel, ynumel = A.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )
    smem_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    return elementwise_add_pipelined_kernel[grid](
        A, B, C, xnumel, ynumel,  #
        *A.stride(), *B.stride(), *C.stride(),  #
        XBLOCK, YBLOCK, smem_layout, num_buffers)


@pytest.mark.parametrize("xnumel, ynumel", [(1000, 2000), (4000, 120)])
@pytest.mark.parametrize("XBLOCK, YBLOCK", [(32, 64)])
@pytest.mark.parametrize("num_buffers", [1, 2, 3])
def test_elementwise_add_pipelined(xnumel, ynumel, XBLOCK, YBLOCK, num_buffers):
    a = torch.randn(xnumel, ynumel, device="cuda")
    b = torch.randn(xnumel, ynumel, device="cuda")
    c = torch.empty_like(a, device="cuda")
    elementwise_add_pipelined(a, b, c, XBLOCK, YBLOCK, num_buffers)
    torch.testing.assert_close(a + b, c, atol=0, rtol=0)

def get_throughput(ms, C):
    # Because this kernel is memory-bound, we will measure bandwidth.
    tbytes = (3 * C.numel() * C.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)

if __name__ == "__main__":
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")
    ms = triton.testing.do_bench(lambda: elementwise_add_pipelined(A, B, C, num_buffers=2))
    print(f"elementwise_add_pipelined (double buffer): {get_throughput(ms, C):.2f} TB/s")
    ms = triton.testing.do_bench(lambda: elementwise_add_pipelined(A, B, C, num_buffers=3))
    print(f"elementwise_add_pipelined (triple buffer): {get_throughput(ms, C):.2f} TB/s")


"""
Pipelining with async copy yields a modest speedup. But notice that increasing the number of buffers further does not yield more performance, confirming that this kernel is memory-bound.

One of the major issues getting in the way of more performance is register pressure. 
For each element, we need to store the 32-bit result, compute a 64-bit address, and the mask. 
With two inputs, this results in a lot of registers, where the maximum registers per thread is 256. 
This is why we used a small [32, 64] block size for the kernel.
In the next tutorial, we will convert tensor descriptors and TMAs, and see how they can help reduce register pressure at the cost of addressing flexibility.
"""