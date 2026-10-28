"""
the main problem with global memory accesses is register pressure. 
For each LDG.E or STG.E, we need to compute the 64-bit address, compute the mask if needed, and store the result in registers. 
Vectorization can reduce register pressure, but the problem remains.

On Hopper and newer, TMA (Tensor Memory Accelerator) is a hardware feature for addressing N-dimensional arrays in global memory. 
TMAs trade the addressing flexibility of regular global memory instructions for a more concise address representation called the “tensor descriptor”.

TMA is used through objects called “tensor descriptors”. Tensor descriptors live in global memory and contain the shape, strides, base pointer, layout, 
and other information about the tensor. TMA reads and writes are fundamentally async, and we will need “mbarrier” objects to synchronize them.
Kernels that use TMAs accept descriptors as kernel arguments, which we can use to issue async tranfers
"""
# Run: MODAL_GPU=B200 modal run modal_runner.py --script 12_tma.py

import argparse

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier


def is_hopper_or_newer():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def require_tma():
    if not is_hopper_or_newer():
        raise RuntimeError(
            "TMA needs Hopper or newer. Run with MODAL_GPU=H100 (the default L4 cannot run TMA)."
        )


@gluon.jit
def memcpy_1d_tma_kernel(source, destination, BLOCK: gl.constexpr):
    start = gl.program_id(0) * BLOCK
    tile = gl.allocate_shared_memory(source.dtype, [BLOCK], source.layout)
    done = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout()) # one TMA load per program, one create [1] barrier
    mbarrier.init(done, count=1)

    # Register the expected byte count BEFORE starting the asynchronous read.
    # The descriptor supplies bounds/strides: no per-element pointers or masks.
    mbarrier.expect(done, source.block_type.nbytes)
    tma.async_load(source, [start], done, tile) # tile is the destination in the shared memory
    mbarrier.wait(done, phase=0)
    mbarrier.invalidate(done)

    # This copy never brings the payload into registers. Both transfers use TMA.
    tma.async_store(destination, [start], tile) # copy tile from smem to global memory
    tma.store_wait(pendings=0) # wait until zero pending TMA stores

    # if another program needs to consume the global memory output before this kernel ends, use:
    # tma.store_wait(pendings=0, read_only=False)

    """
    A TMA store has two parts:

    1. Read tile from shared memory
    2. Write that data into global memory

    | Call | Waits for | What is safe afterwards? |
    | store_wait(0) | Step 1 | Reuse the shared memory tile |
    | store_wait(0, read_only=False) | Both steps | Publish completion so another program can read the output |
    """


def memcpy_1d_tma(source, destination, block=64):
    if source.ndim != 1 or source.shape != destination.shape:
        raise ValueError("Use matching one-dimensional input and output tensors.")
    if source.dtype != torch.float32 or destination.dtype != source.dtype:
        raise ValueError("This learning example uses float32 tensors.")
    if block < 4 or block & (block - 1):
        raise ValueError("block must be a power of two >= 4.")
    shared_layout = gl.NVMMASharedLayout.get_default_for([block], gl.float32)
    src = TensorDescriptor.from_tensor(source, [block], shared_layout)
    dst = TensorDescriptor.from_tensor(destination, [block], shared_layout)
    return memcpy_1d_tma_kernel[(triton.cdiv(source.numel(), block),)](
        src, dst, block, num_warps=1
    )


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
@pytest.mark.parametrize("size", [40, 64, 500])
def test_memcpy_1d_tma(size):
    source = torch.randn(size, device="cuda")
    destination = torch.empty_like(source)
    memcpy_1d_tma(source, destination)
    torch.testing.assert_close(destination, source, atol=0, rtol=0)


def main():
    parser = argparse.ArgumentParser(description="TMA lesson 1: descriptor-based 1D copy")
    parser.add_argument("--test", action="store_true", help="Run correctness cases with pytest")
    args = parser.parse_args()
    require_tma()
    if args.test:
        raise SystemExit(pytest.main([__file__, "-q"]))
    source = torch.arange(40, device="cuda", dtype=torch.float32)
    destination = torch.empty_like(source)
    memcpy_1d_tma(source, destination, block=64)
    torch.testing.assert_close(destination, source, atol=0, rtol=0)
    print("Input:", source.cpu())
    print("Output:", destination.cpu())
    print("PASS: copied 40 elements using a 64-element tile.")
    print("The descriptor handles the 24 out-of-bounds positions automatically.")
    print("Load -> mbarrier wait -> TMA store -> store wait. No payload registers needed.")


if __name__ == "__main__":
    main()
