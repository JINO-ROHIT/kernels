"""TMA lesson 3: publish a message between two CTAs with a full store wait.

Tutorial: https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html
MODAL_GPU=H100 modal run modal_runner.py --script 14_tma_store_wait.py
"""

import argparse
import importlib

import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

basics = importlib.import_module("12_tma")


@gluon.jit
def message_kernel(descriptor, ready, output, SIZE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    positions = gl.arange(0, SIZE, layout=layout)
    staging = gl.allocate_shared_memory(gl.int32, [SIZE], descriptor.layout)
    if gl.program_id(0) == 0:
        staging.store(positions + 1000)
        fence_async_shared()
        tma.async_store(descriptor, [0], staging)
        # read_only=True only permits reuse of staging. Publishing to another
        # CTA requires the global writes to complete BEFORE the release flag.
        tma.store_wait(pendings=0, read_only=False)
        gl.atomic_xchg(ready, 1, sem="release", scope="gpu") # for the program 1 to be notified
    else:
        observed = 0
        while observed == 0:
            observed = gl.atomic_add(ready, 0, sem="acquire", scope="gpu")
        completion = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(completion, count=1)
        mbarrier.expect(completion, descriptor.block_type.nbytes)
        tma.async_load(descriptor, [0], completion, staging)
        mbarrier.wait(completion, phase=0, deps=[staging])
        mbarrier.invalidate(completion)
        gl.store(output + positions, staging.load(layout))


def send_message(size=16):
    message = torch.full((size,), -1, device="cuda", dtype=torch.int32)
    received = torch.empty_like(message)
    ready = torch.zeros(1, device="cuda", dtype=torch.int32)
    layout = gl.NVMMASharedLayout.get_default_for([size], gl.int32)
    descriptor = TensorDescriptor.from_tensor(message, [size], layout)
    message_kernel[(2,)](descriptor, ready, received, size, num_warps=1)
    return received


@pytest.mark.skipif(not basics.is_hopper_or_newer(), reason="Requires Hopper or newer")
@pytest.mark.parametrize("size", [16, 64])
def test_message(size):
    expected = torch.arange(size, device="cuda", dtype=torch.int32) + 1000
    for _ in range(5):
        torch.testing.assert_close(send_message(size), expected, atol=0, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    basics.require_tma()
    if args.test:
        raise SystemExit(pytest.main([__file__, "-q"]))
    result = send_message()
    expected = torch.arange(16, device="cuda", dtype=torch.int32) + 1000
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    print("CTA 0 publishes [1000, ..., 1015] using TMA.")
    print("CTA 1 received:", result.cpu())
    print("PASS: full store wait -> release flag -> acquire flag -> TMA read.")


if __name__ == "__main__":
    main()
