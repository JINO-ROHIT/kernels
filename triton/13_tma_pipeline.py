"""add tiles using descriptors, barriers, and rotating buffers.

MODAL_GPU=H100 modal run modal_runner.py --script 13_tma_pipeline.py
Use --args "--test" for edge cases or --args "--benchmark" for comparison.
"""

import argparse
import importlib

import pytest
import torch
import triton
import triton.testing
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

basics = importlib.import_module("12_tma")


@gluon.jit
def prefetch(tile_id, row, a_desc, b_desc, a_tiles, b_tiles, barriers,
             COLS: gl.constexpr, BUFFERS: gl.constexpr):
    slot = tile_id % BUFFERS
    ready = barriers.index(slot)
    # One barrier tracks the combined payload of A and B for this tile.
    mbarrier.expect(ready, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_load(a_desc, [row, tile_id * COLS], ready, a_tiles.index(slot))
    tma.async_load(b_desc, [row, tile_id * COLS], ready, b_tiles.index(slot))


@gluon.jit
def elementwise_add_tma_kernel(a_desc, b_desc, c_desc, width,
                               ROWS: gl.constexpr, COLS: gl.constexpr, BUFFERS: gl.constexpr):
    row = gl.program_id(0) * ROWS
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    a_tiles = gl.allocate_shared_memory(a_desc.dtype, [BUFFERS, ROWS, COLS], a_desc.layout)
    b_tiles = gl.allocate_shared_memory(b_desc.dtype, [BUFFERS, ROWS, COLS], b_desc.layout)
    output_tile = gl.allocate_shared_memory(c_desc.dtype, [ROWS, COLS], c_desc.layout)
    barriers = gl.allocate_shared_memory(gl.int64, [BUFFERS, 1], mbarrier.MBarrierLayout()) # one barrier for each buffer slot
    for init_slot in gl.static_range(BUFFERS):
        mbarrier.init(barriers.index(init_slot), count=1)

    tiles = gl.cdiv(width, COLS)
    # Fill only the slots that will actually be consumed, even for short rows.
    for initial_tile in gl.static_range(BUFFERS):
        if initial_tile < tiles:
            prefetch(initial_tile, row, a_desc, b_desc, a_tiles, b_tiles, barriers, COLS, BUFFERS)

    for tile_id in range(tiles):
        slot = tile_id % BUFFERS
        # A slot's barrier changes phase each time we reuse that slot.
        phase = (tile_id // BUFFERS) & 1
        mbarrier.wait(barriers.index(slot), phase)
        a = a_tiles.index(slot).load(layout)
        b = b_tiles.index(slot).load(layout)
        result = a + b

        # Do not overwrite the output staging tile until the previous store
        # has finished reading it. Its global write may still be in flight.
        tma.store_wait(pendings=0)
        output_tile.store(result)
        # Order register/shared-memory access before the async proxy accesses:
        # publish output_tile and finish reading the input slot before reuse.
        fence_async_shared()
        tma.async_store(c_desc, [row, tile_id * COLS], output_tile)

        future = tile_id + BUFFERS
        if future < tiles:
            prefetch(future, row, a_desc, b_desc, a_tiles, b_tiles, barriers, COLS, BUFFERS)

    tma.store_wait(pendings=0)
    for final_slot in gl.static_range(BUFFERS):
        mbarrier.invalidate(barriers.index(final_slot))


def elementwise_add_tma(a, b, c, XBLOCK=32, YBLOCK=64, num_buffers=2):
    if a.ndim != 2 or a.shape != b.shape or a.shape != c.shape:
        raise ValueError("A, B, C must have matching 2D shapes.")
    if any(t.dtype != torch.float32 for t in (a, b, c)):
        raise ValueError("This example uses float32 tensors.")
    if a.shape[1] % 4:
        raise ValueError("Use a width divisible by 4: TMA stores operate at 16-byte granularity.")
    if num_buffers not in (1, 2, 3):
        raise ValueError("Choose 1, 2, or 3 buffers for this example.")
    for tensor in (a, b, c):
        if tensor.stride(1) != 1 or tensor.stride(0) * tensor.element_size() % 16:
            raise ValueError("TMA needs contiguous columns and a 16-byte-aligned row stride.")
    shape = [XBLOCK, YBLOCK]
    shared_layout = gl.NVMMASharedLayout.get_default_for(shape, gl.float32)
    descriptors = [TensorDescriptor.from_tensor(t, shape, shared_layout) for t in (a, b, c)]
    return elementwise_add_tma_kernel[(triton.cdiv(a.shape[0], XBLOCK),)](
        *descriptors, a.shape[1], XBLOCK, YBLOCK, num_buffers, num_warps=4
    )


@pytest.mark.skipif(not basics.is_hopper_or_newer(), reason="Requires Hopper or newer")
@pytest.mark.parametrize("shape", [(5, 4), (35, 120), (1000, 2000)])
@pytest.mark.parametrize("num_buffers", [1, 2, 3])
def test_add(shape, num_buffers):
    # Covers fewer tiles than buffers, partial row/column tiles, and phase wrap.
    a = torch.randn(shape, device="cuda")
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    elementwise_add_tma(a, b, c, num_buffers=num_buffers)
    torch.testing.assert_close(c, a + b, atol=0, rtol=0)


# @pytest.mark.skipif(not basics.is_hopper_or_newer(), reason="Requires Hopper or newer")
# def test_padded_row_stride():
#     # Eight logical columns with a twelve-element physical row stride.
#     a = torch.randn((5, 12), device="cuda")[:, :8]
#     b = torch.randn((5, 12), device="cuda")[:, :8]
#     storage = torch.full((5, 12), -123.0, device="cuda")
#     c = storage[:, :8]
#     elementwise_add_tma(a, b, c, num_buffers=3)
#     torch.testing.assert_close(c, a + b, atol=0, rtol=0)
#     assert torch.all(storage[:, 8:] == -123).item()


def benchmark(rows, cols):
    previous = importlib.import_module("11_async_copy_pipeline")
    a = torch.randn((rows, cols), device="cuda")
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    expected = a + b
    cases = [("cp.async 32x64, 2 buffers", lambda: previous.elementwise_add_pipelined(a, b, c))]
    for buffers in (1, 2, 3):
        cases.append((f"TMA 32x64, {buffers} buffers",
                      lambda buffers=buffers: elementwise_add_tma(a, b, c, num_buffers=buffers)))
    cases.append(("TMA 64x128, 3 buffers", lambda: elementwise_add_tma(a, b, c, 64, 128, 3)))
    print(f"Benchmark shape: {rows}x{cols}; bandwidth counts two reads + one write.")
    for name, launch in cases:
        compiled = launch()
        torch.testing.assert_close(c, expected, atol=0, rtol=0)
        ms = triton.testing.do_bench(launch)
        tbps = 3 * a.numel() * a.element_size() / (ms * 1e-3) / 1e12
        print(f"{name}: {ms:.3f} ms, {tbps:.2f} TB/s; "
              f"registers/thread={compiled.n_regs}, shared={compiled.metadata.shared} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true")
    mode.add_argument("--benchmark", action="store_true")
    parser.add_argument("--rows", type=int, default=8192, help="Benchmark rows")
    parser.add_argument("--cols", type=int, default=8192, help="Benchmark columns, multiple of 4")
    args = parser.parse_args()
    basics.require_tma()
    if args.test:
        raise SystemExit(pytest.main([__file__, "-q"]))
    if args.benchmark:
        if args.rows < 1 or args.cols < 4 or args.cols % 4:
            parser.error("Use positive rows and cols >= 4 divisible by 4.")
        benchmark(args.rows, args.cols)
        return
    a = torch.arange(40, device="cuda", dtype=torch.float32).reshape(5, 8)
    b = torch.full_like(a, 100)
    c = torch.empty_like(a)
    for buffers in (1, 2, 3):
        elementwise_add_tma(a, b, c, num_buffers=buffers)
        torch.testing.assert_close(c, a + b, atol=0, rtol=0)
        print(f"PASS: {buffers} buffer(s), one partial tile")
    print("A:\n", a.cpu())
    print("B:\n", b.cpu())
    print("A + B:\n", c.cpu())
    print("Use --test to exercise multiple tiles and barrier phase changes.")


if __name__ == "__main__":
    main()
