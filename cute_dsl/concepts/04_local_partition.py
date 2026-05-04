import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch


@cute.kernel
def local_partition_kernel(A: cute.Tensor, thread_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()

    print(f"A is {A}") # (8,8):(8,1)

    print(f"thread layout: {thread_layout}") # (2, 4):(4, 1)

    thrA = cute.local_partition(A, thread_layout, tidx) # (4, 2):(16, 4)
    print(f"local partition for thrA is {thrA}")
    cute.printf(thrA)


@cute.jit
def explain_local_partition(A: cute.Tensor):
    thread_layout = cute.make_layout((2, 4), stride=(4, 1))

    local_partition_kernel(A, thread_layout).launch(
        grid=(1, 1, 1),
        block=(cute.size(thread_layout), 1, 1),
    )


A = torch.arange(0, 8 * 8, device="cuda", dtype=torch.float32).reshape(8, 8)
tensor = from_dlpack(A, assumed_align=16)

explain_local_partition(tensor)


"""
zipped_divide(A, thread_layout) - this tiles the tensor by the thread layout shape, giving you a view of all threads data together

A: (8,8):(8,1) tiled by thread_layout shape (2,4)

result shape: ((2, 4),                  8)
            (tile shape)   (remaining tiles (8/2=4, 8/4=2))


local_partition(A, thread_layout, tidx) - it gives a specific slice the thread owns.

A.shape = (8, 8)
thread_layout.shape = (2, 4)

So one 2 x 4 thread tile covers 8 positions in A, one per thread.

local_partition(A, thread_layout, tidx) fixes one position inside every 2 x 4 tile, then walks over all repeated tiles in the matrix.

Since:

8 rows / 2 thread rows = 4 row tiles
8 cols / 4 thread cols = 2 col tiles

each thread gets (4, 2) elements.

"""