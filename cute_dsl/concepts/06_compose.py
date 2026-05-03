import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

@cute.kernel
def composition_kernel(A: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()

    print(f"A is {A}")  # (16,):(1,)

    transform = cute.make_layout((4, 2), stride=(2, 1))
    print(f"transform layout: {transform}")  # (4,2):(2,1)

    composed = cute.composition(A, transform)
    print(f"composed layout: {composed}")  # still (4,2) but now mapped into A

    cute.printf(composed)

@cute.jit
def explain_composition(A: cute.Tensor):
    composition_kernel(A).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )

A = torch.arange(0, 16, device="cuda", dtype=torch.float32)
tensor = from_dlpack(A, assumed_align=16)
explain_composition(tensor)

"""
what is cute.composition?

layout is basically a function that maps logical cordinates to memory offsets.

for example
layout (4,2):(4,1)
logical coord (i, j) --> offset = 4*i + 1*j

say you have a flat 1d array 
A = (16) --> [1, 2, 3, 4, 5 .... 16]

if you want to look at it via a 2d lens, that becomes the rhs

for example
rhs = (4, 2) : (2, 1)

composition chains the A and the rhs and gives a 2d way to address the 1d array, no data moved.


1. you can reshape without copying
2. swizzling becomes really easy with this.
3. after applying something like zipped divide for tiling, you need to compose them to get a way to index into the new tiles.
"""