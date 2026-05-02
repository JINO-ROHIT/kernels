import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch

@cute.jit
def hello_world(A: cute.Tensor):
    print(A) # (6, 2) : (2, 1)


    # tile (m, n) by (M, N) to obtain ((M, N), m', n')
    # where M' and N' are the number of block tiles
    tiled_A = cute.zipped_divide(A, (2, 2))
    print(tiled_A) # ((2,2),(3,1)):((2,1),(4,0))



A = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)

tensor = from_dlpack(A, assumed_align=16)
hello_world(tensor)