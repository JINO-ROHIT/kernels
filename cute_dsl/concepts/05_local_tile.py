import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch


@cute.kernel
def local_tile_kernel(A: cute.Tensor):
    print(f"A is {A}")  # (16,16):(16,1)
    cute.print_tensor(A)

    # GEMM style CTA tiler: (M, N, K)
    # A only has (M, K), so the N mode is projected away below.
    cta_tiler = (4, 8, 4)
    tiler_coord = (1, 0, 2) # try (0, 0, 0) then (0, 0, 1) (1, 0, 0) yuull see the pattern 
    # 1 means skip the first fours rows
    # 2 means skip the first two columns 

    print(f"cta_tiler is {cta_tiler}")
    print(f"tiler_coord is {tiler_coord}")

    gA = cute.local_tile(A, cta_tiler, tiler_coord, proj=(1, None, 1))
    print(f"gA is {gA}")  # (4,4):(16,1)
    cute.printf(gA)

    # we can also get the same result using zipped divide
    zip = cute.zipped_divide(A, (4, 4)) #  ((4,4),(4,4)):((16,1),(64,4))>
    # (None, None) --> keep the full 4 x 4 tile
    # (1, 2)       --> choose M tile index 1, K tile index 2
    my_tile = zip[(None, None), (1, 2)]
    cute.printf(my_tile)


@cute.jit
def explain_local_tile(A: cute.Tensor):
    local_tile_kernel(A).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


A = torch.arange(0, 16 * 16, device="cuda", dtype=torch.float32).reshape(16, 16)
tensor = from_dlpack(A, assumed_align=16)

explain_local_tile(tensor)
