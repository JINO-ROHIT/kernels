import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


@cute.kernel
def copy_kernel(A: cute.Tensor, copy_atom: cute.TiledCopy):
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, cute.make_layout((16,)), 16)

    cute.printf(sA)

    thr_copy = copy_atom.get_slice(tidx)
    print(thr_copy)
    tAgA = thr_copy.partition_S(A)   
    tAsA = thr_copy.partition_D(sA)

    cute.printf(tAgA)
    cute.printf(tAsA)

    cute.copy(copy_atom, tAgA, tAsA)

    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)

    cute.printf(sA)

@cute.jit
def explain_copy(A: cute.Tensor):
    async_copy_atom_a = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        cutlass.Float32,
        num_bits_per_copy=A.element_type.width # 32
    )
    thr_layout = cute.make_layout((1,))   # change this thread and see the results vary
    val_layout = cute.make_layout((4,))   # 4 vals per thread
    tiled_copy = cute.make_tiled_copy_tv(async_copy_atom_a, thr_layout, val_layout)

    copy_kernel(A, tiled_copy).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )

A = torch.arange(0, 16, device="cuda", dtype=torch.float32)
tensorA = from_dlpack(A, assumed_align=16)
explain_copy(tensorA)