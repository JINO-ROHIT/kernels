"""Launch the naive GEMM on 1024 x 1024 FP16 inputs for profiling."""

import os
import runpy
import sys
from pathlib import Path

_dlopen_flags = sys.getdlopenflags()
try:
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)
    import triton
    import torch
finally:
    sys.setdlopenflags(_dlopen_flags)


def main():
    matmul = runpy.run_path(str(Path(__file__).with_name("00_naive.py")))["matmul"]
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    c = matmul(a, b)
    torch.cuda.synchronize()
    print(f"v0_naive completed: shape={tuple(c.shape)}, dtype={c.dtype}")


if __name__ == "__main__":
    main()
