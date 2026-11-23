import argparse
import functools
import importlib.util
import inspect
import os
import sys
from pathlib import Path

# rocprofv3 preloads LLVM through COMGR. Keep Triton's bundled LLVM symbols
# local to Triton, then restore normal loading before importing PyTorch.
if "rocprofiler" in os.environ.get("LD_PRELOAD", "") and hasattr(os, "RTLD_DEEPBIND"):
    _dlopen_flags = sys.getdlopenflags()
    try:
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND)
        import triton
    finally:
        sys.setdlopenflags(_dlopen_flags)
else:
    import triton

import torch

# Out-of-tree amdgcnas peephole (post-assembly): install the amdgcn-stage hook
# when TRITON_AMDGCNAS_PLUGIN is set. Pure-Python text transform, no rebuild.
if os.environ.get("TRITON_AMDGCNAS_PLUGIN"):
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "plugins",
            "amdgcnas",
        ),
    )
    import amdgcnas_plugin
    from triton import knobs

    knobs.runtime.add_stages_inspection_hook = amdgcnas_plugin.inspect_stages_hook

DEVICE = triton.runtime.driver.active.get_active_torch_device()

name_to_torch_type = {"fp16": torch.float16, "bf16": torch.bfloat16}


def load_kernel(kernel):
    kernel_path = Path(kernel).expanduser()
    if kernel_path.suffix != ".py":
        kernel_path = kernel_path.with_suffix(".py")
    if not kernel_path.is_absolute():
        kernel_path = Path(__file__).resolve().parent / kernel_path
    if not kernel_path.is_file():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    spec = importlib.util.spec_from_file_location(f"_gemm_bench_{kernel_path.stem}", kernel_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "matmul", None)):
        raise ValueError(f"Kernel file must define a callable matmul: {kernel_path}")
    return module.matmul, kernel_path.stem


def get_x_vals():
    return [
        (4096, 4096, 512),
        (4096, 4096, 1024),
        (4096, 4096, 2048),
        (4096, 4096, 3072),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
        (4096, 4096, 16384),
        (4096, 4096, 32768),
    ]


def get_gemm_sizes(selected_k=None):
    sizes = get_x_vals()

    if selected_k is None:
        return sizes

    filtered = [s for s in sizes if s[2] == selected_k]

    if not filtered:
        raise ValueError(
            f"No GEMM size found with K={selected_k}. "
            f"Available K values: {[k for _, _, k in sizes]}"
        )

    return filtered


def get_dtypes(selected_dtype=None):
    default_dtypes = ["fp16", "bf16"]

    if selected_dtype is None:
        return default_dtypes

    if isinstance(selected_dtype, str):
        selected_dtype = [selected_dtype]

    invalid = set(selected_dtype) - set(default_dtypes)
    if invalid:
        raise ValueError(
            f"Unsupported dtype(s): {sorted(invalid)}. " f"Supported dtypes: {default_dtypes}"
        )

    return selected_dtype


def parse_args():

    parser = argparse.ArgumentParser(description="GEMM benchmark")
    parser.add_argument("--K", type=int, default=None, help="Select GEMM problem size with given K")
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=["fp16", "bf16"],
        default=None,
        help="Data type(s) to benchmark (default: fp16 bf16)",
    )
    parser.add_argument(
        "--kernel",
        default="01_naive",
        help="Kernel filename, with or without .py, relative to this script "
        "(default: 01_naive). Absolute paths are also accepted.",
    )
    parser.add_argument(
        "--rocprof",
        action="store_true",
        help="Rocprof mode: run kernel 1000 times without do_bench. "
        "Use with rocprofv3 --kernel-trace to measure timing externally.",
    )
    parser.add_argument(
        "--rotating-buffer-size",
        type=int,
        default=512,
        help="Total size (MB) of rotating tensors (a, b, c) for rocprof mode. "
        "Should exceed GPU cache (L2+MALL) size. (default: 512)",
    )
    parser.add_argument(
        "--schedule-hint",
        default=None,
        help='Triton schedule_hint compile option for the kernel ("" turns the MFMA scheduler '
        "off; default: the kernel's own choice; ignored if matmul() has no schedule_hint parameter)",
    )
    return parser.parse_args()


def test_correctness(matmul, dtype, gemm_sizes, kernel_name):
    if dtype == "f8":
        torch_dtype = torch.float16
    else:
        torch_dtype = name_to_torch_type[dtype]

    for M, N, K in gemm_sizes:
        a = torch.rand((M, K), device=DEVICE, dtype=torch_dtype) - 0.5
        b = torch.rand((N, K), device=DEVICE, dtype=torch_dtype).T - 0.5
        if dtype == "f8":
            a = a.to(torch.float8_e5m2)
            b = b.to(torch.float8_e5m2)
        triton_output = matmul(a, b)
        if dtype == "f8":
            torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        else:
            torch_output = torch.matmul(a, b)
        if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
            print(f"[{kernel_name}] {M=} {N=} {K=} {dtype=}: ✅ Triton and Torch match")
        else:
            print(f"[{kernel_name}] {M=} {N=} {K=} {dtype=}: ❌ Triton and Torch differ")


def gen_rotating_tensors(M, N, K, torch_dtype, rotating_buffer_size_mb=512):
    """Allocate multiple copies of (a, b, c) tensors to exceed GPU cache size.

    Each iteration of the benchmark loop uses a different copy via i % block_count,
    so cached data from the previous iteration is useless and the kernel always
    starts with cold caches.
    """
    elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
    a_size = M * K * elem_bytes
    b_size = K * N * elem_bytes
    c_size = M * N * elem_bytes
    total_size = a_size + b_size + c_size

    block_count = max(1, rotating_buffer_size_mb * 1024 * 1024 // total_size)

    a_list, b_list, c_list = [], [], []
    for _ in range(block_count):
        a_list.append(torch.randn((M, K), device=DEVICE, dtype=torch_dtype))
        b_list.append(torch.randn((N, K), device=DEVICE, dtype=torch_dtype).T)
        c_list.append(torch.empty((M, N), device=DEVICE, dtype=torch_dtype))

    return a_list, b_list, c_list, block_count


def run_rocprof_iterations(
    matmul, dtypes, gemm_sizes, kernel_name, n_iters=1000, rotating_buffer_size_mb=512
):
    """Run the kernel n_iters times for each dtype/size combo using rotating tensors.

    Rotating tensors ensure each iteration reads from different memory addresses,
    defeating GPU cache and producing cold-cache timings similar to real workloads.
    Designed to be wrapped by rocprofv3 --kernel-trace for external timing.
    """
    for dtype in dtypes:
        torch_dtype = name_to_torch_type[dtype]
        for M, N, K in gemm_sizes:
            a_list, b_list, c_list, block_count = gen_rotating_tensors(
                M, N, K, torch_dtype, rotating_buffer_size_mb
            )
            print(
                f"[{kernel_name}] {M=} {N=} {K=} {dtype=}: "
                f"rotating tensors: {block_count} copies, "
                f"{block_count * (M*K + K*N + M*N) * a_list[0].element_size() / 1024**2:.0f} MB"
            )
            # Warmup
            matmul(a_list[0], b_list[0], c_list[0])
            torch.cuda.synchronize()
            for i in range(n_iters):
                idx = i % block_count
                matmul(a_list[idx], b_list[idx], c_list[idx])
            torch.cuda.synchronize()
            print(f"[{kernel_name}] {M=} {N=} {K=} {dtype=}: {n_iters} iterations done")


def main():
    args = parse_args()
    matmul, kernel_name = load_kernel(args.kernel)
    if args.schedule_hint is not None:
        # Only pass the option to kernels that explicitly support it.
        if "schedule_hint" in inspect.signature(matmul).parameters:
            matmul = functools.partial(matmul, schedule_hint=args.schedule_hint)
        else:
            print(f"[{kernel_name}] does not use the MFMA scheduler; --schedule-hint ignored")

    gemm_sizes = get_gemm_sizes(args.K)
    dtypes = get_dtypes(args.dtype)

    for dtype in dtypes:
        test_correctness(matmul, dtype, gemm_sizes, kernel_name)

    if args.rocprof:
        run_rocprof_iterations(
            matmul,
            dtypes,
            gemm_sizes,
            kernel_name,
            rotating_buffer_size_mb=args.rotating_buffer_size,
        )
        return

    series = [(dtype, provider) for dtype in dtypes for provider in ("Triton", "rocBLAS")]
    configs = [
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=gemm_sizes,
            line_arg="series_key",
            line_vals=series,
            line_names=[f"{provider} {dtype}" for dtype, provider in series],
            ylabel="TFLOPS",
            plot_name=f"matmul-performance-{kernel_name}",
            args={},
        )
    ]

    # Adjacent providers share the same input tensors for each case.
    @functools.lru_cache(maxsize=1)
    def benchmark_inputs(M, N, K, dtype):
        torch_dtype = name_to_torch_type[dtype]
        a = torch.randn((M, K), device=DEVICE, dtype=torch_dtype)
        b = torch.randn((N, K), device=DEVICE, dtype=torch_dtype).T
        return a, b

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, series_key):
        dtype, provider = series_key
        a, b = benchmark_inputs(M, N, K, dtype)
        quantiles = [0.5, 0.2, 0.8]
        if provider == "rocBLAS":
            # hipBLAS dispatches these GEMMs to rocBLAS on AMD.
            previous = torch.backends.cuda.preferred_blas_library()
            try:
                torch.backends.cuda.preferred_blas_library("hipblas")
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.mm(a, b), quantiles=quantiles)
            finally:
                torch.backends.cuda.preferred_blas_library(previous)
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)

        def perf(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return perf(ms), perf(max_ms), perf(min_ms)

    print(f"\n{kernel_name}:")
    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
