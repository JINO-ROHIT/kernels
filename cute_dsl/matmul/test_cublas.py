import torch
import time


def run_cublas_benchmark(
    M: int = 2048,
    K: int = 2048,
    N: int = 2048,
    verify: bool = True,
    warmup_iterations: int = 10,
    iterations: int = 100,
):
    print(f"Running cuBLAS benchmark with M={M}, N={N}, K={K}")

    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    if verify:
        expected = torch.matmul(a, b)
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    for _ in range(warmup_iterations):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    tic = time.perf_counter()
    for _ in range(iterations):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    avg_time_us = (time.perf_counter() - tic) * 1e6 / iterations

    print(f"cuBLAS execution time: {avg_time_us / 1e3:.2f} ms")
    print(f"cuBLAS achieved TFOPS: {(2 * M * N * K) / avg_time_us / 1e6:.2f}")

    if verify:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        torch.testing.assert_close(c, expected, atol=1e-2, rtol=1e-2)
        print("Verification passed!")


if __name__ == "__main__":
    run_cublas_benchmark()