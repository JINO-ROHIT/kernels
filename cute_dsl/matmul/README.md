### matmul in cute dsl

a series of experiments to matmul in cute dsl and reach cublas performance. all experiments on rtx 4060 ti(ada lovelace architecture)


| kernel | TFLOP/s | perf relative to cuBLAS % |
|--------|---------|---------------------------|
| naive matmul | 1.33 | 11.5 |
| MMA matmul | 7.37 | 64.1 |
| MMA matmul with padding for bank conflicts| 8.34 | 72.5 |
| pipelining | 7.40 | 64.3 |
| pipelining coalesced| 11.17 | 97.2 |
| pipelining coalesced col major| 11.82| 102.8 |
| pytorch cuBLAS | 11.50 | 100.0 |


references
1. https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/sgemm.py
2. https://siboehm.com/articles/22/CUDA-MMM
3. https://www.aleksagordic.com/blog/matmul
4. https://salykova.github.io/sgemm-gpu