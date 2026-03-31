### Matrix Multiplication
A series of experiments to understand matrix multiplication on RTX 4060 Ti (Ada architecture, CUDA compute capability 8.9).


consider a matrix mul of 4096 * 4096 matrix

gflops = 10^9 flops/s
tflops = 10^12 flops/s

1. flops = 2 * M * N * K = 2 * 4096 * 4096 * 4096 = 137438953472 = 137.4 Gflops
2. data to read = 3 * 4096^2 * 4B = 201 MB (we need to read A, B and also C since its A * B + C, a FMA op)
3. data to write = 4096^2 * 4B = 67 MB



**whitepaper:** [NVIDIA Ada GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)

to reach: rtx 4060 ti tensor core peak for fp16 is ~88 TFLOP/s(dense)

#### benchmark results

| Kernel | Time (ms) | Throughput (TFLOP/s) | Speedup |
|--------|-----------|----------------------|---------|
| naive  | 165.0473 | 0.8327 | 1.0x |
| naive_ptx | 111.6388 | 1.2311 | 1.48x |
| coalesced  | 98.8539 | 1.3903 | 1.67x |
| coalesced_ptx | 101.0688 | 1.3599 | 1.63x |
| shared memory | 97.0385 | 1.4163 | 1.64x |
| coalesced shared memory | 85.1641 | 1.6138 | 1.94x |
| WMMA | 13.4257 | 10.2370 | 12.29x |
| naive MMA | 42.8513 | 3.2073 | 3.85x |


references -
1. https://siboehm.com/articles/22/CUDA-MMM