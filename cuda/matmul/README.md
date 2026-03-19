### Matrix Multiplication
A series of experiments to understand matrix multiplication on RTX 4060 Ti (Ada architecture, CUDA compute capability 8.9).

**whitepaper:** [NVIDIA Ada GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)


to reach: rtx 4060 ti tensor core peak for fp16 is ~88 TFLOP/s(dense)

#### benchmark results

| Kernel | Time (ms) | Throughput (TFLOP/s) | Speedup |
|--------|-----------|----------------------|---------|
| naive  | 165.0473 | 0.8327 | 1.0x |
| naive_ptx | 111.6388 | 1.2311 | 1.48x |
| coalesced  | 98.8539 | 1.3903 | 1.67x |
| coalesced_ptx | 101.0688 | 1.3599 | 1.63x |
| WMMA | 13.4257 | 10.2370 | 12.29x |