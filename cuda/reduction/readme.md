### implementing reduction kernels

1. `01_sum_reduction_gmem_naive.cu` (probably worse than cpu)

implements a 2 pass kernels -
- first kernel basically sums over the elements for that particular block. save the partial sum for each block into global memory
- second kernel iterates across each block partial sum and adds them.


2. `02_sum_reduction_gmem_single_block.cu`

- implements a tree based reduction that works for power of 2 size vectors. (high branch divergence). has a branch efficieny of 66%

3. `03_sum_reduction_fix_divergence.cu`

- implement tree bas3ed reduction that now strides from the backward direction, halving each time step. branch efficiency jumps to 99.32 %

4. `04_smem.cu`

- implement the reduction now inside shared memory.


### learning the series of reduction implementation by nvidia

1. `01_nvidia_lecture_baseline.cu` - 37.63 microsecond 46.7% DRAM throughput basically the bandwidth utilization 
2. `02_nvidia_lecture_interleaved.cu` - 28.83 microsecond 52.18% DRAM throughput basically the bandwidth utilization 
3. `02_nvidia_lecture_bank_conflict.cu` - 28.32 microsecond 56.27% DRAM throughput basically the bandwidth utilization 
(write a bit on bank conflict here later)
4. `04_nvidia_lecture_thread_idle.cu` - 20.80 microsecond 85.37% DRAM throughput
5. `05_nvidia_lecture_unroll_last_wrap.cu` - 19.17 microsecond 86.09% DRAM throughput
6. `06_nvidia_lecture_complete_unroll.cu` - 20.74 microsecond 86.6% DRAM throughput
7. `07_nvidia_lecture_multi_add.cu` - 17.38 microsecond 86.42% DRAM throughput