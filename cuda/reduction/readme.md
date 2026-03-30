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