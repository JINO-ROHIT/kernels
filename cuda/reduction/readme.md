### implementing reduction kernels

1. `01_sum_reduction_gmem_naive` (probably worse than cpu)

implements a 2 pass kernels -
- first kernel basically sums over the elements for that particular block. save the partial sum for each block into global memory
- second kernel iterates across each block partial sum and adds them.


2. `02_sum_reduction_gmem_single_block`

- implements a tree based reduction that works for power of 2 size vectors. (high branch divergence)



