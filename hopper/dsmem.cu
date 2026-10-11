#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

// launch one cluster containing two thread blocks
//   block rank 0 writes 1234 into its own shared memory
//   block rank 1 reads block rank 0's shared memory through DSMEM
//
// on hopper, blocks in the same cluster can directly address each
// other's shared memory. before hopper, shared memory was private to one block.
__global__ void dsmem_demo_kernel(int* output) {
    extern __shared__ int smem[];

    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    const int block_rank = cluster.block_rank();

    if (threadIdx.x == 0) {
        smem[0] = (block_rank == 0) ? 1234 : 0;
    }

    // Makes both blocks' shared-memory writes visible and also guarantees that
    // both blocks in the cluster are alive before any remote DSMEM access.
    cluster.sync();

    if (block_rank == 1 && threadIdx.x == 0) {
        int* block0_smem = cluster.map_shared_rank(smem, 0);

        output[0] = block0_smem[0];          // remote shared-memory read
        output[1] = smem[0];                 // this block's local shared memory
        output[2] = cluster.dim_blocks().x;  // number of blocks in the cluster
    }

    // Keep block rank 0 alive until block rank 1 finishes the remote read.
    cluster.sync();
}

int main() {

    int* output = nullptr;
    cudaMallocManaged(&output, 3 * sizeof(int));
    output[0] = output[1] = output[2] = -1;

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = {2, 1, 1};

    cudaLaunchConfig_t config = {};
    config.gridDim = 2;                    // total blocks in the grid
    config.blockDim = 32;                  // threads per block
    config.dynamicSmemBytes = sizeof(int); 
    config.attrs = attr;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, dsmem_demo_kernel, output);
    cudaDeviceSynchronize();

    printf("remote value read by block 1: %d\n", output[0]);
    printf("block 1 local shared value:  %d\n", output[1]);
    printf("cluster size:                %d blocks\n", output[2]);

    if (output[0] != 1234 || output[1] != 0 || output[2] != 2) {
        fprintf(stderr, "unexpected DSMEM result\n");
        cudaFree(output);
        return EXIT_FAILURE;
    }

    cudaFree(output);
    return EXIT_SUCCESS;
}
