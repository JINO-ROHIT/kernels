#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cg = cooperative_groups;

static constexpr int CLUSTER_M = 2;
static constexpr int CLUSTER_N = 4;
static constexpr int CLUSTER_SIZE = CLUSTER_M * CLUSTER_N;

static constexpr int GMEM_WIDTH = 64;
static constexpr int GMEM_HEIGHT = 64;
static constexpr int TILE_WIDTH = 16;
static constexpr int TILE_HEIGHT = 8;
static constexpr int TILE_BYTES = TILE_WIDTH * TILE_HEIGHT * sizeof(int);

// 8 CTAs in one cluster:
//
//              n=0  n=1  n=2  n=3
//     m=0       0    1    2    3   <- producer CTAs for B
//     m=1       4    5    6    7
//
// This kernel only multicasts B tiles. Each producer in row m=0 loads one B tile
// from global memory and broadcasts it to the CTA below it:
//
//     CTA0 -> {CTA0, CTA4}
//     CTA1 -> {CTA1, CTA5}
//     CTA2 -> {CTA2, CTA6}
//     CTA3 -> {CTA3, CTA7}
__cluster_dims__(CLUSTER_N, CLUSTER_M, 1) __global__
void tma_multicast_b_kernel(const __grid_constant__ CUtensorMap tensor_map,
                            int* received_value,
                            int* received_mask) {
    __shared__ alignas(128) int smem[TILE_HEIGHT][TILE_WIDTH];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    cg::cluster_group cluster = cg::this_cluster();

    const int rank = cluster.block_rank();
    const int rank_m = rank / CLUSTER_N;
    const int rank_n = rank % CLUSTER_N;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    cluster.sync();

    uint16_t col_mask = 0;
    for (int m = 0; m < CLUSTER_M; ++m) {
        col_mask |= uint16_t(1u << (m * CLUSTER_N));
    }

    const uint16_t b_mask = uint16_t(col_mask << rank_n); // create mask for the mapping

    if (threadIdx.x == 0 && rank_m == 0) { // only top cta and first thread in each of them will issue the load cta0, cta1, cta2, cta3
        uint64_t* local_bar = cuda::device::barrier_native_handle(bar);

        // TMA will complete the barrier in every target CTA, so each target
        // barrier must expect TILE_BYTES before any CTA waits on it.
        for (int m = 0; m < CLUSTER_M; ++m) {
            int dst_rank = m * CLUSTER_N + rank_n;
            uint64_t* dst_bar = cluster.map_shared_rank(local_bar, dst_rank);

            cuda::ptx::mbarrier_expect_tx(cuda::ptx::sem_relaxed,
                                          cuda::ptx::scope_cluster,
                                          cuda::ptx::space_cluster,
                                          dst_bar,
                                          TILE_BYTES);
        }
    }

    cluster.sync();

    if (threadIdx.x == 0 && rank_m == 0) {
        const int coords[2] = {rank_n * TILE_WIDTH, 0};

        cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster,
                                        cuda::ptx::space_global,
                                        smem,
                                        &tensor_map,
                                        coords,
                                        cuda::device::barrier_native_handle(bar),
                                        b_mask);
    }

    auto token = bar.arrive();
    bar.wait(std::move(token));

    if (threadIdx.x == 0) {
        received_value[rank] = smem[0][0];
        received_mask[rank] = b_mask;
    }

    cluster.sync();

    if (threadIdx.x == 0) {
        (&bar)->~barrier();
    }
}

int main() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    if (props.major < 9) {
        fprintf(stderr, "This demo needs a Hopper or newer GPU, got sm_%d%d\n",
                props.major, props.minor);
        return EXIT_FAILURE;
    }

    int* matrix = nullptr;
    int* received_value = nullptr;
    int* received_mask = nullptr;

    cudaMallocManaged(&matrix, GMEM_WIDTH * GMEM_HEIGHT * sizeof(int));
    cudaMallocManaged(&received_value, CLUSTER_SIZE * sizeof(int));
    cudaMallocManaged(&received_mask, CLUSTER_SIZE * sizeof(int));

    for (int y = 0; y < GMEM_HEIGHT; ++y) {
        for (int x = 0; x < GMEM_WIDTH; ++x) {
            matrix[y * GMEM_WIDTH + x] = y * 1000 + x;
        }
    }

    for (int i = 0; i < CLUSTER_SIZE; ++i) {
        received_value[i] = -1;
        received_mask[i] = 0;
    }

    CUtensorMap tensor_map{};
    constexpr cuuint32_t rank = 2;

    cuuint64_t global_dim[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
    cuuint64_t global_stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
    cuuint32_t box_dim[rank] = {TILE_WIDTH, TILE_HEIGHT};
    cuuint32_t element_stride[rank] = {1, 1};

    cuInit(0);
    cuTensorMapEncodeTiled(&tensor_map,
                           CU_TENSOR_MAP_DATA_TYPE_INT32,
                           rank,
                           matrix,
                           global_dim,
                           global_stride,
                           box_dim,
                           element_stride,
                           CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_NONE,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    tma_multicast_b_kernel<<<dim3(CLUSTER_N, CLUSTER_M, 1), dim3(32, 1, 1)>>>(
        tensor_map, received_value, received_mask);
    cudaDeviceSynchronize();

    for (int rank = 0; rank < CLUSTER_SIZE; ++rank) {
        int rank_n = rank % CLUSTER_N;
        int expected = rank_n * TILE_WIDTH;

        if (received_value[rank] != expected) {
            fprintf(stderr,
                    "rank %d expected %d, got %d\n",
                    rank,
                    expected,
                    received_value[rank]);
            return EXIT_FAILURE;
        }

        printf("CTA %d received B tile starting at x=%d with mask 0x%02x\n",
               rank,
               rank_n * TILE_WIDTH,
               received_mask[rank]);
    }

    cudaFree(matrix);
    cudaFree(received_value);
    cudaFree(received_mask);
    return EXIT_SUCCESS;
}
