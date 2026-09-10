#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

static constexpr int GMEM_WIDTH = 64;
static constexpr int GMEM_HEIGHT = 64;
static constexpr int SMEM_WIDTH = 32;
static constexpr int SMEM_HEIGHT = 8;
static constexpr int TILE_BYTES = SMEM_WIDTH * SMEM_HEIGHT * sizeof(int);

__global__ void tma_add_tile_kernel(
    const __grid_constant__ CUtensorMap tensor_map, int tile_x, int tile_y) {
    __shared__ alignas(128) int smem[SMEM_HEIGHT][SMEM_WIDTH];

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar; // smem barrier to know when copy is finished

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    barrier::arrival_token token;

    if (threadIdx.x == 0) {
        const int coords[2] = {tile_x, tile_y};
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_shared,
            cuda::ptx::space_global,
            smem,
            &tensor_map,
            coords,
            cuda::device::barrier_native_handle(bar));

        token = cuda::device::barrier_arrive_tx(bar, 1, TILE_BYTES); // this thread arrived + barrier must wait for tile bytes of async copy traffic
    } else {
        token = bar.arrive(); // this only says this thread arrived
    }

    bar.wait(std::move(token)); // all threads wait till copy is done and all threads arrive at the barrier

    for (int i = threadIdx.x; i < SMEM_WIDTH * SMEM_HEIGHT; i += blockDim.x) {
        int y = i / SMEM_WIDTH;
        int x = i % SMEM_WIDTH;
        smem[y][x] += 1;
    }

    //smem to global memory TMA write
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    __syncthreads();

    if (threadIdx.x == 0) {
        const int coords[2] = {tile_x, tile_y};
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_global,
            cuda::ptx::space_shared,
            &tensor_map,
            coords,
            smem);
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{}); // Waits until there are 0 pending async bulk groups left before continuing
    }

    if (threadIdx.x == 0) {
        (&bar)->~barrier(); // destroy this barrier
    }
}

int main() {

    int* matrix = nullptr;
    cudaMallocManaged(&matrix, GMEM_WIDTH * GMEM_HEIGHT * sizeof(int));

    for (int y = 0; y < GMEM_HEIGHT; ++y) {
        for (int x = 0; x < GMEM_WIDTH; ++x) {
            matrix[y * GMEM_WIDTH + x] = y * 1000 + x;
        }
    }

    CUtensorMap tensor_map{};
    constexpr cuuint32_t rank = 2; // 2d tensor

    cuuint64_t global_dim[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
    cuuint64_t global_stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
    cuuint32_t box_dim[rank] = {SMEM_WIDTH, SMEM_HEIGHT}; // tile shape tma copies per instruction
    cuuint32_t element_stride[rank] = {1, 1}; // load every element

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

    int tile_x = 16;
    int tile_y = 24;

    tma_add_tile_kernel<<<1, 256>>>(tensor_map, tile_x, tile_y);
    cudaDeviceSynchronize();

    for (int y = 0; y < GMEM_HEIGHT; ++y) {
        for (int x = 0; x < GMEM_WIDTH; ++x) {
            int expected = y * 1000 + x;
            if (tile_x <= x && x < tile_x + SMEM_WIDTH &&
                tile_y <= y && y < tile_y + SMEM_HEIGHT) {
                expected += 1;
            }

            if (matrix[y * GMEM_WIDTH + x] != expected) {
                fprintf(stderr,
                        "mismatch at (%d,%d): expected %d, got %d\n",
                        x,
                        y,
                        expected,
                        matrix[y * GMEM_WIDTH + x]);
                cudaFree(matrix);
                return EXIT_FAILURE;
            }
        }
    }

    printf("passed: TMA copied a %dx%d tile at (%d,%d), added one, wrote it "
           "back\n",
           SMEM_WIDTH,
           SMEM_HEIGHT,
           tile_x,
           tile_y);

    cudaFree(matrix);
}
