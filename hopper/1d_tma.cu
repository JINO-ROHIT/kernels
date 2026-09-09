#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

static constexpr size_t buf_len = 1024;
static constexpr size_t buf_bytes = buf_len * sizeof(int);

__global__ void add_one_kernel(int* data, size_t offset) {
    __shared__ alignas(16) int smem_data[buf_len];

// #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x); // inits barrier in shared mem, barrier waits for all threads in the block
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared); // make the barrier state visible to async proxy
    }
    __syncthreads();

    // from global to smem
    if (threadIdx.x == 0) {
        cuda::memcpy_async(smem_data,
                           data + offset,
                           cuda::aligned_size_t<16>(buf_bytes),
                           bar);
    }

    bar.arrive_and_wait(); // wait till all the threads arrive

    for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
        smem_data[i] += 1;
    }

    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    __syncthreads();

    if (threadIdx.x == 0) {
        cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                 cuda::ptx::space_shared,
                                 data + offset,
                                 smem_data,
                                 buf_bytes);
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
    }
}

void run_add_one(size_t offset) {
    constexpr int block_size = 256;
    const size_t n = offset + buf_len;
    const size_t bytes = n * sizeof(int);

    int* data = NULL;
    cudaMallocManaged(&data, bytes);

    for (size_t i = 0; i < n; ++i) {
        data[i] = (int)i;
    }

    add_one_kernel<<<1, block_size>>>(data, offset);
    cudaGetLastError();
    cudaDeviceSynchronize();

    for (size_t i = 0; i < n; ++i) {
        const int expected = (int)(i + (i >= offset));
        if (data[i] != expected) {
            fprintf(stderr, "mismatch at %zu: expected %d, got %d\n",
                    i, expected, data[i]);
            exit(EXIT_FAILURE);
        }
    }

    printf("passed: incremented %zu ints starting at offset %zu\n",
           buf_len, offset);

    cudaFree(data);
}

int main() {
    run_add_one(128);
    return 0;
}
