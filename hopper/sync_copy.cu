#include <cuda_runtime.h>

__global__ void CopyToSharedMemOne(const int* global_data, int* out) {
    extern __shared__ int shm[];

    shm[threadIdx.x] = global_data[threadIdx.x];
    __syncthreads();

    out[threadIdx.x] = shm[threadIdx.x];
}

__global__ void CopyToSharedMemLoop(const int* global_data, int* out,
                                    int num_per_thread) {
    extern __shared__ int shm[];

    for (int i = 0; i < num_per_thread; ++i) {
        shm[threadIdx.x * num_per_thread + i] =
            global_data[threadIdx.x * num_per_thread + i];
    }
    __syncthreads();

    for (int i = 0; i < num_per_thread; ++i) {
        out[threadIdx.x * num_per_thread + i] =
            shm[threadIdx.x * num_per_thread + i];
    }
}

int main() {
    constexpr int block_size = 256;
    constexpr int num_per_thread = 4;
    constexpr int n = block_size * num_per_thread;

    int* global_data;
    int* out;
    cudaMalloc(&global_data, n * sizeof(int));
    cudaMalloc(&out, n * sizeof(int));

    CopyToSharedMemOne<<<1, block_size, block_size * sizeof(int)>>>(
        global_data, out);

    CopyToSharedMemLoop<<<1, block_size,
                          block_size * num_per_thread * sizeof(int)>>>(
        global_data, out, num_per_thread);

    cudaFree(global_data);
    cudaFree(out);
    return 0;
}
