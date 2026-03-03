// launch multiple blocks with a single thread, partial sum across each block, and second pass sum them

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void reduce_gmem(const float* __restrict__ input,
                            float* __restrict__ output,
                            const int size) {
    unsigned int block_start = blockIdx.x * blockDim.x;
    unsigned int block_end   = min(block_start + blockDim.x, size);

    float s = 0.0f;
    for (unsigned int i = block_start; i < block_end; i++) {
        s += input[i];
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = s;
    }
}

__global__ void reduce_final(const float* __restrict__ partial,
                             float* __restrict__ result,
                             const int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += partial[i];
    *result = s;
}

int main() {
    const int M         = 10000;
    const int BLOCKSIZE = 1;
    const int NUM_BLOCKS = CEIL_DIV(M, BLOCKSIZE);
    const int WARMUP_ITERS = 10;
    const int BENCH_ITERS  = 100;

    float *h_A = (float*)malloc(M * sizeof(float));

    float cpu_sum = 0.0f;
    for (int i = 0; i < M; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        cpu_sum += h_A[i];
    }

    float *d_A, *d_partial, *d_out;
    cudaMalloc(&d_A,       M          * sizeof(float));
    cudaMalloc(&d_partial, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_out,     1          * sizeof(float));

    cudaMemcpy(d_A, h_A, M * sizeof(float), cudaMemcpyHostToDevice);

    // --- correctness ---
    reduce_gmem<<<NUM_BLOCKS, BLOCKSIZE>>>(d_A, d_partial, M);
    cudaDeviceSynchronize();
    reduce_final<<<1, 1>>>(d_partial, d_out, NUM_BLOCKS);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float gpu_sum = 0.0f;
    cudaMemcpy(&gpu_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("cpu sum : %.6f\n", cpu_sum);
    printf("gpu sum : %.6f\n", gpu_sum);
    printf("diff    : %.6f\n", fabsf(cpu_sum - gpu_sum));
    printf("match   : %s\n\n", fabsf(cpu_sum - gpu_sum) < 1e-1f ? "YES" : "NO");

    // --- benchmarking with cuda events(also do ncu) ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        reduce_gmem<<<NUM_BLOCKS, BLOCKSIZE>>>(d_A, d_partial, M);
        reduce_final<<<1, 1>>>(d_partial, d_out, NUM_BLOCKS);
    }
    cudaDeviceSynchronize();

    // timed runs
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; i++) {
        reduce_gmem<<<NUM_BLOCKS, BLOCKSIZE>>>(d_A, d_partial, M);
        reduce_final<<<1, 1>>>(d_partial, d_out, NUM_BLOCKS);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_us = (ms / BENCH_ITERS) * 1000.0f;

    float bytes = (M + NUM_BLOCKS) * sizeof(float);  // reads + partial reads
    float gb_s  = (bytes / (avg_us * 1e-6f)) / 1e9f;

    printf("blocksize   : %d\n", BLOCKSIZE);
    printf("num blocks  : %d\n", NUM_BLOCKS);
    printf("avg time    : %.2f us\n", avg_us);
    printf("bandwidth   : %.2f GB/s\n", gb_s);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_partial); cudaFree(d_out);
    free(h_A);
    return 0;
}
