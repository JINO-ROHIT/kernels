#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

template <const int BLOCKSIZE>
__global__ void naive_kernel(int M, int N, int K,
                             const float *A, const float *B, float *C) {

    const int row = blockIdx.x * BLOCKSIZE + threadIdx.y;
    const int col = blockIdx.y * BLOCKSIZE + threadIdx.x;

    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = tmp;
    }
}

int main() {
    const int M = 4096, N = 4096, K = 4096;
    const int BLOCKSIZE = 16;
    
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C; 
    

    h_A = (float*)malloc(M * K * sizeof(float));
    h_B = (float*)malloc(K * N * sizeof(float));
    h_C = (float*)malloc(M * N * sizeof(float));
    

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;
    

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE), 1);
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);  // 32x32 = 1024 threads
    
    naive_kernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}


// total SM = 34
// each SM has 1536 threads
// Register per SM = 65536


// kernel
// 1 block = 16 * 16 = 256 threads
// 40 registers per thread

// total register in a block = 256 * 40 = 10240

// thread limit = 1536 / 256 = 6 blocks
// register limit = 65536 / 10240 = 6 blocks

// total blocks launched = (4096 * 4096) / (16 * 16) = 65536 blocks
// how many blocks active = 34 * 6 = 204 blocks

// waves needed = 321.25 (matches the profiler)