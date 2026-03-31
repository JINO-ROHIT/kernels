// remember this is A = M x K and B = K x N

#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

template <const int BLOCKSIZE>
__global__ void smem_kernel(int M, int N, int K,
                             const float *A, const float *B, float *C) {
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];
    
    const int block_row = blockIdx.y; const int block_col = blockIdx.x;
    const int thread_row = threadIdx.y; const int thread_col = threadIdx.x;
    
    // global row and column for this thread's output
    const int global_row = block_row * BLOCKSIZE + thread_row;
    const int global_col = block_col * BLOCKSIZE + thread_col;
    
    float tmp = 0.0f;
    // we need to move across the k dimension loading tiles of size BLOCKSIZE into the shared mem
    for(int tile_idx = 0; tile_idx < K / BLOCKSIZE; tile_idx += 1){
        // find the correct row of A, then the correct tile of A, and then move within the tile
        As[thread_row][thread_col] = A[global_row * K + tile_idx * BLOCKSIZE + thread_col];
        Bs[thread_row][thread_col] = B[(tile_idx * BLOCKSIZE + thread_row) * N + global_col];

        __syncthreads();

        for(int t = 0; t < BLOCKSIZE; t++){
            tmp += As[thread_row][t] * Bs[t][thread_col];
        }

        __syncthreads();
    }
    // C is M x N
    C[global_row * N + global_col] = tmp;
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
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);  // 16x16 = 256 threads
    
    smem_kernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);    
    cudaDeviceSynchronize();

    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);

    cudaEventRecord(tStart);
    smem_kernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);    
    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, tStart, tStop);

    double flops  = 2.0 * M * N * K;                
    double tflops = flops / (ms * 1e9);     
    printf("kernel time : %.4f ms\n", ms);
    printf("throughput  : %.4f TFLOP/s\n\n", tflops);

}