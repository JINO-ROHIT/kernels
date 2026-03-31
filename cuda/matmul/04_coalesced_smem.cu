// remember this is A = M x K and B = K x N
// from simons blog

#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// here we focus on 1d indexing
template <const int BLOCKSIZE>
__global__ void smem_coalescedkernel(int M, int N, int K,
                             const float *A, const float *B, float *C) {
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
    
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int thread_row = threadIdx.x / BLOCKSIZE; 
    const int thread_col = threadIdx.x % BLOCKSIZE;

    // move to starting positions
    A += block_row * BLOCKSIZE * K;// to advance rows we need to skip K elements
    B += block_col * BLOCKSIZE; // to move columns, we only need to +1
    C += block_row * BLOCKSIZE * N + block_col * BLOCKSIZE;
    
    float tmp = 0.0f;
    // we need to move across the k dimension loading tiles of size BLOCKSIZE into the shared mem
    for(int tile_idx = 0; tile_idx < K; tile_idx += BLOCKSIZE){
        As[thread_row * BLOCKSIZE + thread_col] = A[thread_row * K + thread_col];
        Bs[thread_row * BLOCKSIZE + thread_col] = B[thread_row * N + thread_col];

        __syncthreads();

        for(int t = 0; t < BLOCKSIZE; t++){
            tmp += As[thread_row * BLOCKSIZE + t] * Bs[t * BLOCKSIZE + thread_col];
        }

        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }
    // C is M x N
    C[thread_row * N + thread_col] = tmp;
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
    
    smem_coalescedkernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);    
    cudaDeviceSynchronize();

    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);

    cudaEventRecord(tStart);
    smem_coalescedkernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);    
    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, tStart, tStop);

    double flops  = 2.0 * M * N * K;                
    double tflops = flops / (ms * 1e9);     
    printf("kernel time : %.4f ms\n", ms);
    printf("throughput  : %.4f TFLOP/s\n\n", tflops);

}