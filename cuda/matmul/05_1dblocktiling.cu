// remember this is A = M x K and B = K x N
// from simons blog

#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// here we focus on 1d indexing
template <const int BM, const int BN, const int BK, const int TM>
__global__ void blocktiling_1d(int M, int N, int K,
                             const float *A, const float *B, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int thread_row = threadIdx.x / BN; 
    const int thread_col = threadIdx.x % BN;

    // move to starting positions
    A += block_row * BM * K;// to advance rows we need to skip K elements
    B += block_col * BN; // to move columns, we only need to +1
    C += block_row * BM * N + block_col * BN;
    
    const int innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const int innerRowA = threadIdx.x / BK;
    const int innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
    const int innerRowB = threadIdx.x / BN;

    float thread_results[TM] = {0.0f};
    // we need to move across the k dimension loading tiles of size BLOCKSIZE into the shared mem
    for(int tile_idx = 0; tile_idx < K; tile_idx += BK){
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();

        for(int k = 0; k < BK; k++){
            float cached_B = Bs[k * BN + thread_col];
            for (uint resIdx = 0; resIdx < TM; resIdx++) {
                thread_results[resIdx] += As[(thread_row * TM + resIdx) * BK + k] * cached_B;
            }
        }

        __syncthreads();

        A += BK;
        B += BK * N;
    }

    for(int r = 0; r < TM; r++){
        C[(thread_row * TM + r) * N + thread_col] = thread_results[r];
    }
    // C is M x N
};

int main() {
    const int M = 4096, N = 4096, K = 4096;
    const int BM = 64, BN = 64, BK = 8;
    const int TM = 8;

    // TM = how many output cells in the same column one thread is responsible for.
    // With TM = 4, one thread handles C[0,5], C[1,5], C[2,5], C[3,5] - a vertical strip of 4.
    
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

    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN), 1);
    dim3 blockDim((BM * BN) / TM);  // 16x16 = 256 threads
    
    blocktiling_1d<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);    
    cudaDeviceSynchronize();

    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);

    cudaEventRecord(tStart);
    blocktiling_1d<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);    
    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, tStart, tStop);

    double flops  = 2.0 * M * N * K;                
    double tflops = flops / (ms * 1e9);     
    printf("kernel time : %.4f ms\n", ms);
    printf("throughput  : %.4f TFLOP/s\n\n", tflops);

}