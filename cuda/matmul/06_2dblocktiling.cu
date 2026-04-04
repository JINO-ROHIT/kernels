// remember this is A = M x K and B = K x N
// from simons blog

#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// some thought process
// M=N=K=8, BM=BN=BK=4, TM=TN=2
// each block computes a 4×4 tile of C, 
// and each thread computes a 2×2 sub-tile. 
// this gives us (4×4)/(2×2) = 4 threads per block.

// here we focus on 2d indexing
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void blocktiling_2d(int M, int N, int K,
                             const float *A, const float *B, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    const int total_results_blocktile = BM * BN;
    // A thread is responsible for calculating TM*TN elements in the blocktile
    const int num_threads_blocktile = (BM * BN) / (TM * TN);

    // BN/TN are the number of threads to span a column
    // each thread owns a TM×TN patch of the output tile. 
    // threadCol/threadRow identify which 2×2 patch within the 4×4 blocktile belongs to this thread.
    const int thread_row = threadIdx.x / (BN / TN); 
    const int thread_col = threadIdx.x % (BN / TN);

    // move to starting positions
    A += block_row * BM * K;// to advance rows we need to skip K elements
    B += block_col * BN; // to move columns, we only need to +1
    C += block_row * BM * N + block_col * BN;
    
    // calculate the indices this thread will load into smem
    const int innerColA = threadIdx.x % BK; 
    const int innerRowA = threadIdx.x / BK;

    const int innerColB = threadIdx.x % BN; 
    const int innerRowB = threadIdx.x / BN;

    const int stride_A = num_threads_blocktile / BK;
    const int stride_B = num_threads_blocktile / BN;

    float thread_results[TM * TN] = {0.0};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // we need to move across the shared k dimension 
    for(int tile_idx = 0; tile_idx < K; tile_idx += BK){
        // load A first
        for(int loadoffset = 0; loadoffset < BM; loadoffset += stride_A){
            //As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
            As[(innerRowA + loadoffset) * BK + innerColA] = A[(innerRowA + loadoffset) * K + innerColA];
        }

        for (uint loadOffset = 0; loadOffset < BK; loadOffset += stride_B) {
            // Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // for(int k = 0; k < BK; k++){
        //     float cached_B = Bs[k * BN + thread_col];
        //     for (uint resIdx = 0; resIdx < TM; resIdx++) {
        //         thread_results[resIdx] += As[(thread_row * TM + resIdx) * BK + k] * cached_B;
        //     }
        // }

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // move from smem into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(thread_row * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + thread_col * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    thread_results[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // for(int r = 0; r < TM; r++){
    //     C[(thread_row * TM + r) * N + thread_col] = thread_results[r];
    // }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(thread_row * TM + resIdxM) * N + thread_col * TN + resIdxN] = thread_results[resIdxM * TN + resIdxN];
        }
    }
    // C is M x N
};

int main() {
    const int M = 4096, N = 4096, K = 4096;
    const int BM = 64, BN = 64, BK = 8;
    const int TM = 8, TN = 8;

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

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1); // griddim.x is column
    dim3 blockDim((BM * BN) / (TM * TN));
    
    blocktiling_2d<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);    
    cudaDeviceSynchronize();

    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);

    cudaEventRecord(tStart);
    blocktiling_2d<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);    
    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, tStart, tStop);

    double flops  = 2.0 * M * N * K;                
    double tflops = flops / (ms * 1e9);     
    printf("kernel time : %.4f ms\n", ms);
    printf("throughput  : %.4f TFLOP/s\n\n", tflops);

}