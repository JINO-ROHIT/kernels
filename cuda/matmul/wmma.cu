// we are doing (MxK) @ (KxN) = (MxN)

#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "utils.h"

using namespace nvcuda;

// as long as the matrices are / by the WMMA, the kernel is correct
#define M 4096
#define N 4096
#define K 4096

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 4 * 2 = 8 warps = 256 threads per block
#define WARPS_X 4
#define WARPS_Y 2
#define WARP_SIZE 32

// we layout 4 * 2 warps for the 256 x 256 matrix
// each warp takes care of 16 x 16 tile
// so block tile across x = 16 * 4 = 64 and across y = 16 * 2 = 32

#define BLOCK_TILE_M (WMMA_M * WARPS_Y)   // 32 rows per block
#define BLOCK_TILE_N (WMMA_N * WARPS_X)   // 64 cols per block

__global__ void wmma_matmul(
    const __half* __restrict__ A,   
    const __half* __restrict__ B,  
    float* __restrict__ C, 
    int m, int n, int k
){
    // 256 threads divided into 8 warps laid out as 
    // 4 rows and 2 cols
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / WARPS_X;
    const int warp_col = warp_id % WARPS_X;

    // where should the output be in the C matrix
    const int cRow = blockIdx.y * BLOCK_TILE_M + warp_row * WMMA_M;
    const int cCol = blockIdx.x * BLOCK_TILE_N + warp_col * WMMA_N;

    if (cRow >= m || cCol >= n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accFrag;

    wmma::fill_fragment(accFrag, 0.0f);

    for(int kiter = 0; kiter < K; kiter += WMMA_K){
        const int aRow = cRow,   aCol = kiter;
        const int bRow = kiter,  bCol = cCol;

        if (aRow < m && aCol < k && bRow < k && bCol < n)
        {
            wmma::load_matrix_sync(aFrag, A + aRow * k + aCol, k);
            wmma::load_matrix_sync(bFrag, B + bRow * n + bCol, n);

            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }
    wmma::store_matrix_sync(C + cRow * n + cCol, accFrag, n, wmma::mem_row_major);
}

int main()
{
    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    __half* h_A      = (__half*)malloc(sizeA * sizeof(__half));
    __half* h_B      = (__half*)malloc(sizeB * sizeof(__half));
    float*  h_C      = (float*) malloc(sizeC * sizeof(float)); // to store gpu result
    float* h_C_cpu   = (float*) malloc(sizeC * sizeof(float)); // to store cpu result

    srand(42);
    for (size_t i = 0; i < sizeA; ++i) {
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_A[i] = __float2half(v);
    }
    for (size_t i = 0; i < sizeB; ++i) {
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_B[i] = __float2half(v);
    }

    __half* d_A;  __half* d_B;  float* d_C;
    cudaMalloc(&d_A, sizeA * sizeof(__half));
    cudaMalloc(&d_B, sizeB * sizeof(__half));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    cudaMemcpy(d_A, h_A, sizeA * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0,   sizeC * sizeof(float));


    dim3 blockDim(WARPS_X * WARPS_Y * WARP_SIZE);                // 256
    dim3 gridDim((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,          // ceil(N/64)
                 (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);         // ceil(M/32)


    // warm up
    wmma_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();


    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);

    cudaEventRecord(tStart);
    wmma_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, tStart, tStop);

    double flops  = 2.0 * M * N * K;                
    double tflops = flops / (ms * 1e9);     
    printf("kernel time : %.4f ms\n", ms);
    printf("throughput  : %.4f TFLOP/s\n\n", tflops);

    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // some cpu stuff(turn off for large matrices)
    // cpu_matmul_half(h_A, h_B, h_C_cpu, M, N, K);
    // bool correct = check_correctness(h_C_cpu, h_C, sizeC);
}