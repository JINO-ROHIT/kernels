// mma.m16n8k16 performs matrix mul on m x k A(16 x 16) and k x n B(16 x 8) = m x n = C(16 x 8) 

// some information to work out
// A = (16 x 16) total elements = 256 fp16 --> 128 fp32
// this 128 32 bit is divided across the 32 threads(1 warp), so 1 thread will handle 128/32 = 4 32 bit elements or 8 fp16 elements

// for B --> 64 32 bit
// 1 thread will handle 2 32 bit element

#include <stdio.h>
#include <stdlib.h>
#include <mma.h>
#include <cuda_fp16.h>

// you cant pick anything, has to be a multiple of 16 and 8 since one warp handle 16 x 8
// #define BM 32
// #define BN 32
// #define WARP_NUM (BM * BN) / (16 * 8)

#define M 4096
#define N 4096
#define K 4096
#define WARP_SIZE 32

template <const int BM, const int BN>
__global__ void mma_m16n8k16_ptx(half* A, half* B, float* C) {
    //const int warp_M = BM / 16;
    const int warp_N = BN / 8;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int warp_row = warp_id / warp_N;
    const int warp_col = warp_id % warp_N;

    const int A_THREAD_ROW = lane_id / 4;
    const int A_THREAD_COL = lane_id % 4;

    const int B_THREAD_ROW = lane_id / 4;
    const int B_THREAD_COL = lane_id % 4;

    const int C_THREAD_ROW = lane_id / 4;
    const int C_THREAD_COL = lane_id % 4;

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};   
    half a[8];    
    half b[4];   
    half2 ra[4];  
    half2 rb[2];  

    //where in C to write the results
    const int WARP_ROW_OFFSET = blockIdx.y * BM + warp_row * 16; // in which block and which warp
    const int WARP_COL_OFFSET = blockIdx.x * BN + warp_col * 8;

    for (int i = 0; i < K; i += 16) {
        a[0] = A[(WARP_ROW_OFFSET + A_THREAD_ROW) * K + (A_THREAD_COL*2 + i)];
        a[1] = A[(WARP_ROW_OFFSET + A_THREAD_ROW) * K + (A_THREAD_COL*2 + i + 1)];

        a[2] = A[(WARP_ROW_OFFSET + A_THREAD_ROW + 8) * K + (A_THREAD_COL*2 + i)];
        a[3] = A[(WARP_ROW_OFFSET + A_THREAD_ROW + 8) * K + (A_THREAD_COL*2 + i + 1)];

        a[4] = A[(WARP_ROW_OFFSET + A_THREAD_ROW) * K + (A_THREAD_COL*2 + i + 8)];
        a[5] = A[(WARP_ROW_OFFSET + A_THREAD_ROW) * K + (A_THREAD_COL*2 + i + 9)];

        a[6] = A[(WARP_ROW_OFFSET + A_THREAD_ROW + 8) * K + (A_THREAD_COL*2 + i + 8)];
        a[7] = A[(WARP_ROW_OFFSET + A_THREAD_ROW + 8) * K + (A_THREAD_COL*2 + i + 9)];

        b[0] = B[(i + B_THREAD_ROW*2)     * N + (WARP_COL_OFFSET + B_THREAD_COL)];
        b[1] = B[(i + B_THREAD_ROW*2 + 1) * N + (WARP_COL_OFFSET + B_THREAD_COL)];
        b[2] = B[(i + B_THREAD_ROW*2 + 8) * N + (WARP_COL_OFFSET + B_THREAD_COL)];
        b[3] = B[(i + B_THREAD_ROW*2 + 9) * N + (WARP_COL_OFFSET + B_THREAD_COL)];

        // pack them into a single half2
        ra[0] = __halves2half2(a[0], a[1]);
        ra[1] = __halves2half2(a[2], a[3]);
        ra[2] = __halves2half2(a[4], a[5]);
        ra[3] = __halves2half2(a[6], a[7]);
        
        rb[0] = __halves2half2(b[0], b[1]);
        rb[1] = __halves2half2(b[2], b[3]);

        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(*(reinterpret_cast<int*>(&ra[0]))),
          "r"(*(reinterpret_cast<int*>(&ra[1]))),
          "r"(*(reinterpret_cast<int*>(&ra[2]))),
          "r"(*(reinterpret_cast<int*>(&ra[3]))),
          "r"(*(reinterpret_cast<int*>(&rb[0]))),
          "r"(*(reinterpret_cast<int*>(&rb[1]))));
    }

    C[(WARP_ROW_OFFSET + C_THREAD_ROW)     * N + (WARP_COL_OFFSET + C_THREAD_COL*2)]     = c[0];
    C[(WARP_ROW_OFFSET + C_THREAD_ROW)     * N + (WARP_COL_OFFSET + C_THREAD_COL*2 + 1)] = c[1];
    C[(WARP_ROW_OFFSET + C_THREAD_ROW + 8) * N + (WARP_COL_OFFSET + C_THREAD_COL*2)]     = c[2];
    C[(WARP_ROW_OFFSET + C_THREAD_ROW + 8) * N + (WARP_COL_OFFSET + C_THREAD_COL*2 + 1)] = c[3];
}


int main() {
    const int BM = 32;
    const int BN = 32;
    const int WARP_NUM = (BM / 16) * (BN / 8);
    const int THREADS  = WARP_NUM * WARP_SIZE;

    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    half*  h_A = (half*) malloc(sizeA * sizeof(half));
    half*  h_B = (half*) malloc(sizeB * sizeof(half));
    float* h_C = (float*)malloc(sizeC * sizeof(float));

    srand(42);
    for (size_t i = 0; i < sizeA; i++)
        h_A[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    for (size_t i = 0; i < sizeB; i++)
        h_B[i] = __float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f);

    half*  d_A; half*  d_B; float* d_C;
    cudaMalloc(&d_A, sizeA * sizeof(half));
    cudaMalloc(&d_B, sizeB * sizeof(half));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    cudaMemcpy(d_A, h_A, sizeA * sizeof(half),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(half),  cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0,   sizeC * sizeof(float));

    dim3 block_dim(THREADS);
    dim3 grid_dim(N / BN, M / BM);

    //warmup
    mma_m16n8k16_ptx<BM, BN><<<grid_dim, block_dim>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);

    cudaEventRecord(tStart);
    mma_m16n8k16_ptx<BM, BN><<<grid_dim, block_dim>>>(d_A, d_B, d_C);
    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, tStart, tStop);

    double tflops = (2.0 * M * N * K) / (ms * 1e9);
    printf("kernel time : %.4f ms\n", ms);
    printf("throughput  : %.4f TFLOP/s\n", tflops);

    cudaEventDestroy(tStart);
    cudaEventDestroy(tStop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A);     free(h_B);     free(h_C);
    return 0;
}