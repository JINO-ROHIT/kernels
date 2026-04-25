#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

template <const int BLOCKSIZE>
__global__ void mma_ptx(int M, int N, int K,
                             const float *A, const float *B, float *C) {

    const int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (row < M && col < N) {
        asm(".reg .f32 f1, f2, f3;\n"
            "mov.f32 f1, 0.0;\n" // eq to float tmp = 0.0f;
        );
        for (int i = 0; i < K; ++i) {
            asm("ld.global.f32 f2, [%0];\n"
                "ld.global.f32 f3, [%1];\n"
                "fma.rn.f32 f1, f2, f3, f1;\n"
                :
                : "l"(&A[row * K + i]), "l"(&B[i * N + col]) // eq to tmp += A[row * K + i] * B[i * N + col];
            );
        }
        asm("st.global.f32 [%0], f1;\n" ::"l"(&C[row * N + col])); //eq to C[row * N + col] = tmp;
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
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);  // 16x16 = 256 threads
    
    mma_ptx<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);    
    cudaDeviceSynchronize();

    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);

    cudaEventRecord(tStart);
    mma_ptx<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);    
    cudaEventRecord(tStop);
    cudaEventSynchronize(tStop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, tStart, tStop);

    double flops  = 2.0 * M * N * K;                
    double tflops = flops / (ms * 1e9);     
    printf("kernel time : %.4f ms\n", ms);
    printf("throughput  : %.4f TFLOP/s\n\n", tflops);

}