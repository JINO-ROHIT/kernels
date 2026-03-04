#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKDIM 1024 // for smem since it needs to know at compile time

__global__ void smem_reduce(const float* __restrict__ input, float* __restrict__ output){
    __shared__ float smem[BLOCKDIM];

    unsigned int tidx = threadIdx.x;

    smem[tidx] = input[tidx] + input[tidx + blockDim.x];

    __syncthreads(); // make all threads first do the first reduction in smem



    for(unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2){
        if(threadIdx.x < stride){
            smem[tidx] += smem[tidx + stride];
        }
    }

    if(threadIdx.x == 0){
        *output = smem[threadIdx.x];
    }

}

int main() {
    const int M            = 2048;
    const int BLOCKSIZE    = M / 2;
    const int NUM_BLOCKS   = 1;

    // 1 block of (1024 / 2) threads

    float *h_A = (float*)malloc(M * sizeof(float));

    float cpu_sum = 0.0f;
    for (int i = 0; i < M; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        cpu_sum += h_A[i];
    }

    float *d_A, *d_out;
    cudaMalloc(&d_A, M * sizeof(float));
    cudaMalloc(&d_out, 1 * sizeof(float));

    cudaMemcpy(d_A, h_A, M * sizeof(float), cudaMemcpyHostToDevice);

    smem_reduce<<<NUM_BLOCKS, BLOCKSIZE>>>(d_A, d_out);
    cudaDeviceSynchronize();

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
    printf("match   : %s\n",   fabsf(cpu_sum - gpu_sum) < 1e-1f ? "YES" : "NO");

    cudaFree(d_A);
    free(h_A);
    return 0;
}