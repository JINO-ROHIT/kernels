// single block of total elements/2 threads(obviously wont work for larger sequence)
// also wont work of non power of 2 ( check the kernel and figure this out :)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))



__global__ void reduce_gmem(float* __restrict__ input, float* output){
    unsigned int i = 2 * threadIdx.x; // each thread loads 2 elements

    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
        if(threadIdx.x % stride == 0){
            input[i] += input[i + stride];
        }
        __syncthreads(); // wait for all threads to reach this stride first, then next stride etc, wont make sense outside for loop
    }

    if(threadIdx.x == 0){
        *output = input[threadIdx.x];
    }


}

int main() {
    const int M            = 1024;
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

    reduce_gmem<<<NUM_BLOCKS, BLOCKSIZE>>>(d_A, d_out);
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