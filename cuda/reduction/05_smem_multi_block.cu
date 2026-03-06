#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCKDIM 512 // for smem since it needs to know at compile time

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void smem_reduce(const float* __restrict__ input, float* __restrict__ output){
    __shared__ float smem[BLOCKDIM];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tidx = threadIdx.x;

    smem[tidx] = input[idx] + input[idx + gridDim.x * blockDim.x]; // this accesses out of memory bounds btw but okay to show the logic, the other one is safer

    __syncthreads(); // make all threads first load


    for(unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2){
        if(threadIdx.x < stride){
            smem[tidx] += smem[tidx + stride];
        }
    }

    if(threadIdx.x == 0){
        atomicAdd(output, smem[threadIdx.x]);
    }
}

// another way to do this

// __global__ void smem_reduce(const float* __restrict__ input, float* __restrict__ output, int M){
//     __shared__ float smem[BLOCKDIM];

//     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int local_idx = threadIdx.x;

//     if(idx < M){
//         smem[local_idx] = input[idx];
//     }
//     else{
//         smem[local_idx] = 0.0f;
//     }

//     __syncthreads(); // wait till all threads load their elements into smem

//     for(unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2){
//         if(local_idx < stride){
//             smem[local_idx] += smem[local_idx + stride];
//         }
//         __syncthreads();
//     }

//     if(local_idx == 0){
//         atomicAdd(output, smem[local_idx]);
//     }

// }

int main() {
    const int M            = 1024;
    const int BLOCKSIZE    = M / 2;
    const int NUM_BLOCKS   = CEIL_DIV(M, BLOCKSIZE);

    // N blocks of (1024 / 2) threads

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