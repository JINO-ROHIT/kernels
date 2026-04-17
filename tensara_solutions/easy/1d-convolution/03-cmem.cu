#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define MASK 100

constexpr int BLOCK_SIZE = 512;
__constant__ float cmem[MASK]; // you wanna put the mask size here

__global__ void convolution(const float* __restrict__ input, const float* __restrict__ mask, float* __restrict__ output, const int mask_size, const int n){
    extern __shared__ float smem[]; // use extern to allow for dynamic allocation

    const int mask_radius = (mask_size - 1) / 2;
    const int tile_size = blockDim.x + mask_size - 1;

    const int tile_idx = blockIdx.x * blockDim.x - mask_radius;
    for(int i = threadIdx.x; i < tile_size; i += blockDim.x){
        int local_idx = tile_idx + i;
        if (local_idx >= 0 && local_idx < n) {
            smem[i] = input[local_idx];
        } else {
            smem[i] = 0.0f;
        }
    }
    __syncthreads();

    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(output_idx < n){
        float tmp = 0.0f;
        #pragma unroll // gives a slight speedup
        for(int i = 0; i < mask_size; i++){
            tmp += smem[threadIdx.x + i] * cmem[i];
        }
        output[output_idx] = tmp;
    }
}

// host code
int main(){
    int n = 100000;
    float input[n], output[n], mask[MASK];

    for(int i = 0; i < n; i++) input[i] = (float)i;
    for(int i = 0; i < MASK; i++) mask[i] = (float)i;

    float* d_input, *d_output, *d_mask;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMalloc(&d_mask, MASK * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, MASK * sizeof(float), cudaMemcpyHostToDevice);

    // dim3 block(512);
    // dim3 grid((n + block.x - 1) / block.x);

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block = BLOCK_SIZE;
    size_t shared_mem_size = (BLOCK_SIZE + MASK - 1) * sizeof(float);

    // why we need block size + mask - 1
    // BLOCK_SIZE = 4 (Threads 0, 1, 2, 3) MASK = 3 
    // each thread i needs to access indices [i-1, i, i+1].
    // thread 0 needs [-1, 0, 1]
    // thread 1 needs [0, 1, 2]
    // thread 2 needs [1, 2, 3]
    // thread 3 needs [2, 3, 4]
    // If you look at the total set of unique indices needed for this block to finish its work, it is {-1, 0, 1, 2, 3, 4\} ie 6, thats the formula

    // for constant memory
    cudaMemcpyToSymbol(cmem, d_mask, MASK * sizeof(float));

    printf("launching with %d blocks and %d threads and %d smem size\n", grid, block, shared_mem_size);
    convolution<<<grid, block, shared_mem_size>>>(d_input, d_mask, d_output, MASK, n); 

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("output: ");
    // for(int i = 0; i < n; i++) printf("%.1f ", output[i]);
    // printf("\n");

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_mask);
    return 0;
}