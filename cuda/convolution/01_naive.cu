#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define MASK 3

// each thread computes N mask elements 

__global__ void convolution(const float* __restrict__ input, const float* __restrict__ mask, float* __restrict__ output, const int mask_size, const int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int middle = mask_size / 2;
    int start = idx - middle; 

    float curr_sum = 0; 

    for(int i = 0; i < mask_size; i++){
        if ((start + i >= 0) && (start + i < n)) { 
            curr_sum += mask[i] * input[start + i];
        }
    }
    output[idx] = curr_sum;
}

int main(){
    int n = 10;
    float input[n], output[n], mask[MASK];

    for(int i = 0; i < n; i++) input[i] = (float)i;
    for(int i = 0; i < MASK; i++) mask[i] = (float)i;

    float* d_input, *d_output, *d_mask;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMalloc(&d_mask, MASK * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, MASK * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((n + block.x - 1) / block.x);
    convolution<<<grid, block>>>(d_input, d_mask, d_output, MASK, n); 

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("output: ");
    for(int i = 0; i < n; i++) printf("%.1f ", output[i]);
    printf("\n");

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_mask);
    return 0;
}