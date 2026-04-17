#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define MASK 100

__global__ void convolution(const float* __restrict__ input, const float* __restrict__ mask, float* __restrict__ output, const int mask_size, const int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    const int middle = mask_size / 2;
    int start = idx - middle; // align to the center

    float s = 0.0f;
    for(int i = 0; i < mask_size; i++){
        if( (start + i >= 0) && (start + i < n)){
            s += input[start + i] * mask[i];
        }
    }
    output[idx] = s;
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

    dim3 block(512);
    dim3 grid((n + block.x - 1) / block.x);

    printf("launching with %d blocks and %d threads\n", grid.x, block.x);
    convolution<<<grid, block>>>(d_input, d_mask, d_output, MASK, n); 

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("output: ");
    // for(int i = 0; i < n; i++) printf("%.1f ", output[i]);
    // printf("\n");

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_mask);
    return 0;
}