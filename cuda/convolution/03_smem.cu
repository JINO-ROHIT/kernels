// for the tiling, remember we have hanging elements in the left and right side of the array, call them halo
// in the smem we need to handle them differently unlike gmem.

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define MASK_SIZE 3

__constant__ float MASK[MASK_SIZE]; 

__global__ void convolution(const float* __restrict__ input, float* __restrict__ output, const int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int middle = MASK_SIZE / 2;
    int start = idx - middle;

    float curr_sum = 0;

    for(int i = 0; i < MASK_SIZE; i++){
        if ((start + i >= 0) && (start + i < n)){
            curr_sum += MASK[i] * input[start + i]; 
        }
    }
    output[idx] = curr_sum;
}

int main(){
    int n = 10;
    float input[n], output[n], mask[MASK_SIZE];

    for(int i = 0; i < n; i++) input[i] = (float)i;
    for(int i = 0; i < MASK_SIZE; i++) mask[i] = (float)i;

    float* d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    // no cudaMalloc for mask constant memory is pre-allocated

    cudaMemcpyToSymbol(MASK, mask, MASK_SIZE * sizeof(float));  // special copy for __constant__
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((n + block.x - 1) / block.x);
    convolution<<<grid, block>>>(d_input, d_output, n); 

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("output: ");
    for(int i = 0; i < n; i++) printf("%.1f ", output[i]);
    printf("\n");

    cudaFree(d_input); cudaFree(d_output); 
    return 0;
}