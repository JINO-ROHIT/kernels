#include <cuda_runtime.h>
#include <iostream>

#define N 4096
#define BLOCK_SIZE 256

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n * m){
        output[idx] = fmax(0.0f, input[idx]);
    }
}


int main() {

    size_t total_elems = N * N;
    size_t bytes = total_elems * sizeof(float);

    float *input_host  = (float*)malloc(bytes);
    float *output_host = (float*)malloc(bytes);

    for (size_t i = 0; i < total_elems; i++) {
        input_host[i] = (float)i - (total_elems / 2); // try to use some negatives
    }

    float *input_device, *output_device;
    cudaMalloc(&input_device, bytes);
    cudaMalloc(&output_device, bytes);

    cudaMemcpy(input_device, input_host, bytes, cudaMemcpyHostToDevice);

    int block_num = (total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(block_num);
    dim3 block(BLOCK_SIZE);

    relu_kernel<<<grid, block>>>(input_device, output_device, N, N);

    cudaMemcpy(output_host, output_device, bytes, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < 10; i++) {
    //     std::cout << input_host[i] << std::endl;
    //     std::cout << output_host[i] << std::endl;
    // }
}