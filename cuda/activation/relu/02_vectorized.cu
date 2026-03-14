// to-do need to handle non multiple of 4

#include <cuda_runtime.h>
#include <iostream>

#define N 4
#define BLOCK_SIZE 256

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, size_t vec_elems){

    const int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (tx < vec_elems) { // m / 4 since we use float 4 elements packed into one
        
        float4 v = __ldg(reinterpret_cast<const float4*>(input) + tx);
        
        v.x = fmaxf(v.x, 0.0f);
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);
        
        reinterpret_cast<float4*>(output)[tx] = v;
    }
}


int main() {

    size_t total_elems = N * N; // div by 4
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

    size_t vec_elems = total_elems / 4;  // number of float4 chunks
    int block_num = (vec_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(block_num);
    dim3 block(BLOCK_SIZE);

    relu_kernel<<<grid, block>>>(input_device, output_device, vec_elems);

    cudaMemcpy(output_host, output_device, bytes, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N * N; i++) {
    //     std::cout << input_host[i] << std::endl;
    //     std::cout << output_host[i] << std::endl;
    // }
}