#include <cuda_runtime.h>
#include <iostream>

#define N 1024 * 1024
#define BLOCK_SIZE 256

__global__ void reduce_v1(float* __restrict__ input, float* __restrict output){
    extern __shared__ float smem[]; // extern to be dynamic

    unsigned int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x; // 2 * blockidx
    unsigned int tid = threadIdx.x; // local to the block

    smem[tid] = input[i] + input[i + blockDim.x]; // each block handling 2 * block_dim elements
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride /= 2){ 
        if(tid < stride){
            smem[tid] += smem[tid + stride];
        }
        __syncthreads(); 
    }

    if(tid == 0){
        output[blockIdx.x] = smem[0]; 
    }
}

int main() {
    float *input_host = (float*)malloc(N*sizeof(float));
    float *input_device;
    cudaMalloc(&input_device, N*sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 2.0;
    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2; // we need to launch half the blocks
    float *output_host = (float*)malloc(block_num * sizeof(float));
    float *output_device;
    cudaMalloc(&output_device, block_num * sizeof(float));
    
    dim3 grid(block_num); // we pass block num now
    dim3 block(BLOCK_SIZE);
    reduce_v1<<<grid, block>>>(input_device, output_device);
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    //for(int i = 0; i < block_num; i++) std::cout << output_host[i] << std::endl;
    return 0;
}