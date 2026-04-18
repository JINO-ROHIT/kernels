#include <cuda_runtime.h>
#include <iostream>

#define N 1024 * 1024
#define BLOCK_SIZE 256

__global__ void reduce_v1(float* __restrict__ input, float* __restrict output){
    extern __shared__ float smem[]; // extern to be dynamic

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // global index
    unsigned int tid = threadIdx.x; // local to the block

    smem[tid] = input[i];
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride /= 2){ // here each elements next to each other
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

    int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *output_host = (float*)malloc((N / BLOCK_SIZE) * sizeof(float));
    float *output_device;
    cudaMalloc(&output_device, (N / BLOCK_SIZE) * sizeof(float));
    
    dim3 grid(N / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    reduce_v1<<<grid, block>>>(input_device, output_device);
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    //for(int i = 0; i < (N/BLOCK_SIZE); i++) std::cout << output_host[i] << std::endl;
    return 0;
}