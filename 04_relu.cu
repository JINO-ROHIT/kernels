#include <cuda_runtime.h>
#include <iostream>

#define DEBUG 0 //use this to check correctness

__global__ void naive_relu(const float* __restrict__ input, float* __restrict__ output, const int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

int main(){
    const int N = 10000;
    size_t size = N * sizeof(float);

    float *d_input, *d_output;
    float *h_input, *h_output;

    if(DEBUG){
        h_input = new float[N]{-1.0f, 10.0f, 20.0f, -10.0f};
    }
    else{
        h_input = new float[N];
    }

    h_output = new float[N];

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, size, cudaMemcpyHostToDevice);

    int blocksize = 1024;
    int gridsize = (N + blocksize - 1) / blocksize;

    cudaEvent_t start, end;
    cudaEventCreate(&start), cudaEventCreate(&end);

    cudaEventRecord(start);
    naive_relu<<<gridsize, blocksize>>>(d_input, d_output, N);
    cudaEventRecord(end);

    cudaEventSynchronize(end);

    float ms1 = 0;
    cudaEventElapsedTime(&ms1, start, end);

    std::cout << "vector size: " << N << std::endl;
    std::cout << "naive relu time: " << ms1 << " ms" << std::endl;

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    if(DEBUG){
        for(int i = 0; i < 4; i++){
            std::cout << h_output[i] << std::endl;
        }
    }

    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output);
}
