#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMul_1d(const float* __restrict__ M, const float* __restrict__ N, float* __restrict__ O, const int width){
    // assume a matrix of size (width x width)

    //lets do 1d indexing (can you see why this is bad?)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx <  width * width){

        int row = idx / width;
        int col = idx % width;

        float val = 0.0f;
        for(int k = 0; k < width; k++){
            val += M[row * width + k] * N[k * width + col];
        }
        O[row * width + col] = val;
    }
    // then 2d indexing

}


int main() {
    const int width = 1024;
    size_t size = width * width * sizeof(float);

    float *h_M = (float*)malloc(size);
    float *h_N = (float*)malloc(size);
    float *h_O = (float*)malloc(size);

    for (int i = 0; i < width * width; ++i) {
        h_M[i] = 1.0f; 
        h_N[i] = 2.0f;
    }

    float *d_M, *d_N, *d_O;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_O, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * width + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_N, d_O, width);

    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_O, h_O, size, cudaMemcpyDeviceToHost);

    std::cout << "kernel Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "matrix Width: " << width << std::endl;


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_O);
    free(h_M); free(h_N); free(h_O);

    return 0;
}