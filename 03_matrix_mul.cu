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
}


__global__ void matrixMul_2d(const float* __restrict__ M, const float* __restrict__ N, float* __restrict__ O, const int width){
    // assume a matrix of size (width x width)

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if( (row <  width)  && (col < width) ){

        float val = 0.0f;
        for(int k = 0; k < width; k++){
            val += M[row * width + k] * N[k * width + col];
        }
        O[row * width + col] = val;
    }
}


int main() {
    const int width = 8192;
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

    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1); cudaEventCreate(&stop1);
    cudaEventCreate(&start2); cudaEventCreate(&stop2);

    int threads1d = 256;
    int blocks1d = (width * width + threads1d - 1) / threads1d;

    cudaEventRecord(start1);
    matrixMul_1d<<<blocks1d, threads1d>>>(d_M, d_N, d_O, width);
    cudaEventRecord(stop1);

    dim3 threads2d(16, 16); 
    dim3 blocks2d((width + threads2d.x - 1) / threads2d.x, 
                  (width + threads2d.y - 1) / threads2d.y);

    cudaEventRecord(start2);
    matrixMul_2d<<<blocks2d, threads2d>>>(d_M, d_N, d_O, width);
    cudaEventRecord(stop2);

    cudaEventSynchronize(stop1);
    cudaEventSynchronize(stop2);

    float ms1 = 0, ms2 = 0;
    cudaEventElapsedTime(&ms1, start1, stop1);
    cudaEventElapsedTime(&ms2, start2, stop2);

    std::cout << "matrix width: " << width << std::endl;
    std::cout << "1D Kernel Time: " << ms1 << " ms" << std::endl;
    std::cout << "2D Kernel Time: " << ms2 << " ms" << std::endl;

    cudaEventDestroy(start1); cudaEventDestroy(stop1);
    cudaEventDestroy(start2); cudaEventDestroy(stop2);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_O);
    free(h_M); free(h_N); free(h_O);

    return 0;
}