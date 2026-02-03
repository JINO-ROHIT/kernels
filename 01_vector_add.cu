#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int DSIZE = 1000; 
const int BLOCK_SIZE = 1024; // basically the number of threads

__global__ void vecAdd(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int ds){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < ds){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *h_A, *h_B, *h_out;
    float *d_A, *d_B, *d_out;

    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_out = new float[DSIZE];

    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float) RAND_MAX;
        h_B[i] = rand() / (float) RAND_MAX;
    }

    cudaMalloc(&d_A, DSIZE * sizeof(float));
    cudaMalloc(&d_B, DSIZE * sizeof(float));
    cudaMalloc(&d_out, DSIZE * sizeof(float));

    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vecAdd<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_out, DSIZE);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);

    delete[] h_A;
    delete[] h_B;
    delete[] h_out;

    return 0;
}