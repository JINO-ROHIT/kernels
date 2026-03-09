#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define TILE 10

__global__ void transpose(float *input, float *output, int length, int width) {
    __shared__ float smem[TILE][TILE];

    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    // load tiles into smem
    if(x < width && y < length){ 
        smem[y][x] = input[y * width + x]; 
    }
    
    __syncthreads();
    
    if(x < width && y < length){
        int output_idx = x * length + y;
        output[output_idx] = smem[y][x]; 
    }
    
}

void initMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)(i + 1);
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int rows = 20, cols = 20; // try larger sizes
    size_t bytes = rows * cols * sizeof(float);
    
    float *h_matrix = (float*)malloc(bytes);
    float *h_transposed = (float*)malloc(bytes);
    float *d_matrix, *d_transposed;
    
    cudaMalloc(&d_matrix, bytes);
    cudaMalloc(&d_transposed, bytes);
    
    initMatrix(h_matrix, rows, cols);
    
    // printf("original matrix:\n");
    // printMatrix(h_matrix, rows, cols);
    
    cudaMemcpy(d_matrix, h_matrix, bytes, cudaMemcpyHostToDevice);
    
    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose<<<grid, block>>>(d_matrix, d_transposed, rows, cols);
    
    cudaMemcpy(h_transposed, d_transposed, cols * rows * sizeof(float), cudaMemcpyDeviceToHost);
    
    // printf("\ntransposed matrix:\n");
    // printMatrix(h_transposed, cols, rows);

    
    cudaFree(d_matrix);
    cudaFree(d_transposed);
    free(h_matrix);
    free(h_transposed);
}
