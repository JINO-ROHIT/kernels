#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void transpose(float *input, float *output, int length, int width) {
    int x = threadIdx.x; // col
    int y = threadIdx.y; // row
    
    if(x < width && y < length){
        int input_idx = y * width + x; 
        int output_idx = x * length + y;
        output[output_idx] = input[input_idx]; 
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
    int rows = 4, cols = 3;
    size_t bytes = rows * cols * sizeof(float);
    
    float *h_matrix = (float*)malloc(bytes);
    float *h_transposed = (float*)malloc(bytes);
    float *d_matrix, *d_transposed;
    
    cudaMalloc(&d_matrix, bytes);
    cudaMalloc(&d_transposed, bytes);
    
    initMatrix(h_matrix, rows, cols);
    
    printf("original matrix:\n");
    printMatrix(h_matrix, rows, cols);
    
    cudaMemcpy(d_matrix, h_matrix, bytes, cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    transpose<<<1, block>>>(d_matrix, d_transposed, rows, cols);
    
    cudaMemcpy(h_transposed, d_transposed, cols * rows * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("\ntransposed matrix:\n");
    printMatrix(h_transposed, cols, rows);
    
    cudaFree(d_matrix);
    cudaFree(d_transposed);
    free(h_matrix);
    free(h_transposed);
}
