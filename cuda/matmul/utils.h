// a set of fns to mainly check for matrix correctness

#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>

#define EPSILON 1e-2

void cpu_matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void cpu_matmul_half(const __half* h_A, const __half* h_B, float* h_C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                float a_val = __half2float(h_A[i * k + l]);
                float b_val = __half2float(h_B[l * n + j]);
                sum += a_val * b_val;
            }
            h_C[i * n + j] = sum;
        }
    }
}

bool check_correctness(const float* cpu_C, const float* gpu_C, int size) {
    bool correct = true;
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    int num_errors = 0;
    
    for (int i = 0; i < size; i++) {
        float diff = fabs(cpu_C[i] - gpu_C[i]);
        avg_diff += diff;
        if (diff > max_diff) max_diff = diff;
        
        if (diff > EPSILON) {
            if (num_errors < 10) {
                printf("mismatch at index %d: CPU=%f, GPU=%f, diff=%f\n", 
                       i, cpu_C[i], gpu_C[i], diff);
            }
            num_errors++;
            correct = false;
        }
    }
    
    avg_diff /= size;
    printf("max difference: %f\n", max_diff);
    printf("avg difference: %f\n", avg_diff);
    return correct;
}