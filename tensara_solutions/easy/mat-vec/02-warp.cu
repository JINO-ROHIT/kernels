#include <cuda_runtime.h>

// A = M x K
// B = K x 1
__global__ void matvec(const float* __restrict__ A, 
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        const int M,
                        const int K){
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= M) return;

    float tmp = 0.0f;
    for(int k = lane; k < K; k+=32){
        tmp += A[warp_id * K + k] * B[k];
    }
    tmp += __shfl_down_sync(0xffffffff, tmp, 16);
    tmp += __shfl_down_sync(0xffffffff, tmp,  8);
    tmp += __shfl_down_sync(0xffffffff, tmp,  4);
    tmp += __shfl_down_sync(0xffffffff, tmp,  2);
    tmp += __shfl_down_sync(0xffffffff, tmp,  1);

    if (lane == 0) C[warp_id] = tmp;

}
// Note: input_a, input_b, output_c are device pointers
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    int num_threads = 1024;
    //int num_blocks = (m + num_threads - 1)/ num_threads;

    dim3 gridDim((m * 32 + num_threads - 1)/ num_threads);
    dim3 blockDim(num_threads); 

    matvec<<<gridDim, blockDim>>>(input_a, input_b, output_c, m, k);
}