#include <cuda_runtime.h>

// A = M x K
// B = K x 1

// the problem w this kernel is the uncoalesced access, thread 1 handle A[0 to K], then thread 2 handles A[k strided], its not A[0] A[1]
// we want the threads in warp to access consecutive elements

__global__ void matvec(const float* __restrict__ A, 
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        const int M,
                        const int K){
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    if(tidx < M){
        float tmp = 0.0f;
        for(int k = 0; k < K; k++){
            tmp += A[tidx * K + k] * B[k];
        }
        C[tidx] = tmp; 
    }
    

}
// Note: input_a, input_b, output_c are device pointers
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    int num_threads = 256;
    //int num_blocks = (m + num_threads - 1)/ num_threads;

    dim3 gridDim((m + num_threads - 1)/ num_threads);
    dim3 blockDim(num_threads); 

    matvec<<<gridDim, blockDim>>>(input_a, input_b, output_c, m, k);
}