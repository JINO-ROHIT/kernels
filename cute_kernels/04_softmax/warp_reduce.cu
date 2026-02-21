// inspired from https://github.com/Dao-AILab/flash-attention

#include <iostream>
#include <stdint.h>

template<typename T>
struct SumOp {
__device__ inline T operator()(T const &x, T const &y) { return x + y; }
};

template <int tHREADS>
struct AllReduce{
    template <typename Operator>
    static __device__ inline float run(float x, Operator &op){
        constexpr int OFFSET = tHREADS / 2;
        x = op(x, __shfl_xor_sync(0xffffffff, x, OFFSET)); // set the mask to all 1 to activate all threads
        return AllReduce<OFFSET>::run(x, op);
    }
};


template <>
struct AllReduce<2> {
    template<typename Operator>
    static __device__ inline float run(float x, Operator &op) {
        x = op(x, __shfl_xor_sync(0xffffffff, x, 1));
        return x;
    }
};

__global__ void allreduce(
    float *input,
    float *out_sum,
    float *dbg_round1
) {
    // each thread takes one value, all the end all of them agree
    int tid = threadIdx.x;
    float val = input[tid];

    SumOp<float> sum_op;
    float sum_result = AllReduce<4>::run(val, sum_op);
    out_sum[tid] = sum_result;

    //__syncthreads(); dont even need this

    if (tid == 0) {
        printf("\n  [SUM]:\n");
        printf("    thread 0: %.1f\n", out_sum[0]);
        printf("    thread 1: %.1f\n", out_sum[1]);
        printf("    thread 2: %.1f\n", out_sum[2]);
        printf("    thread 3: %.1f\n", out_sum[3]);
    }
}

int main() {
    float h_input[4]  = {3.0f, 7.0f, 2.0f, 5.0f};
    float h_sum[4]    = {0};

    float *d_input, *d_sum, *d_round1;
    cudaMalloc(&d_input,  4 * sizeof(float));
    cudaMalloc(&d_sum,    4 * sizeof(float));
    cudaMalloc(&d_round1, 4 * sizeof(float));

    cudaMemcpy(d_input, h_input, 4 * sizeof(float), cudaMemcpyHostToDevice);

    printf("input: T0=%.1f  T1=%.1f  T2=%.1f  T3=%.1f\n\n",
           h_input[0], h_input[1], h_input[2], h_input[3]);

    allreduce<<<1, 4>>>(d_input, d_sum, d_round1);
    cudaDeviceSynchronize();

    cudaMemcpy(h_sum,  d_sum,  4 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_sum);   
    cudaFree(d_round1);
    return 0;
}