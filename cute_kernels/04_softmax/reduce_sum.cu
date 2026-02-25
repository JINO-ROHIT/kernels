// inspired from https://github.com/Dao-AILab/flash-attention

#include <cute/tensor.hpp>

#include <iostream>
#include <stdint.h>

using namespace cute;

template<typename T>
struct SumOp {
__device__ inline T operator()(T const &x, T const &y) { return x + y; }
};

template <int THREADS>
struct AllReduce{
    template <typename Operator>
    static __device__ inline float run(float x, Operator &op){
        constexpr int OFFSET = THREADS / 2;
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


template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0)); // first time step, assign first element to summary
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = AllReduce<4>::run(src(i), op);
    }
}


__global__ void sum_reduce(float* input_ptr, float* output_ptr){
    int tid = threadIdx.x; // can be 0, 1, 2, 3

    float* input = input_ptr + tid * 8; // first row , second row, third row, fourth row

    Tensor tensor = make_tensor(make_gmem_ptr(input),
                                make_layout(
            make_shape(Int<2>{}, Int<4>{}),
            make_stride(Int<4>{}, Int<1>{})
        ));
    Tensor summary = make_tensor<float>(Shape<Int<2>>{}); // in register 1d tensor

    if(tid == 0){
        print_tensor(tensor);
        print_tensor(summary);
    }

    SumOp<float> sum;
    thread_reduce_(tensor, summary, sum);

    if(tid == 0){
        printf("Thread %d: summary = [%.0f, %.0f]\n",
           tid, summary(0), summary(1));
    }

    if(tid == 1){
        printf("Thread %d: summary = [%.0f, %.0f]\n",
           tid, summary(0), summary(1));
    }

    if(tid == 2){
        printf("Thread %d: summary = [%.0f, %.0f]\n",
           tid, summary(0), summary(1));
    }

    if(tid == 3){
        printf("Thread %d: summary = [%.0f, %.0f]\n",
           tid, summary(0), summary(1));
    }

    quad_allreduce_(summary, summary, sum);

    print("after quad summary\n");

    if(tid == 0){
        printf("Thread %d: summary= [%.0f, %.0f]\n",
           tid, summary(0), summary(1));
    }

    if(tid == 1){
        printf("Thread %d: summary= [%.0f, %.0f]\n",
           tid, summary(0), summary(1));
    }

    if(tid == 2){
        printf("Thread %d: summary= [%.0f, %.0f]\n",
           tid, summary(0), summary(1));
    }

    if(tid == 3){
        printf("Thread %d: summary= [%.0f, %.0f]\n",
           tid, summary(0), summary(1));
    }

};

int main(){
    float h_input[32] = { // 8 x 4
        1,2,3,4,  1,1,1,1,
        5,6,7,8,  2,2,2,2,
        1,1,1,1,  3,3,3,3,
        2,2,2,2,  4,4,4,4
    };
    float h_output[8] = {0};

    float *d_input, *d_output;
    
    cudaMalloc(&d_input, 32 * sizeof(float));
    cudaMalloc(&d_output, 8 * sizeof(float));

    cudaMemcpy(d_input, h_input, 32 *sizeof(float), cudaMemcpyHostToDevice);

    sum_reduce<<<1, 4>>>(d_input, d_output); // 4 threads
    cudaDeviceSynchronize();

}