__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n, size_t m){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n * m){
        output[idx] = fmax(0.0f, input[idx]);
    }
}