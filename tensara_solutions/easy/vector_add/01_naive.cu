__global__ void vecAdd(const float* A, const float* B, float* output, size_t n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n){
        output[i] = A[i] + B[i];
    }
}