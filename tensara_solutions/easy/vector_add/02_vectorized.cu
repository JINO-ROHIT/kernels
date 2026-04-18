__global__ void vecAdd(const float* __restrict__ A, const float* __restrict__ B, 
                       float* __restrict__ output, size_t n){
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    
    if(i + 3 < n){
        float4 a = reinterpret_cast<const float4*>(A)[i/4];
        float4 b = reinterpret_cast<const float4*>(B)[i/4];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4*>(output)[i/4] = c;
    }
    // Handle remainder elements
    else if(i < n){
        for(int j = i; j < n && j < i + 4; j++){
            output[j] = A[j] + B[j];
        }
    }
}

extern "C" void solution(const float* d_input1, const float* d_input2, 
                         float* d_output, size_t n) {
    int blocksize = 1024;
    int numblocks = ((n + 3) / 4 + blocksize - 1) / blocksize;
    vecAdd<<<numblocks, blocksize>>>(d_input1, d_input2, d_output, n);
}