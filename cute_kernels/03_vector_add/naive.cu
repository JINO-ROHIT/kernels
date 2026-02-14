#include <cute/tensor.hpp>

template <int numElementsPerThread = 8>
__global__ void vector_add(half* __restrict__ x, half* __restrict__ y, half* __restrict__ z, const int num){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int offset = idx * numElementsPerThread; // let each thread work on 8 elements, spawn lesser threads this way
    if(idx > num / numElementsPerThread){
        return;
    };
    half2* x_ptr = reinterpret_cast<half2*>(x + offset); 
    half2* y_ptr = reinterpret_cast<half2*>(y + offset); 
    half2* z_ptr = reinterpret_cast<half2*>(z + offset);
    
    #pragma unroll
    for(int i = 0; i < numElementsPerThread/2; i++){ 
        half2 x_val, y_val;
        x_val = __ldg(x_ptr++);
        y_val = __ldg(y_ptr++);
        *z_ptr++ = x_val + y_val;
    }

};


int main(){
    const int elementsPerThread = 8;
    const int totalElements = 1024 * 8192;

    cudaEvent_t start, end;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    half *hostX = (half *)malloc(totalElements * sizeof(half));
    half *hostY = (half *)malloc(totalElements * sizeof(half));
    half *hostZ = (half *)malloc(totalElements * sizeof(half));
    for (int i = 0; i < totalElements; i++){
        hostX[i] = 1;
        hostY[i] = 1;
        hostZ[i] = 0;
    }

    half *deviceX, *deviceY, *deviceZ;
    cudaMalloc(&deviceX, totalElements * sizeof(half));
    cudaMalloc(&deviceY, totalElements * sizeof(half));
    cudaMalloc(&deviceZ, totalElements * sizeof(half));
    cudaMemcpy(deviceX, hostX, totalElements * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceY, hostY, totalElements * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceZ, hostZ, totalElements * sizeof(half), cudaMemcpyHostToDevice);

    const int blockSize = 1024;
    const int gridSize = totalElements / (blockSize * elementsPerThread);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++){
        vector_add<elementsPerThread><<<gridSize, blockSize>>>(deviceZ, deviceX, deviceY, totalElements);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "vector add naive took " << elapsedTime / 100 << "ms." << std::endl;
}
