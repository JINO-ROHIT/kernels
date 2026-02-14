#include <cute/tensor.hpp>

using namespace cute;

template <int numElementsPerThread = 8>
__global__ void vector_add(half* __restrict__ x, half* __restrict__ y, half* __restrict__ z, const int num){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx > num / numElementsPerThread){
        return;
    };

    Tensor gx = make_tensor(make_gmem_ptr(x), make_shape(num));
    Tensor gy = make_tensor(make_gmem_ptr(y), make_shape(num));
    Tensor gz = make_tensor(make_gmem_ptr(z), make_shape(num));

    Tensor tile_gx = local_tile(gx, make_shape(Int<numElementsPerThread>()), idx);
    Tensor tile_gy = local_tile(gy, make_shape(Int<numElementsPerThread>()), idx);
    Tensor tile_gz = local_tile(gz, make_shape(Int<numElementsPerThread>()), idx);


    //create in registers
    Tensor tile_rx = make_tensor_like(tile_gx);
    Tensor tile_ry = make_tensor_like(tile_gy);
    Tensor tile_rz = make_tensor_like(tile_gz);

    // move the values to registers
    copy(tile_gx, tile_rx);
    copy(tile_gy, tile_ry);

    #pragma unroll
    for (int i = 0; i < size(tile_rx); i++)
    {
        tile_rz(i) = tile_rx(i) + tile_ry(i);
    }

    // move back to global
    copy(tile_rz, tile_gz);
}

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
    std::cout << "vector add cute version took " << elapsedTime / 100 << "ms." << std::endl;
}

