#include <cuda_runtime.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void rgb(const unsigned char* __restrict__ input_image, unsigned char* __restrict__ output_image, const int width, const int height){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // use 1d indexing for now, later benchmark both

    if(idx < width * height){

        int offset = idx * 3; // assuming 3 channels

        unsigned char red_pixel = input_image[offset];
        unsigned char green_pixel = input_image[offset + 1];
        unsigned char blue_pixel = input_image[offset + 2];

        output_image[idx] = 0.21 * red_pixel + 0.71 * green_pixel + 0.07 * blue_pixel;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor rgb_to_gray(torch::Tensor img) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);

    const int threads = 256;
    const int blocks = cdiv(width * height, threads);

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    rgb<<<blocks, threads, 0, torch::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(), result.data_ptr<unsigned char>(), width, height);
    
    cudaDeviceSynchronize()
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}