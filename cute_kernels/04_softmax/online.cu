#include <cute/tensor.hpp>
#include <cute/layout.hpp>

#include <cmath>
#include <iostream>
#include <random>

using namespace cute;


// lets assume 32k vocab

template <int vocabSize = 32000>
__global__ void online_softmax(const float* __restrict__ vocab, float* __restrict__ probs, const int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        Tensor global_vocab = make_tensor(make_gmem_ptr(vocab + idx * vocabSize), make_shape(vocabSize)); // the entire row
        Tensor global_probs = make_tensor(make_gmem_ptr(probs + idx * vocabSize), make_shape(vocabSize)); 

        float sum_exp = 0.0f;
        float max_val = -INFINITY;
        for (int i = 0; i < vocabSize; i++) {
            float curr_max = fmax(max_val, global_vocab(i));
            if(curr_max > max_val){
                float norm = expf(max_val - curr_max);
                max_val = curr_max;
                sum_exp *= norm;
            }

            float exp_val = expf(global_vocab(i) - max_val);

            //global_probs(i) = exp_val; think about why this is wrong :)
            sum_exp += exp_val;
        }

        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < vocabSize; i++) {
            float exp_val = expf(global_vocab(i) - max_val);
            global_probs(i) = exp_val * inv_sum;
        }  
    }

}


int main(){
    // const int vocabSize = 32000;
    // const int rows = 1000;

    const int vocabSize = 4;
    const int rows = 2;
    
    cudaEvent_t start, end;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //float* tensor = (float*)malloc(rows * vocabSize * sizeof(float));

    float tensor[rows * vocabSize] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, -1.0f, 0.0f, 2.0f
    };

    float* probs  = (float*)malloc(rows * vocabSize * sizeof(float));

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<float> dis(-10.0f, 10.0f); // Typical logit range
    
    // for(int i = 0; i < rows * vocabSize; i++){
    //     tensor[i] = dis(gen);
    // }

    float* device_tensor, *device_probs;
    cudaMalloc(&device_tensor, rows * vocabSize * sizeof(float));
    cudaMalloc(&device_probs,  rows * vocabSize * sizeof(float));

    cudaMemcpy(device_probs, probs, rows * vocabSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_tensor, tensor, rows * vocabSize * sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = rows;
    const int gridSize = 1;
    
    std::cout << "launching with " << gridSize << " blocks " << blockSize << " threads" << std::endl;
    cudaEventRecord(start);
    online_softmax<vocabSize><<<gridSize, blockSize>>>(device_tensor, device_probs, rows);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "online softmax took " << elapsedTime << "ms." << std::endl;

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaMemcpy(probs, device_probs, rows * vocabSize * sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < rows * vocabSize; i++){
    //     std::cout << tensor[i] << std::endl;
    // }

    for(int i = 0; i < 10; i++){ // results for the first row
        std::cout << probs[i] << std::endl;
    }
}