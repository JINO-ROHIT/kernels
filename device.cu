// run this to check your device specifications

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace std;

int main(){
  int deviceCount = 0, device = 0;
  cudaDeviceProp deviceProp;

  cudaGetDeviceCount(&deviceCount);
  cudaGetDeviceProperties(&deviceProp, device);

  cout << "total CUDA device: " << deviceCount << endl;
  cout << "card: " << deviceProp.name << endl;
  cout << "CUDA compute capability: "
       << deviceProp.major << "." << deviceProp.minor << endl;
  cout << "SM count: "
     << deviceProp.multiProcessorCount << endl;
  cout << "total global memory: "
       << (float)deviceProp.totalGlobalMem / pow(1024.0, 3) << " GB" << endl;
  cout << "clock rate: " << deviceProp.clockRate * 1e-3f << " MHz" << endl;
  cout << "l2 cache size: " << deviceProp.l2CacheSize << endl;
  cout << "total constant memory: " << deviceProp.totalConstMem << endl;
  cout << "total shared memory per block: "
       << deviceProp.sharedMemPerBlock << endl;
  cout << "total registers per block: " << deviceProp.regsPerBlock << endl;
  cout << "warp size: " << deviceProp.warpSize << endl;
  cout << "max threads per SM: "
       << deviceProp.maxThreadsPerMultiProcessor << endl;
  cout << "max size of each dimension in a block: "
       << deviceProp.maxThreadsDim[0] << " x "
       << deviceProp.maxThreadsDim[1] << " x "
       << deviceProp.maxThreadsDim[2] << endl;
  cout << "max size of each dimension in a grid: "
       << deviceProp.maxGridSize[0] << " x "
       << deviceProp.maxGridSize[1] << " x "
       << deviceProp.maxGridSize[2] << endl;
}