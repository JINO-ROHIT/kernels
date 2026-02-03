### writing your first kernel

host - is the CPU from where you call the GPU kernel
device - is where you execute the CUDA instruction


```
void vecAdd(float* A, float* B, float* C, int n) {
int size = n* sizeof(float);
float *d_A *d_B, *d_C;
// Part 1: Allocate device memory for A, B, and C
// Copy A and B to device memory
...
// Part 2: Call kernel – to launch a grid of threads
// to perform the actual vector addition
...
}
// Part 3: Copy C from the device memory
// Free device vectors
...
```

step 1 - before you can invoke the CUDA kernel, you need to first allocate some memory on device.

similarly like malloc in C, you have cudaMalloc which has two params -
1. address of the pointer to the allocated object
2. size to be allocated

The address of the pointer variable should be cast to (void) because the function expects a generic pointer; the
memory allocation function is a generic function that is not restricted to any particular type of objects.

a short example

```
float* A_d;
int size= n * sizeof(float);
cudaMalloc((void**)&A_d, size);


cudaFree(A_d); # at the end free up memory
```

once from the host you allocate space in the device, you now need to move the data from host to device using cudaMemcpy.

The cudaMemcpy function takes four parameters. 
 - The first parameter is a pointer to the destination location for the data object to be
copied.
 - The second parameter points to the source location. 
 - The third parameter specifies the number of bytes to be copied. 
 - The fourth parameter indicates the types of memory involved in the copy: from host to host, from host to device,
from device to host, and from device to device. For example, the memory copy
function can be used to copy data from one location in the device global memory
to another location in the device global memory.


a more complete example

```
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
int size = n * sizeof(float);
float *A_d, *B_d, *C_d;
cudaMalloc((void **) &A_d, size);
cudaMalloc((void **) &B_d, size);
cudaMalloc((void **) &C_d, size);
cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
// Kernel invocation code 
...
cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
}
cudaFree(A_d);
cudaFree(B_d);
cud
```