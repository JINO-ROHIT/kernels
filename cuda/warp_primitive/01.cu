#include <stdio.h>

__global__ void shuffle_down_example(int *out) {
    int lane = threadIdx.x % 32;  // lane ID within warp
    int val = lane;               // each thread starts with its lane ID

    unsigned mask = 0xffffffff; // this means all the threads are active

    // get value from thread shifted down by offset, each thread tries to read from lane_id + offset
    int shuffled = __shfl_down_sync(mask, val, 1);

    out[threadIdx.x] = shuffled;
}

int main() {
    const int N = 32;
    int h_out[N];
    int *d_out;

    cudaMalloc(&d_out, N * sizeof(int));

    // 1 block with 32 threads (1 warp)
    shuffle_down_example<<<1, N>>>(d_out);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("Thread %d got %d\n", i, h_out[i]);
    }

    cudaFree(d_out);
    return 0;
}