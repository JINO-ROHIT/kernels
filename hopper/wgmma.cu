#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

using bf16 = __nv_bfloat16;

static constexpr int M = 64;
static constexpr int N = 8;
static constexpr int K = 16;
static constexpr int NUM_THREADS = 128;

__device__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFF) >> 4;
}

// Minimal descriptor for a shared-memory matrix used by WGMMA.
//
// start_addr: shared-memory byte address, encoded in 16-byte units
// leading_off: distance from one row/column to the next major element
// stride_off: distance between repeated 8x8 chunks
__device__ uint64_t make_smem_desc(bf16* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    uint64_t desc = 0;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode(uint64_t(16)) << 16;
    desc |= matrix_descriptor_encode(uint64_t(1024)) << 32;
    return desc;
}

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7);
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// compute a 64x8 output tile
// d[4] is this thread’s piece of the output accumulator. The full WGMMA output is 64 x 8 = 512 floats. A warp group has 128 threads:
// 512 output floats / 128 threads = 4 floats per thread
// So each thread owns 4 accumulator registers.
template <int ScaleD>
__device__ void wgmma_m64n8k16(float d[4], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(sA);
    uint64_t desc_b = make_smem_desc(sB);

    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, "
        "%4, "
        "%5, "
        "p, 1, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(ScaleD)));
}

__global__ void wgmma_demo_kernel(float* out) {
    __shared__ alignas(128) bf16 sA[M * K];
    __shared__ alignas(128) bf16 sB[K * N];

    for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
        sA[i] = __float2bfloat16(1.0f);
    }

    for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
        sB[i] = __float2bfloat16(1.0f);
    }

    __syncthreads();

    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    warpgroup_arrive();
    wgmma_m64n8k16<0>(d, sA, sB);  // ScaleD=0 means D = A * B
    warpgroup_commit_batch();
    warpgroup_wait<0>();

    // Every output element should be 16 because A and B are filled with ones
    // and K=16. This stores each thread's four accumulator registers, not a
    // nicely laid out C matrix.
    int base = threadIdx.x * 4;
    out[base + 0] = d[0];
    out[base + 1] = d[1];
    out[base + 2] = d[2];
    out[base + 3] = d[3];
}

int main() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    if (props.major < 9) {
        fprintf(stderr, "This demo needs a Hopper or newer GPU, got sm_%d%d\n",
                props.major, props.minor);
        return EXIT_FAILURE;
    }

    float* out = nullptr;
    cudaMallocManaged(&out, NUM_THREADS * 4 * sizeof(float));

    wgmma_demo_kernel<<<1, NUM_THREADS>>>(out);
    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_THREADS * 4; ++i) {
        if (out[i] != 16.0f) {
            fprintf(stderr, "mismatch at accumulator %d: got %f\n", i, out[i]);
            cudaFree(out);
            return EXIT_FAILURE;
        }
    }

    printf("passed: one warp group computed m64n8k16 bf16*bf16 -> f32\n");

    cudaFree(out);
    return EXIT_SUCCESS;
}
