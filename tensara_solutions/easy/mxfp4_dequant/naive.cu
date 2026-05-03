#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float decode_e2m1(uint8_t bits) {
    // 4 bit E2M1: [sign(1)][exp(2)][man(1)]
    if (bits == 0x0) return 0.0f;

    uint8_t sign     = (bits >> 3) & 0x1;
    uint8_t exp_bits = (bits >> 1) & 0x3;
    uint8_t man_bits =  bits       & 0x1;

    float value;
    if (exp_bits == 0) {
        // subnormal: 2^(-1) * man
        value = ldexpf((float)man_bits, -1);
    } else {
        // normal: 2^(exp-1) * (1 + man/2), bias = 1
        value = ldexpf(1.0f + (float)man_bits * 0.5f, (int)exp_bits - 1);
    }
    return sign ? -value : value;
}

__global__ void mxfp4_dequant_kernel(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ scale,
    float* __restrict__ out,
    int M, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= K) return;

    uint8_t packed = q[row * (K / 2) + col / 2];
    uint8_t nibble = (col % 2 == 0) ? (packed & 0xF) : (packed >> 4);

    float f_q = decode_e2m1(nibble);

    uint8_t s_val = scale[row * (K / 32) + col / 32];
    float f_scale = (s_val == 0xFF) ? 0.0f : ldexpf(1.0f, (int)s_val - 127);

    out[row * K + col] = f_q * f_scale;
}

extern "C" void solution(
    const uint8_t* q, const uint8_t* scale, float* out, size_t m, size_t k)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (k + threads.x - 1) / threads.x,
        (m + threads.y - 1) / threads.y);

    mxfp4_dequant_kernel<<<blocks, threads>>>(q, scale, out, (int)m, (int)k);
}