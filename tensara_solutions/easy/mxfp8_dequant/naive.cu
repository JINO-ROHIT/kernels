#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float decode_e4m3(uint8_t bits) {
    uint8_t sign     = (bits >> 7) & 0x1;
    uint8_t exp_bits = (bits >> 3) & 0xF;
    uint8_t man_bits =  bits       & 0x7;

    float value;
    if (exp_bits == 0) {
        value = ldexpf((float)man_bits, -9);  
    } else {
        value = ldexpf(1.0f + (float)man_bits * 0.125f, (int)exp_bits - 7);
    }
    return sign ? -value : value;
}

__global__ void mxfp8_dequant_kernel(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ scale,
    float* __restrict__ out,
    int M, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= K) return;

    uint8_t q_val = q[row * K + col];
    float f_q = decode_e4m3(q_val);

    // decode E8M0 scale (1 per 32 elements along K)
    uint8_t s_val = scale[row * (K / 32) + col / 32];
    float f_scale = (s_val == 0xFF) ? 0.0f : ldexpf(1.0f, (int)s_val - 127);

    out[row * K + col] = f_q * f_scale;
}

extern "C" void solution(
    const uint8_t* q, const uint8_t* scale, float* out, size_t m, size_t k)
{
    dim3 threads(32, 32);
    dim3 blocks(
        (k + threads.x - 1) / threads.x,
        (m + threads.y - 1) / threads.y);

    mxfp8_dequant_kernel<<<blocks, threads>>>(q, scale, out, (int)m, (int)k);
}