#include "../util.h"

using namespace cute;

template <typename T, typename G2SCopy, typename S2RCopy, typename SmemLayout, int M, int N>
__global__ void copy_global_shm_register(const T *ptr)
{
    int idx = threadIdx.x;
    extern __shared__ T shm_data[];
    T *Ashm = shm_data;

    auto gA = make_tensor(make_gmem_ptr(ptr), make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayout{});

    auto rA = make_tensor_like(gA);

    G2SCopy g2s_tiled_copy;
    auto g2s_thr_copy = g2s_tiled_copy.get_slice(idx);
    auto tAgA = g2s_thr_copy.partition_S(gA);
    auto tAsA = g2s_thr_copy.partition_D(sA);
    cute::copy(g2s_tiled_copy, tAgA, tAsA);

    S2RCopy s2r_tiled_copy;
    auto s2r_thr_copy = s2r_tiled_copy.get_slice(idx);
    // auto stAsA = s2r_thr_copy.partition_S(sA); this will cause error
    auto stAsA = s2r_thr_copy.retile_S(tAsA);
    auto tArA = s2r_thr_copy.partition_D(rA);
    cute::copy(s2r_tiled_copy, stAsA, tArA);

    if (idx == 0)
    {
        PRINT("tAgA", tAgA.shape());
        PRINT("tAsA", tAsA.shape());
        PRINT("stAsA", stAsA.shape());
        PRINT("tArA", tArA.shape());
    }
}


int main()
{
    using T = cute::half_t; //fp16
    int device;
    cudaGetDevice(&device);

    int sharedMemPerBlock;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);

    printf("max shared memory per block for device %d is %d bytes\n", device, sharedMemPerBlock);

    // gpu has limited on chip memory split between shm and l1 cache
    // prefer more shm and less L1 cache
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    constexpr int M = 128;
    constexpr int N = 128;

    cudaEvent_t start, end;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>; // copy 128 bit at a time
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopy =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}), // rrange 128 threads in a 32 x 4 grid
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{})))); // Each thread handles 8 elements 
                                 // Because 8 halfs = 16 bytes = 128 bits (one copy atom)
                                 // Each thread loads one 128-bit chunk


    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}))));
    using SmemLayout = decltype(tile_to_shape(SmemLayoutAtom{},
                                              make_shape(Int<M>{}, Int<N>{})));

    static constexpr int shm_size = cute::cosize(SmemLayout{}) * sizeof(T);

    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopy =
        decltype(make_tiled_copy(s2r_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));

    T *ptr;
    cudaMalloc(&ptr, sizeof(T) * M * N);
    dim3 block(128);
    cudaEventRecord(start);
    int count = 100;
    for (int i = 0; i < count; ++i)
    {
        copy_global_shm_register<T, G2SCopy, S2RCopy, SmemLayout, M, N><<<1, block, shm_size>>>(ptr);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "copy_global_shm_register took " << elapsedTime / count << "ms." << std::endl;
}