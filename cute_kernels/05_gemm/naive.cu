// we are doing a (MxK) x (KxN) = (MxN)
#include <cute/tensor.hpp>

using namespace cute;

template <typename T, int Bm, int Bn, int Bk>
__global__ void gemm(const T* a_ptr, const T* b_ptr, T* c_ptr, int M, int N, int K){
    Tensor A = make_tensor(make_gmem_ptr(a_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(b_ptr), make_shape(K, N), make_stride(N, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(c_ptr), make_shape(M, N), make_stride(N, Int<1>{}));


    int ix = threadIdx.x;
    int iy = threadIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<Bm>{}, Int<Bk>{}), make_coord(iy, _)); // (BM, BK, num tiles along K) --> (16, 16, 128/16)
    Tensor gB = local_tile(B, make_tile(Int<Bn>{}, Int<Bk>{}), make_coord(ix, _)); // (BN, BK, num tiles along K) --> (16, 16, 8)
    Tensor gC = local_tile(C, make_tile(Int<Bm>{}, Int<Bn>{}), make_coord(iy, ix)); // (BM, BN) --> (16, 16)

    // if(ix == 1 && iy == 1){
    //     printf("A: shape = (%d, %d)   stride = (%d, %d)\n",
    //         int(size<0>(A)), int(size<1>(A)),
    //         int(stride<0>(A)), int(stride<1>(A)));
    //     print("tile specs:\n");
    //     print(gA.layout());
    //     print(gB.layout());
    //     print(gC.layout());
    //     print("\n");
    // };

    // until this part, we have slices of matrices a thread block operates on

    // now to find work between the threads of the block itself
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    using MMA = decltype(make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                        Layout<Shape<_1, _1, _1>>{})); 
    
    MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x); // gets a threads work
    auto tAgA = thr_mma.partition_A(gA);  // becomes MMA, MMA_M, MMA_K, num_tiles_k)
    auto tBgB = thr_mma.partition_B(gB);  
    auto tCgC = thr_mma.partition_C(gC);  

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); //create registers to hold one tile, thats why 0
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); 
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _)); 

    // if(iy == 0 && blockIdx.x == 0 && blockIdx.y == 0 && ix == 0){

    //     print(tAgA.layout());
    //     print(tArA.layout());
    //     printf("\n");
        

    //     print(tBrB.layout());
    //     printf("\n");
        
    //     print(tCrC.layout());
    //     printf("\n\n");
    // }

    clear(tCrC);
    
    int num_tile_k = size<2>(gA); // across the kth dim(not 2 tiles) - how many tiles to iterate?
    #pragma unroll 1
        for(int itile = 0; itile < num_tile_k; ++itile) {
            cute::copy(tAgA(_, _, _, itile), tArA);
            cute::copy(tBgB(_, _, _, itile), tBrB);

            cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
        }

        cute::copy(tCrC, tCgC); 
    };

int main(){
    int M = 128, N = 128, K = 128;
    float *h_a = new float[M * K];
    float *h_b = new float[K * N];
    float *h_c = new float[M * N];
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));
    
    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // warmup
    gemm<float, 16, 16, 16><<<1, dim3(10, 10)>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int num_iterations = 10;
    cudaEventRecord(start);
    for(int i = 0; i < num_iterations; ++i) {
        gemm<float, 16, 16, 16><<<1, dim3(10, 10)>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    double flops_per_matmul = 2.0 * M * N * K;
    double total_flops = flops_per_matmul * num_iterations;
    double seconds = ms / 1000.0;  // convert ms to seconds
    double gflops = (total_flops / seconds) / 1e9; 
    
    printf("time: %.3f ms per run\n", ms / num_iterations);
    printf("total time: %.3f ms for %d runs\n", ms, num_iterations);
    printf("FLOPS: %.2f\n", total_flops);
    printf("GFLOPS: %.2f\n", gflops);
    
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}