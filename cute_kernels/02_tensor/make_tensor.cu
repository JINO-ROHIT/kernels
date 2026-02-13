#include "../util.h"
#include <stdlib.h>

using namespace cute;
using namespace std;

__global__ void create_register_tensor()
{
    // register memory (static layouts only)
    auto rshape = make_shape(Int<4>{}, Int<2>{});
    auto rstride = make_stride(Int<2>{}, Int<1>{});
    auto rtensor = make_tensor<float>(make_layout(rshape, rstride));

    PRINT("rtensor.layout", rtensor.layout());
    PRINT("rtensor.shape", rtensor.shape());
    PRINT("rtensor.stride", rtensor.stride());
    PRINT("rtensor.size", rtensor.size());
    PRINT("rtensor.data", rtensor.data());

    print("\n");
}

__global__ void create_global_tensor(int* ptr)
{
    Layout glayout = make_layout(make_shape(10, 2));
    Tensor gtensor = make_tensor(make_gmem_ptr(ptr), glayout);
    PRINTTENSOR("global tensor", gtensor);

    auto coord = make_coord(0, 1);
    PRINT("zeroth row first column", gtensor(coord));

    auto coord1 = make_coord(9, 0);
    PRINT("ninth row zeroth column", gtensor(coord1));

    auto tensor_slice_col = gtensor(_, 1);
    PRINTTENSOR("all of first column", tensor_slice_col);

    auto tensor_slice_row = gtensor(9, _);
    PRINTTENSOR("all of ninth row", tensor_slice_row);

    // think about what this does?
    // partition tensor into 2 x 2 tiles --> row tiles = 10/2 = 5 and col tiles = 2/1 = 1
    // so we have (0, 0) (1, 0) (2, 0) (3, 0) (4, 0)
    auto tensor_tile = local_tile(gtensor, make_shape(Int<2>(), Int<2>()), make_coord(1, 0));
    PRINTTENSOR("tensor tile (2,2) index (1,0)", tensor_tile);

    // pay attention here - this does strided partitioning unlike the previous ones
    // we know thread 1 corresponds to (0, 1)
    // overlay a repeat 2x2 thread pattern over the tensor
    // this gets all the elements whose cords satisfy 
    // row % 2 == 0 (all the even rows)
    // col % 2 == 1 (we only have col 0 and col 1) so col 1 gets picked
    // so we finally get
    // (0,1) = 10
    // (2,1) = 12
    // (4,1) = 14
    // (6,1) = 16
    // (8,1) = 18
    int thr_idx = 1;
    auto tensor_partition = local_partition(gtensor, Layout<Shape<_2, _2>, Stride<_2, _1>>{}, thr_idx);
    PRINTTENSOR("tensor partition tile (2,2) index (1)", tensor_partition);
}

int main(){
    create_register_tensor<<<1, 1>>>();

    int *d_ptr;
    int size = 20;
    cudaMalloc(&d_ptr, size * sizeof(int));
    int *h_ptr = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        h_ptr[i] = i;
    }
    cudaMemcpy(d_ptr, h_ptr, size * sizeof(int), cudaMemcpyHostToDevice);
    create_global_tensor<<<1, 1>>>(d_ptr);
    cudaDeviceSynchronize();


    // copy tensor
    auto rtensor = make_tensor<int>(make_layout(make_shape(Int<10>{},Int<10>{})));
    auto ctensor = make_fragment_like(rtensor);
    PRINT("ctensor.layout", ctensor.layout());
}