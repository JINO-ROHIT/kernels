// inspired from https://github.com/Dao-AILab/flash-attention

#include <cute/tensor.hpp>
#include <stdio.h>
#include <math.h>

using namespace cute;

template <bool scale_max = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline void scale_exp2(
    Tensor<Engine0, Layout0> &tensor,      
    Tensor<Engine1, Layout1> const &max, 
    const float scale
) 

{
    static_assert(Layout0::rank == 2, "needs to be 2d");
    static_assert(Layout1::rank == 1, "needs to be 1d");

    #pragma unroll
    for(int i = 0; i < size<0>(tensor); ++i){
        const float max_scaled = max(i) == -INFINITY ? 0.f : max(i) * (scale_max ? scale : float(M_LOG2E)); // btw max(i) is the tensor max for that row

        #pragma unroll
        for(int j = 0; j < size<1>(tensor); ++j){
            tensor(i, j) = exp2f(tensor(i, j) * scale - max_scaled);
            // allows the compiler to use the ffma instruction instead of fadd and fmul separately.
            // :3 check the ptx
        }
    }
}


template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline void naive_scale_exp2(
    Tensor<Engine0, Layout0> &tensor,      
    Tensor<Engine1, Layout1> const &max
) {
    static_assert(Layout0::rank == 2, "needs to be 2d");
    static_assert(Layout1::rank == 1, "needs to be 1d");
    
    for(int i = 0; i < size<0>(tensor); ++i) {
        float row_max = max(i);
        for(int j = 0; j < size<1>(tensor); ++j) {
            // exp(x - max)
            tensor(i, j) = expf(tensor(i, j) - row_max);
        }
    }
}

int main() {
    const int rows = 2;
    const int cols = 3;
    const float scale = float(M_LOG2E);  // ~1.44
    
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f,   
                                 2.0f, 5.0f, 3.0f}; 
    
    std::vector<float> max_vals = {3.0f, 5.0f};
    
    std::vector<float> data_b = data_a; 
    std::vector<float> data_c = data_a;  
    
    auto layout_2d = make_shape(rows, cols);
    auto layout_1d = make_shape(rows);
    
    Tensor tensor_a = make_tensor(data_a.data(), layout_2d);
    Tensor tensor_b = make_tensor(data_b.data(), layout_2d);
    Tensor tensor_c = make_tensor(data_c.data(), layout_2d);
    Tensor max_tensor = make_tensor(max_vals.data(), layout_1d);
    
    print_tensor(tensor_a);
    print_tensor(max_tensor);
    
    scale_exp2<true>(tensor_b, max_tensor, scale);
    naive_scale_exp2(tensor_c, max_tensor);
    
    std::cout << "exp2 version" << std::endl;
    print_tensor(tensor_b);

    std::cout << "naive version" << std::endl;
    print_tensor(tensor_c);

}