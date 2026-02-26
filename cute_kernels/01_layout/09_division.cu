#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(
        make_shape(4, 6),
        make_stride(6, 1)
    );

    // auto divider = make_tile(
    //     make_shape(2, 2),
    //     make_stride(2, 1)
    // );
    auto divider = make_tile(
        Layout<_2, _1>{},   
        Layout<_2, _1>{} 
    );

    auto out = zipped_divide(layout, divider);
    print_layout(layout);
    print_layout(out);
}