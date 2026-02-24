#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(
        make_shape (4,  make_shape (2, 4)),
        make_stride(2,  make_stride(1, 8))
    );

    auto coord = make_coord(0, make_coord(_, _));
    auto out = slice(coord, layout);
    print_layout(layout);
    print_layout(out);
}