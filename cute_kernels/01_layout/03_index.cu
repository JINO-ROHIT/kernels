#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {
    Layout lol = make_layout(make_shape(4, 3), make_stride(1, 4)); // column major, whats at (2, 1)
    std::cout << lol << std::endl;

    std::cout << "shape  = " << shape(lol) << "\n";
    std::cout << "stride = " << stride(lol) << "\n";
    std::cout << "size = " << size(lol) << "\n";
    std::cout << "rank = " << rank(lol) << "\n";

    std::cout << "element at (2, 1) = " << lol(2, 1) << std::endl;

    print_layout(lol);
}