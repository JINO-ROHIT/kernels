#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {
    Layout lol = make_layout(make_shape(2, 4), LayoutRight()); // stride as (4, 1)
    std::cout << lol << std::endl;

    std::cout << "shape  = " << shape(lol) << "\n";
    std::cout << "stride = " << stride(lol) << "\n";
    std::cout << "size = " << size(lol) << "\n";
    std::cout << "rank = " << rank(lol) << "\n";

    print_layout(lol);
}