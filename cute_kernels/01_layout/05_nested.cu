#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {
    auto shape = make_shape(3, make_shape(2, 3));
    Layout lol = make_layout(shape);
    
    std::cout << "stride = " << stride(lol) << "\n";
    std::cout << "size = " << size(lol) << "\n";
    std::cout << "rank = " << rank(lol) << "\n";

    print_layout(lol);
}