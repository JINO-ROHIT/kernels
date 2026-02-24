#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {
    Layout lol = make_layout(make_shape(4), make_stride(2));
    auto result = complement(lol, 8);
    
    std::cout << result << std::endl;
    std::cout << "stride = " << stride(result) << "\n";
    std::cout << "size = " << size(result) << "\n";
    std::cout << "rank = " << rank(result) << "\n";

    print_layout(result);
}