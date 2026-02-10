#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {

    Layout s8 = make_layout(Int<8>{});
    Layout d8 = make_layout(8);

    Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
    Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

    Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                                 make_stride(Int<12>{},Int<1>{}));
    Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                                   LayoutLeft{});
    Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                                   LayoutRight{});

    Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                               make_stride(4,make_stride(2,1)));
    Layout s2xh4_col = make_layout(shape(s2xh4),
                                   LayoutLeft{});

    std::cout << "s8        : "; print(s8);        std::cout << "\n";
    std::cout << "d8        : "; print(d8);        std::cout << "\n";
    std::cout << "s2xs4     : "; print(s2xs4);     std::cout << "\n";
    std::cout << "s2xd4     : "; print(s2xd4);     std::cout << "\n";
    std::cout << "s2xd4_a   : "; print(s2xd4_a);   std::cout << "\n";
    std::cout << "s2xd4_col : "; print(s2xd4_col); std::cout << "\n";
    std::cout << "s2xd4_row : "; print(s2xd4_row); std::cout << "\n";
    std::cout << "s2xh4     : "; print(s2xh4);     std::cout << "\n";
    std::cout << "s2xh4_col : "; print(s2xh4_col); std::cout << "\n";

    std::cout << "\ns2s4 layout : "; print_layout(s2xs4); std::cout << "\n";// wont work on the 1d ones
}