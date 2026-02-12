#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {
    auto lol = Layout<Shape <_2,Shape <_1,_6>>,
                     Stride<_1,Stride<_6,_2>>>{};

    auto result = coalesce(lol);
    
    std::cout << result << std::endl;
    std::cout << "stride = " << stride(result) << "\n";
    std::cout << "size = " << size(result) << "\n";
    std::cout << "rank = " << rank(result) << "\n";

    //print_layout(result);
}


// Mode 0: _2:_1 (size 2, stride 1)
// Mode 1: _1:_6 (size 1, stride 6)
// Mode 2: _6:_2 (size 6, stride 2)

// first --> _2:_1 ++ _1:_6	1 (size-1)	_2:_1
// second --> _2:_1 ++ _6:2 = _12:_1