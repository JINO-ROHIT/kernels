### CuTe

CuTe is a layout abstraction for working on multidimensional layouts of data. i think it was introduced as a part of Cutlass 3.xxx. the idea is while it handles the abstractions for the data and layout, programmers can focus on the actual kernel logic.

CuTe is a header only library!


1. integers

- dynamic/run-time integers are ordinary types like int or size_t etc
- static/compile time integers like Int<1>, Int<2>. These types encode the value as a static constexpr member

2. tuple

- ordered list of zero or more elements.

3. intTuple

- integer or a tuple of IntTuples(this is recursive)

4. shapes and strides

- they are intTuple

5. layout

- tuple of (shape, stride)
- when stride is not specified, it it created from the shape with layoutleft as default. creates exclusive prefix product from left to right.
- layout right is from right to left.

6. tensor

- layout can be composed with data like a pointer or an array to create a tensor.

