### CuTe

CuTe is a layout abstraction for working on multidimensional layouts of data. i think it was introduced as a part of Cutlass 3.xxx. the idea is while it handles the abstractions for the data and layout, programmers can focus on the actual kernel logic.

CuTe is a header only library!


```
some terms to know -

1. rank(Layout) - the number of modes in a Layout. Equivalent to the tuple size of the Layout’s shape.

2. depth(Layout): The depth of the Layout’s shape. A single integer has depth 0, a tuple of integers has depth 1, a tuple of tuples of integers has depth 2, etc.

3. shape(Layout): The shape of the Layout.

4. stride(Layout): The stride of the Layout.

5. size(Layout): The size of the Layout function’s domain. Equivalent to size(shape(Layout)).

6. cosize(Layout): The size of the Layout function’s codomain (not necessarily the range). Equivalent to A(size(A) - 1) + 1.

For 4:1 mapping to [0,1,2,3]:

cosize = 4 (spans addresses 0 through 3)

For 4:2 mapping to [0,2,4,6]:

cosize = 7 (spans addresses 0 through 6, even though we only use 4 of them)

For (2,3):(1,4) mapping to [0,1,4,5,8,9]:

cosize = 10 (spans addresses 0 through 9)

7. codomain: set of all memory addresses that layout produces.

for 4:2: codomain = {0, 2, 4, 6}
```

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
- when stride is not specified, it it created from the shape with layoutleft as default(column major). creates exclusive prefix product from left to right.
- layout right is from right to left.(row major)

note: column major puts all the elements one by one in the first column.

6. tensor

- layout can be composed with data like a pointer or an array to create a tensor.


#### co-ordinate system

the map from an input coordinate to a natural coordinate is the application of a colexicographical order (reading right to left, instead of “lexicographical,” which reads left to right) within the Shape.


for example `(3, (2, 3))`

```
Shape breakdown:
┌─────────────────────────────────────┐
│  (3,        (2, 3))                 │
│   │          │  │                   │
│   │          │  └─ Inner size: 3    │
│   │          └─ Outer size: 2       │
│   └─ Size: 3                        │
│                                     │
│  Total elements: 3 × 2 × 3 = 18     │
└─────────────────────────────────────┘

3 "blocks" (dimension 0)
each block has 2 "rows" (dimension 1 inner)
each row has 3 "cols" (dimension 1 outer)
```

mapping table

```
| 1-D | 2-D   | Natural   | Explanation                                 |
| --- | ----- | --------- | ------------------------------------------- |
| 0   | (0,0) | (0,(0,0)) | start: block 0, sub-row 0, col 0            |
| 1   | (1,0) | (1,(0,0)) | move to next block (dim 0), same position   |
| 2   | (2,0) | (2,(0,0)) | mext block                                  |
| 3   | (0,1) | (0,(1,0)) | back to block 0, next sub-row (dim 1 inner) |
| ... | ...   | ...       | ...                                         |
| 9   | (0,3) | (0,(1,1)) | Block 0, sub-row 1, col 1                   |

```

colexographical visualized

```
Shape (3, (2, 3)) - Think of it as 3 blocks of 2×3 matrices:

Block 0:          Block 1:          Block 2:
┌───────┐        ┌───────┐        ┌───────┐
│0  3  6│        │1  4  7│        │2  5  8│   <- Row 0 of each sub-block
│9 12 15│        │10 13 16│       │11 14 17│  <- Row 1 of each sub-block
└───────┘        └───────┘        └───────┘
  ↑                ↑                ↑
  │                │                │
  └────────────────┴────────────────┘
   These are the 3 "outer" elements (dim 0)
   
Within each block, we read column-major (colexicographic on the inner shape)
```



```
auto shape = Shape<_3,Shape<_2,_3>>{};
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```


#### index mapping

the map from a natural coordinate to an index is performed by taking the inner product of the natural coordinate with the Layout’s Stride.

#to-do add examples for index mapping and layout manipulation


#### layout algebra

1. coalesce - coalesce is a simplify operation on layouts. It reduces the number of modes (dimensions) in a layout without changing the underlying mapping from integers to integers.


the four cases of coalescing

| Case  | Condition             | Result            | Explanation                                                                                           |
| ----- | --------------------- | ----------------- | ----------------------------------------------------------------------------------------------------- |
| **1** | `s0:d0 ++ _1:d1`      | `s0:d0`           | size-1 modes contribute nothing (always index 0)                                                      |
| **2** | `_1:d0 ++ s1:d1`      | `s1:d1`           | Size-1 modes contribute nothing                                                                       |
| **3** | `s0:d0 ++ s1:(s0×d0)` | `(s0×s1):d0`      | **Contiguous merge**: If stride of second equals the "reach" of first, they form one contiguous block |
| **4** | Otherwise             | `(s0,s1):(d0,d1)` | Cannot combine, keep separate                                                                         |


check 06_coalesce.cu for example.

2. composition - is a way of chaining layouts (to-do understand in detail)

```
R = A o B means apply B first and then A
```
# to-do later


3. complement - given a layout A that selects some coordinates from a space of size M, what layout describes all the coordinates NOT selected by A?

```
Layout complement(LayoutA const& layout_a, Shape const& cotarget)

layout_a: The original layout (what we're already covering)
cotarget: The total space size we want to cover (often just an integer like 24)

original layout: 4:1
target space: 24
question: What is the complement?

layout 4:1 --> 4 elements strided at 1 [0, 1, 2, 3]

cotarget = 24 --> we need to cover [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

whats missing = [4, 5, 6, 7 ..... 23]

but we need a pattern? the original 4:1 is a tile of 4 elements, to reach 24, we repeat the tile 6 times (4 * 6)

each tile starts at 

tile 0 - 0
tile 1 - 4
tile 2 - 8

difference between each position(ie the stride) is 4

hence the pattern is 6:4

```

some post conditions are -

- size constraint

```
cosize(make_layout(layout_a, result)) >= size(cotarget)
```

when you combine the original layout with its complement, you can cover at least the entire cotarget space.

- ordered property 

```
result(i-1) < result(i)  // for all ic
```

the complement is strictly ordered - indices increase monotonically. this makes the complement unique and predictable

- disjoint codomains

```
result(i) != layout_a(j)  // for all i, j
```
the  complement never produces indices that the original layout already covers (no overlap).
#### tensor

a tensor is represented by two template parameters - engine and layout.

tensors can be owning or non-owning.

owning tensors behave like std::array. when you copy the tensor, you deep-copy its elements, and the tensor’s destructor deallocates the array of elements.

nonowning tensor’s behave like a (raw) pointer. copying the tensor doesn’t copy the elements, and destroying the tensor doesn’t deallocate the array of elements.

1. non owning tensor

a tensor is usually a nonowning view of existing memory.
created by calling make_tensor with two arguments: a random-access iterator, and the layout or arguments to construct a layout.

```
float* A = ...;

// Untagged pointers
Tensor tensor_8   = make_tensor(A, make_layout(Int<8>{}));  // Construct with Layout
Tensor tensor_8s  = make_tensor(A, Int<8>{});               // Construct with Shape
Tensor tensor_8d2 = make_tensor(A, 8, 2);                   // Construct with Shape and Stride

// Global memory (static or dynamic layouts)
Tensor gmem_8s     = make_tensor(make_gmem_ptr(A), Int<8>{});
Tensor gmem_8d     = make_tensor(make_gmem_ptr(A), 8);
Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{},16));
Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape (      8  ,Int<16>{}),
                                                   make_stride(Int<16>{},Int< 1>{}));

// Shared memory (static or dynamic layouts)
Layout smem_layout = make_layout(make_shape(Int<4>{},Int<8>{}));
__shared__ float smem[decltype(cosize(smem_layout))::value];   // (static-only allocation)
Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});`
```


2. owning tensor

owning tensors are created by calling make_tensor<T>, where T is the type of each element of the array, and a layout or arguments to construct a layout. 
The array is allocated analogously to std::array<T,N> and, therefore, owning tensors must be constructed with a layout that has static shapes and static strides. CuTe does not perform dynamic memory allocation in Tensors as it is not a common or performant operation within CUDA kernels.

```
// Register memory (static layouts only)
Tensor rmem_4x8_col = make_tensor<float>(Shape<_4,_8>{});
Tensor rmem_4x8_row = make_tensor<float>(Shape<_4,_8>{},
                                         LayoutRight{});
Tensor rmem_4x8_pad = make_tensor<float>(Shape <_4, _8>{},
                                         Stride<_32,_2>{});
Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
```

### Matrix Multiply Accumulate(MMA) [for volta section]

the abstraction abstraction wraps GPU Tensor Core instructions in such a way so you dont have to write assembly. it has a 4 level hierarchy.

1. operation struct - a minimal wrapper around the ptx instruction. it knows nothing about layouts or tensors.

```
SM70_8x8x4_F32F16F16F32_NT

| Component      | Meaning                                             |
| -------------- | --------------------------------------------------- |
| `SM70`         | Architecture (Volta)                                |
| `8x8x4`        | M×N×K dimensions                                    |
| `F32F16F16F32` | D=A×B+C types (D, A, B, C)                          |
| `NT`           | A=No-transpose (col-major), B=Transpose (row-major) |

```

2. traits - they add semantic metadata to the raw operation; shapes, types, and crucially, layouts that map threads and values to matrix coordinates.

```
template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>
{
    // Logical compute types
    using ValTypeD = float;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = float;

    // Logical shape of the MMA operation
    using Shape_MNK = Shape<_8, _8, _4>;

    // Thread mapping: which warp threads participate?
    using ThrID = Layout<Shape<_4, _2>, Stride<_1, _16>>;
    // Maps logical thread [0-7] → actual warp threads [0,1,2,3] ∪ [16,17,18,19]
    // This is the "quadpair" (QP) pattern: 8 threads working together

    // Layouts for each matrix: (thread, value) → (M, K) or (M, N) coordinate
    using ALayout = SM70_8x4_Col;   // 8×4 column-major layout for A
    using BLayout = SM70_8x4_Col;   // 8×4 column-major layout for B  
    using CLayout = SM70_8x8_32b;   // 8×8 layout for C/D
};
```

3. atom

combines operation and trait into a usable object

```
using MyAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>;
MyAtom atom;
```

4. titlemmma - scales up a single atom to handle larger tiles by replicating and interleaving Atoms across threads and values.

```
// Single Atom (8×8×4)
TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{});

// Equivalent explicit form:
TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                              Layout<Shape<_1,_1,_1>>{},  // 1×1×1 Atom layout
                              Tile<_8,_8,_4>{});          // Tile size matches Atom
```