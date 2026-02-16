### CuTe

## to-do learn and write more notes on nested cases

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

2. composition - is a way of chaining layouts

```
R = A o B means apply B first and then A
```

a simple example -

```
Function B(x) = x + 2
Function A(x) = x × 3

Composition R = A ∘ B

R(5) = A(B(5))
     = A(5 + 2)
     = A(7)
     = 7 × 3
     = 21
```

remember layout is just a way to map cordinates to memory addresses.

```
Layout B = 4:3
This means: address = coordinate * 3

B(0) = 0 × 3 = 0
B(1) = 1 × 3 = 3
B(2) = 2 × 3 = 6
B(3) = 3 × 3 = 9
```


lets work through an example now -

```
A = 12:2  (12 elements, stride 2)
B = 4:3   (4 elements, stride 3)

R = A o B

What is R = A o B?


layout a

A = 12:2

A(0) = 0 × 2 = 0
A(1) = 1 × 2 = 2
A(2) = 2 × 2 = 4
A(3) = 3 × 2 = 6
A(4) = 4 × 2 = 8
A(5) = 5 × 2 = 10
A(6) = 6 × 2 = 12
A(7) = 7 × 2 = 14
A(8) = 8 × 2 = 16
A(9) = 9 × 2 = 18
A(10) = 10 × 2 = 20
A(11) = 11 × 2 = 22

A maps to: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

layout b

B = 4:3

B(0) = 0 × 3 = 0
B(1) = 1 × 3 = 3
B(2) = 2 × 3 = 6
B(3) = 3 × 3 = 9

B maps to: [0, 3, 6, 9]


compute R = A o B

R(0) = A(B(0))
     = A(0)
     = 0

R(1) = A(B(1))
     = A(3)
     = 6

R(2) = A(B(2))
     = A(6)
     = 12

R(3) = A(B(3))
     = A(9)
     = 18

composition R produces addresses = [0, 6, 12, 18]

basically R = (4:6)

so what happened?

Original A = 12:2
Selects every 2nd element: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

Layout B = 4:3
Selects every 3rd element from whatever it's applied to

Composition A ∘ B:
Takes every 3rd element FROM the "every 2nd element" sequence
Result: [0, 6, 12, 18]

Memory:     [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17][18][19][20][21][22]
A selects:   ^     ^     ^     ^     ^      ^      ^      ^       ^       ^       ^       ^
                   (every 2nd position)
                   
B then selects every 3rd from those:
             ^                 ^                     ^                          ^
            0                 6                    12                         18



formula for simple composition

```
A = a:b  (shape a, stride b)
B = s:d  (shape s, stride d)

result R = A o B = s:(b×d)
```

apply to our example

```
A = 12:2  (a=12, b=2)
B = 4:3   (s=4, d=3)

R = s:(bxd)
  = 4:(2x3)
  = 4:6 
```

```

# to-do later nested 


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

# to-do
4. division(tiling)

5. product(tiling)

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

### Matrix Multiply Accumulate(MMA)

tensor cores introduced with volta are designed to matrix mul extremely fast. earlier we depended on cuda cores.

we have two level ways to access them -
1. WMMA (Warp Matrix Multiply Accumulate) - Somewhat abstracted, easier but less control
2. MMA (Matrix Multiply Accumulate) - Direct assembly instructions, maximum control but very difficult

it has 5 layers of abstract so you dont have to write assembly directly 0

1. mmOperation - his is the actual assembly instruction that talks directly to the Tensor Core hardware.

SM75_16x8x8_F32F16F16F32_TN
let's decode this:

SM75 = Turing architecture GPU
16x8x8 = Matrix sizes (M=16, N=8, K=8)
F32F16F16F32 = Data types for D, A, B, C (float32, float16, float16, float32)
TN = Matrix orientations (Transposed, Normal)

It contains a fma (fused multiply-add) function that executes the actual hardware instruction.

2. MMA_Traits - the information bridge
think of this as a translator or specification sheet.
The hardware layer (MMAOperation) just executes instructions. But programmers need additional information:

What types of data can I use?
What's the logical shape of the matrices?
How are threads organized?
How is data laid out in memory?

MMA_Traits provides this metadata - information about the operation that doesn't exist in the hardware instruction itself but is essential for using it correctly.

3. MMA_Atom - the smallest unit
an atom is the smallest matrix multiplication the hardware can perform in one operation.
for example: A specific Tensor Core might be able to multiply a 16×8 matrix by an 8×8 matrix in one go. That's one "atom" of computation.
you can't break it down smaller - it's atomic.

4. TiledMMA - scaling Up
how do I combine multiple atoms to handle larger matrices?"

It has two extension methods:
 - AtomLayoutMNK - Use more execution threads (parallel processing)

Like hiring more workers to do more tasks simultaneously

 - ValLayoutMNK - Repeat the same atom multiple times (serial processing)

Like one worker doing the same task multiple times

5. ThrMMA - thread level view

in CUDA programming, we need to write code for individual threads

each thread needs to know what work to do

ThrMMA takes a specific thread ID and tells that thread:
- Here's your slice of matrix A
- Here's your slice of matrix B  
- Here's where your results go in matrix C

- `partition_A/B/C()` - What part of the matrix does this thread handle?
- `partition_fragment_A/B/C()` - Put that data into registers for computation



### pipelining

taking the instance of blocked gemm, if we have 1000 x 1000 matrix, we usually create smaller blocks of example 100 x 100 and then iterate across the depth direction and calculate the dot products to get the result. the idea of pipelining is to overlap the operations so the gpu is used more efficiently.

for the tile processing we do, the flo is basically -
```
load from global ---> shared memory --> registers


without pipelining

Tile 0: [LDGSTS -> LDSM -> MMA] -> wait -> 
Tile 1: [LDGSTS -> LDSM -> MMA] -> wait -> 
Tile 2: [LDGSTS -> LDSM -> MMA]


with pipelining

Time 1: Tile 0 LDGSTS
Time 2: Tile 0 LDSM,    Tile 1 LDGSTS
Time 3: Tile 0 MMA,     Tile 1 LDSM,    Tile 2 LDGSTS
Time 4: Tile 1 MMA,     Tile 2 LDSM,    Tile 3 LDGSTS

```


after the ampere gpus, we have this `cp.async` that issues copy and then move onto to another instruction. for synchronization, we have the commit and wait -
1. commit: mark a checkpoint - "i've issued a copy command"
2. wait<N>: Wait until at most N copy operations are still incomplete


wait<1>  // Wait until at most 1 copy is incomplete

wait<0>  // Wait until 0 copies are incomplete
         // This means ALL copies are DONE