### implementing convolution in 1d

convolution is where you take an input tensor and a mask tensor and then do a weighted sum.

btw you need to align the centre of the mask to each element you compute convolution for.

input size = output size

1. naive.cu - this version uses global memory and loads each element and does a weighted sum.
2. smem.cu - this version uses shared memory to load the elements into tiles and compute the result.
3. cmem.cu - this version uses fixed constant memory for the mask and the rest of the computation is the same.
4. [to-do] vectorized_load.cu - this version uses float4 instead of using a single float.