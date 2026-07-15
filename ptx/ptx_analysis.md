### ptx source code analysis for different kernels

- for a naive matmul, let's analyze its generated ptx code: `naive_mm.sm_89.ptx`

## 1. the entrypoint

```
(
    .param .align 8 .b8  param_0[8],   // pointer to matrix A
    .param .align 8 .b8  param_1[8],   // pointer to matrix B
    .param .align 8 .b8  param_2[8],   // pointer to matrix C
    .param .u32           param_3,      // M
    .param .u32           param_4,      // N
    .param .u32           param_5       // K
)
```

- 8 byte aligned in memory
- `b8` - raw 8 bytes = 64 bits = a pointer. ptx doesn't have a native pointer type, so pointers are passed as 8-byte raw byte arrays
- `.u32` - 32 bit unsigned integer

```
.reqntid 32, 32, 1
```

requires the thread block dimensions to be exactly 32x32x1 (1024 threads).

## 2. register declaration

```
.reg .pred  %p<14>;   // 14 predicate registers
.reg .b16   %rs<2>;   // 2 x 16-bit registers
.reg .b32   %r<54>;   // 54 x 32-bit registers
.reg .f32   %f<71>;   // 71 x 32-bit float registers
.reg .b64   %rd<25>;  // 25 x 64-bit registers
```

ptx uses virtual registers. declare upfront how many you need and the compiler maps them to actual hardware registers.
the <N> syntax is shorthand for declaring N registers at once.

| type | width | purpose |
|------|-------|---------|
| pred | 1-bit | boolean flags for conditional branches |
| b16 | 16-bit | raw bits - used for fp16 round-trip |
| b32 | 32-bit | integers - loop counters, indices, dimensions |
| f32 | 32-bit | floats - accumulator and matrix values |
| b64 | 64-bit | pointers to a, b, c in global memory |

## 3. loading params

```
ld.param.u64  %rd6, [param_2]   // C pointer ---> %rd6
ld.param.u64  %rd5, [param_1]   // B pointer ---> %rd5
ld.param.u64  %rd4, [param_0]   // A pointer ---> %rd4
ld.param.u32  %r21, [param_3]   // M ---> %r21
ld.param.u32  %r22, [param_4]   // N ---> %r22
ld.param.u32  %r24, [param_5]   // K ---> %r24
```

ld.param read from param space to the registers

## 4. reading built-in thread/block ids

```
mov.u32  %r25, %ctaid.x   // blockIdx.x
mov.u32  %r26, %ctaid.y   // blockIdx.y
mov.u32  %r27, %ntid.x    // blockDim.x  (= 32, always)
mov.u32  %r28, %ntid.y    // blockDim.y  (= 32, always)
mov.u32  %r29, %tid.x     // threadIdx.x
mov.u32  %r30, %tid.y     // threadIdx.y
```

## 5. compute global row and col
```
mad.lo.s32  %r31, %r25, %r27, %r29   // col = blockIdx.x * blockDim.x + threadIdx.x
mad.lo.s32  %r2,  %r26, %r28, %r30   // row = blockIdx.y * blockDim.y + threadIdx.y
```

mad.lo.s32 = multiply-add that is a * b + c. this is the standard CUDA index calculation, just written in one instruction instead of two.

## 6. boundary checking

```
setp.ge.s32  %p1, %r2,  %r21    // p1 = (row >= M)
setp.ge.s32  %p2, %r31, %r22    // p2 = (col >= N)
or.pred      %p3, %p1,  %p2     // p3 = (row >= M || col >= N)
setp.lt.s32  %p4, %r24, 1       // p4 = (K < 1)
mov.f32      %f70, 0f00000000   // f70 = 0.0f  (default output)
or.pred      %p5, %p3,  %p4     // p5 = out-of-bounds || K==0
@%p5 bra     $L__BB0_13         // if p5, jump to store 0 and return
```

## 7. loop

```
shl.b32   %r3,  %r2,  11       // r3  = row * 2048  (row << 11, since 2^11 = 2048)
and.b32   %r4,  %r24, 7        // r4  = K % 8       (remainder for cleanup loops)
setp.lt.u32 %p6, %r24, 8       // p6  = (K < 8)
mov.b32   %r51, 0              // r51 = k = 0       (loop counter)
mov.f32   %f69, 0f00000000     // f69 = acc = 0.0f  (accumulator)
@%p6 bra  $L__BB0_4            // if K < 8, skip main loop entirely
```

shl.b32 %r3, %r2, 11 is an optimization. multiplying by 2048 is just a left shift by 11 bits (since 2¹¹ = 2048), which is much faster than a multiply. This pre-computes the row offset into matrix A.

```
and.b32   %r34, %r24, 2147483640   // r34 = K rounded down to multiple of 8
                                    //       (2147483640 = 0xFFFFFFF8)
neg.s32   %r48, %r34               // r48 = loop trip counter (counts up to 0)
add.s64   %rd3, %rd4, 16           // rd3 = A + 16 bytes (pointer pre-adjustment)
mov.u32   %r47, %r3                // r47 = row offset (loop variable)
mov.u32   %r49, %r31               // r49 = col (loop variable)
```