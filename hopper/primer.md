### primer on some fundamentals

there are some important operations we need to understand before stepping into kernels.

#### hexadecimal

hexadecimal is actually base 16 instead of the normal base 10.

| Hex  | Decimal |
| ---- | ------- |
| 0x1  | 1       |
| 0xA  | 10      |
| 0xF  | 15      |
| 0x10 | 16      |


```
0x4      = 4
0x10     = 16
0x3FFFF  = 262143 ( 0x3FFFF = 3×16⁴ + F×16³ + F×16² + F×16¹ + F×16⁰)
```

#### bitwise &

```
5  = 0101
3  = 0011
---------
&  = 0001  ---> 1
---------

1 & 1 = 1
anything else = 0
```

so when you use a mask like (x & mask), then you -
- keeps only the bits where mask has 1
- zeros everything else

#### right shift >>

shift bits to the right, every shift is a divide by 2

```
8 = 1000
8 >> 1 = 0100 = 4
8 >> 2 = 0010 = 2

x >> 4  = x / 16
```