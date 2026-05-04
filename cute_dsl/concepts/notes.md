# notes

a collection of random patterns and notes i find interesting.

1. `(1,16) : (0,1)` - 1 row of 16 elements next to each other, basically coalesced.

if each element is stored in bf16, then this would issue 2 `LD128` instructions, why?

```
1 LDG = 128 bits
16 elements stored in 16 bit = 16 * 16 = 256
                                        = 2 * 128 instruction
```