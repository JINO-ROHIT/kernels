# notes

a collection of random patterns and notes i find interesting.

1. `(1,16) : (0,1)` - 1 row of 16 elements next to each other, basically coalesced.

if each element is stored in bf16, then this would issue 2 `LD128` instructions, why?

```
1 LDG = 128 bits
16 elements stored in 16 bit = 16 * 16 = 256
                             = 2 * 128 instruction
```

### get_slice vs partition

```
get_slice(tid_x) picks the per thread slice object for a particular tiled operation.

  in your code:

  thr_copy_a = tiled_copy_a.get_slice(tid_x)
  thr_mma = tiled_mma.get_slice(tid_x)

  this does not yet give you data. it gives you a thread-specific mapper/view that knows how thread tid_x participates in that copy or MMA layout.

  then you apply that mapper to actual tensors with partition_*:

  - thr_copy_a.partition_S(gA) -> which part of global A this thread reads
  - thr_copy_a.partition_D(sA) -> which part of shared A this thread writes
  - thr_mma.partition_A(sA) -> which part of shared A this thread uses for MMA
  - thr_mma.partition_C(gC) -> which part of C this thread owns

  so the distinction is:

  - get_slice(tid_x) = “what is thread tid_x’s role in this tiled copy/MMA?”
  - partition_* = “apply that role to this specific tensor”

  a good mental model is:

  - tiled_copy_a / tiled_mma = the full thread-data mapping for the whole block
  - get_slice(tid_x) = extract one thread’s mapping
  - partition_* = use that mapping on a concrete tensor

  so get_slice is like selecting the thread’s lane descriptor; partition_* uses that descriptor to produce actual per-thread tensor views.
```