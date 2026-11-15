### amd kernels

all the kernels are in triton/gluon and written for gfx942.

Benchmark a kernel by filename (`.py` is optional). Relative kernel names resolve
inside `amd/gemm`, regardless of the working directory:

```sh
python amd/gemm/bench.py --kernel 01_naive --K 4096 --dtype fp16
```

'''
uv pip install 'torch==2.15.0+rocm10.0' \
  --index-url https://download.pytorch.org/whl/rocm10.0

uv pip install numpy matplotlib pandas

apt-get update -qq
apt-get install -y python3.12-dev
'''


kernel 2 - buffer load

Buffer operations are an AMD-specific memory access mechanism with built-in out-of-bounds (OOB) checking.
The key insight is that buffer ops handle OOB checks in hardware, eliminating the need for branches in the generated code. This is particularly valuable for masked loads common in GEMM kernels.

Pros:

Automatic OOB check — Hardware handles out-of-bounds access, producing efficient codegen for masked loads
Potentially fewer VGPRs for addresses — Buffer ops keep the shared base separate from per-lane 32-bit offsets; savings depend on the addressing mode the compiler would otherwise generate

Cons:

32-bit offset limitation — In theory, buffer ops cannot access address spaces larger than 4 GB
Range analysis constraints — The compiler's range analysis only works reliably when the whole tensor is small (< 2 GB), further limiting use cases

VGPRs vs SGPRs

On gfx942, a wavefront contains 64 lanes:

| Register type | Meaning | Values held by one register across a wave |
| --- | --- | --- |
| VGPR | Vector General-Purpose Register | 64 separate 32-bit values, one per lane |
| SGPR | Scalar General-Purpose Register | One 32-bit value shared by all lanes |

Each wave has its own scalar-register values; the sharing is within a wave.

For example, lanes can access different elements of the same array using one
shared base address and different offsets. These example offsets are in bytes:

```text
Shared 64-bit base in s[0:1] = 1000

                    Lane 0   Lane 1   Lane 2   Lane 3   ...
Offsets in VGPR v0       0        4        8       12   ...
Memory addresses     1000     1004     1008     1012   ...
```

The base uses two SGPRs because it is 64 bits. The offsets use one VGPR because
each lane's offset is 32 bits. Values loaded from A/B and per-lane intermediate
results are other examples of data held in VGPRs.

This explains the potential address-register saving from buffer operations:
materializing a full 64-bit address in each lane needs two VGPRs, whereas buffer
addressing can use one VGPR for the offset and keep the buffer descriptor
(including its base address and bounds) in SGPRs.

The calculation is still `base + offset`. Ordinary AMD global loads can also use
a scalar base plus a 32-bit vector offset. If the compiler already generates that
form for `gl.load`, switching to buffer operations may not reduce VGPR usage.


kernel 3 - async copy

load data directly from global memory into LDS, bypassing registers entirely.

This has two immediate advantages:

Register savings — Data does not need to be staged in VGPRs before being written to LDS
Instruction reduction — Eliminates ds_write instructions entirely (ds write is when value from vgprs is written into lds)


to profile

```
/opt/rocm/core-10.1/bin/rocprofv3 \
  --kernel-trace --stats --output-format csv \
  -d /root/profile_results/bench_fixed -- \
  python3 bench.py --kernel 01_naive --K 4096 --dtype fp16 --rocprof
```

ds_read when load data from LDS ionto each lanes VGPRs so another instruction can calculate w them.

A SIMD is a hardware execution unit that applies one instruction to multiple lanes’ data.

For example, the instruction:

v2 = v0 + v1

makes each active lane add its own values:

Lane 0: v2[0] = v0[0] + v1[0]
Lane 1: v2[1] = v0[1] + v1[1]
...

A wave is the group of threads being executed; a SIMD is hardware that executes their instructions. Its associated register file holds their VGPR values

For your CDNA3/gfx942 GPU, each compute unit has four 16-wide SIMD units. A wave has 64 logical lanes, so ordinary vector arithmetic processes that wave in groups rather than requiring 64 physical arithmetic lanes in one SIMD.

Its vector arithmetic hardware handles operations such as:

| Work | Examples |
|---|---|
| Floating-point arithmetic | Add, multiply, fused multiply-add |
| Integer arithmetic | Index calculations, offsets |
| Bit operations | AND, OR, XOR, shifts |
| Comparisons and selection | Compare values, choose results |
| Type conversions | Convert between supported numeric types |

So when documentation says “data returns from LDS to the SIMD,” it means the data arrives in the SIMD’s associated VGPR register file, ready for subsequent instructions.

