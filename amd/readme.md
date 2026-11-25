# AMD kernels

These GEMM kernels are written in Triton/Gluon for the AMD Instinct MI300X
(`gfx942`).

## Setup

```sh
uv pip install 'torch==2.15.0+rocm10.0' \
  --index-url https://download.pytorch.org/whl/rocm10.0
uv pip install numpy matplotlib pandas

apt-get update -qq
apt-get install -y python3.12-dev
```

## Benchmarking

Run the benchmark from the repository root. The `--kernel` argument accepts a
filename with or without the `.py` extension. Relative kernel names resolve
inside `amd/gemm`, regardless of the working directory.

```sh
python amd/gemm/bench.py --kernel 01_naive --K 4096 --dtype fp16
```

### FP16 results

All four runs used `M = N = K = 4096`. Each row includes the rocBLAS throughput
measured in the same run.

| Kernel | Implementation | Triton (TFLOPS) | rocBLAS (TFLOPS) |
| --- | --- | ---: | ---: |
| [01_naive.py](gemm/01_naive.py) | Regular dot product | 9.879713 | 598.596515 |
| [02_mfma.py](gemm/02_mfma.py) | 16 × 16 × 16 MFMA instruction | 224.862615 | 595.990350 |
| [03_buffer_load.py](gemm/03_buffer_load.py) | Buffer loads | 305.228615 | 591.268802 |
| [04_async_copy.py](gemm/04_async_copy.py) | Direct copy from global memory to LDS | 172.488217 | 591.883725 |

## Profiling[WIP] not to be taken seriously yet

From the repository root, run the benchmark under `rocprofv3` to collect a
kernel trace and summary statistics in CSV format.

```sh
/opt/rocm/core-10.1/bin/rocprofv3 \
  --kernel-trace --stats --output-format csv \
  -d /root/profile_results/bench_fixed -- \
  python3 amd/gemm/bench.py --kernel 01_naive --K 4096 --dtype fp16 --rocprof
```

## Kernel notes

### Buffer loads

AMD buffer operations check bounds in hardware, which avoids explicit branches
for masked loads in GEMM kernels. They can also reduce the number of vector
registers needed for addresses by keeping a shared base separate from each
lane's 32-bit offset.

Two constraints affect their use:

- **Offset width.** A 32-bit byte offset can address a 4 GB range from the base.
- **Compiler range analysis.** The compiler's range analysis works reliably
  when the whole tensor is smaller than 2 GB.

### Async copy and local data share (LDS)

The async-copy kernel loads data directly from global memory into LDS. This
saves the VGPRs used to stage the data and removes the corresponding
`ds_write` instructions that would copy values from those registers into LDS.
A subsequent `ds_read` loads data from LDS into each lane's VGPRs for arithmetic.

On the MI300X, a direct global-to-LDS load transfers up to 4 bytes per lane per
instruction, enough for two FP16 or BF16 values. With all 64 lanes active, one
instruction can copy 256 bytes into LDS. This transfer width motivates the
kernel's `[1, 2]` per-thread layout and helps explain its lower throughput in
the benchmark above.

## Registers and execution on gfx942

### Vector and scalar registers

A wavefront contains 64 lanes. Scalar register values are shared within each
wavefront.

| Register | Meaning | Values held by one register in a wavefront |
| --- | --- | --- |
| VGPR | Vector general-purpose register | 64 separate 32-bit values, one per lane |
| SGPR | Scalar general-purpose register | One 32-bit value shared by all lanes |

For example, lanes can access different elements of an array using a shared base
address and a per-lane byte offset.

```text
Shared 64-bit base in s[0:1] = 1000

                    Lane 0   Lane 1   Lane 2   Lane 3   ...
Offsets in VGPR v0       0        4        8       12   ...
Memory addresses     1000     1004     1008     1012   ...
```

The 64-bit base occupies two SGPRs. One VGPR holds all 64 lanes' 32-bit
offsets. VGPRs also hold values loaded from matrices A and B and per-lane
intermediate results.

Materializing a full 64-bit address in each lane takes two VGPRs. Buffer
addressing computes `base + offset` using one VGPR for the offsets and SGPRs
for the buffer descriptor, which includes the base address and bounds.
Ordinary AMD global loads can also use a scalar base plus a 32-bit vector
offset. If the compiler already generates that form for `gl.load`, switching
to buffer operations may not reduce VGPR usage.

### Wavefronts and SIMD units

A wavefront is a group of threads. A SIMD unit executes one instruction across
multiple lanes, and its register file holds their VGPR values. For example,
`v2 = v0 + v1` adds each active lane's own values:

```text
Lane 0: v2[0] = v0[0] + v1[0]
Lane 1: v2[1] = v0[1] + v1[1]
...
```

On CDNA 3 (`gfx942`), each compute unit has four 16-wide SIMD units. A SIMD unit
processes a wavefront's 64 logical lanes in groups for ordinary vector
arithmetic. When data “returns from LDS to the SIMD,” it arrives in the SIMD
unit's VGPR file, ready for subsequent instructions.

| Vector operation | Examples |
| --- | --- |
| Floating-point arithmetic | Add, multiply, fused multiply-add |
| Integer arithmetic | Index calculations, offsets |
| Bit operations | AND, OR, XOR, shifts |
| Comparisons and selection | Compare values, choose results |
| Type conversions | Convert between supported numeric types |
