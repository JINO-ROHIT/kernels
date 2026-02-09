### memory

# ref - https://www.aleksagordic.com/blog/matmul


the GPU memory system consists of:

1. device memory (VRAM) - off-chip DRAM physically separate from the GPU die but packaged together on the same board—implemented as stacked HBM. It hosts global memory (GMEM), per-thread "local" memory (register spill space), etc.

2. l2 cache - a large, k-way set-associative cache built from SRAM. It is physically partitioned into two parts; each SM connects directly to only one partition and indirectly to the other through the crossbar.

3. distributed shared memory (DSMEM). The pooled shared memories (SMEM) of a physically close group of SMs (a GPC).

4. l1 cache and shared memory
 - l1 cache. A smaller, k-way set-associative SRAM cache private to each SM.
 - shared memory (SMEM). Programmer-managed on-chip memory. SMEM and L1 share the same physical storage, and their relative split can be configured in software.

5. register file (RMEM). The fastest storage, located next to the compute units. Registers are private to individual threads. Compared to CPUs, GPUs contain far more registers, and the total RMEM capacity is of the same size as the combined L1/SMEM storage.

<img src="../assets/memory.png" alt="alt text" width="800"/>


as we move from device memory to the register, you can see bandwith go up, latency go down and size go down. so keep the freq accessed data close to the chip


note - *tensor memory accelerator(TMA)* - introduced with hopper, enables asynchronous data transfers between global memory and shared memory, as well as across shared memories within a cluster.


### compute

fundamental unit of compute is SM(streaming multiprocessor). each SM contains -

1. tensor cores - specialized units that execute matrix multiplications on small tiles at high throughput. large matrix multiplications are decomposed into many such tile operations, so leveraging them effectively is critical for reaching peak performance.
2. CUDA cores and SFUs. The so called "CUDA cores" execute standard floating-point operations such as FMA (fused multiply-add: c = a * b + c). Special Function Units (SFUs) handle transcendental functions such as sin, cos, exp, log, but also algebraic functions such as sqrt, rsqrt, etc.
3. Load/Store (LD/ST) units. Circuits that service load and store instructions, complementary to the TMA engine.
4. Warp schedulers. each SM contains schedulers that issue instructions for groups of 32 threads (called warps in CUDA). A warp scheduler can issue one warp instruction per cycle.

An SM can issue instructions from at most four warps simultaneously (i.e., 128 threads in true parallel execution at a given cycle).

However, an SM can host up to 2048 concurrent threads (64 warps). These warps are resident and scheduled in and out over time, allowing the hardware to hide memory/pipeline latency. ( in my gpu is just 1024)

In other words, instruction parallelism (how many threads start executing an instruction on a given cycle) is limited to 128 threads per SM at once (4 32-wide warp instructions), while concurrency (how many threads are tracked in the scheduler and eligible to run) extends to 2048 threads.


### speed of light

the max compute throughput of the gpu determined by the physical characteristics of the gpu.

(for my card : rtx 4060 ti)

number of tensor cores = number of SM * number of tensor core per SM
                       = 34 * 4
                       = 136 
one tensor core can do 64 FMA per cycle = 128 Flops (1 FMA = 2 Flops)

peak throughput = maximum clock frequency x number of tensor cores x FLOPs per tensor core per cycle
                = 2595 MHz x 136 x 128
                = 45.3 TFLOPs (fp16 tensor)


a thread block should contain at least 4 warps (128 threads).
Why?
- a thread block is resident on a single SM.
- each SM has 4 warp schedulers so to fully utilize the hardware, you don't want them sitting idle.


1. global memory(gmem)

here the access patterns matter, because of the physics of the dram cells. when people say “GMEM coalescing is very important”, this is what they mean: threads should access contiguous memory locations to minimize the number of DRAM rows touched.

2. shared memory(smem)

these are made of sram cells. smem is organized into 32 banks, each bank 32 bits wide (4 bytes).

SMEM can serve data from all 32 banks (128B) in a single cycle — but only if one rule is respected:

Threads in a warp must not access different addresses within the same bank. Otherwise, those requests are serialized across multiple cycles.

This situation is known as a bank conflict. If N threads access different addresses of the same bank, the result is an N-way bank conflict and the warp’s memory request takes N cycles to complete.


#TO-DO - learn to read ptx