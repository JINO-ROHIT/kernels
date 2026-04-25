## a deep dive on hopper architecture

this article should help you get started with hopper architecture. we will mostly look at h100 as an example.

- h100 has 2 form factors - pcie and sxm. sxm has higher power (700w) and gives better performance than pcie (300w).
- 80gb HBM3 memory with peak bandwith of 3.35 tb/s
- 132 SM
- 528 tensor cores (4 per SM)
- and a lot of async features.


### gpu organization

the full die is divided into -

1. nvidia gigathread engine - distributes thread blocks (ctas) to sms
2. 8 GPC (graphics processing clusters) - each containing 18 sms
3. HBM3 memory stacks - 5 active stacks (6th disabled for manufacturing yield)
4. HBM3 memory controllers - 10 independent 512-bit controllers
5. pcie 5.0 host interface - the highway that connects cpu to gpu
6. nvlink switches/ports/hub - high speed gpu-to-gpu fabric
7. L2 cache slices - 50 mb total, partitioned across gpcs



### gigathread engine

the gigathread engine is the hardware dispatcher for kernel launches. it -
(cta is also thread block)
1. tracks ctas that are pending, running, or finished
2. assigns ctas to SM when they have available capacity
3. an SM can only hold so many thread blocks at once depending on how many registers and how much shared memory the kernel uses. the gigathread engine knows this and won't overschedule.


### TMA (tensor memory accelerator)

before TMA (pre-hopper), loading data from global memory to shared memory was the programmer's problem. every thread had to compute its own index, figure out its row offset, handle boundary checks, and then issue the load. for a 2d tile this means you'd write something like `smem[threadIdx.y][threadIdx.x] = gmem[row * stride + col]` and every thread in the warp is doing this in lockstep, which wastes execution cycles that could be spent on math.

with TMA, you describe the data movement once using a tensor descriptor which is a small metadata object you create on the cpu that says "here's the base pointer, here's the shape, here's the stride between rows." then at runtime, a single thread in the warp fires off one tma instruction: "go fetch a 128x64 tile from coordinate (x, y) and drop it into this shared memory address." that's it. the other 31 threads in the warp don't need to participate at all and they're immediately free to do something else while the tma hardware handles everything in the background.


### 4th generation tensor core

1. 4 tensor cores per SM, 528 total across the chip.
2. introduces wgmma (warpgroup matrix multiply accumulate) — in previous generations (ampere and before), a single warp of 32 threads issued a `wmma` instruction to do a small matrix multiply. the operands were small enough to fit in registers owned by those 32 threads. hopper changes this: now a warp group where 4 warps working together, 128 threads total  issues a single `wgmma` instruction to do a much larger matrix multiply. this matters because bigger tiles mean fewer round trips to shared memory, which means better arithmetic intensity, which means you're spending more time doing math and less time waiting on memory.
3. native fp8 support  fp8 is a new lower-precision format (8 bits vs fp16's 16 bits). you fit twice as many values in the same register space and get roughly 2x the throughput on tensor core ops compared to fp16. 



### HBM memory

1. 80 GB capacity, 3.35 TB/s bandwidth
2. divided across 5 active stacks (6th disabled for manufacturing yield)
3. 5120 bit bus width(10 * 512) that enables TMA to move 128 bytes per transaction
4. Connected via 10 independent 512-bit memory controllers

### L2 Cache

1. 50 MB total, split into two 25 MB partitions
2. the cache line is 128 bytes and the sector is 32 bytes. what does this mean? 
when the GPU looks for data, it checks if the Address is present in the cache. It doesn't track every single byte; it tracks "tags" for 128-byte chunks.
If you want to store even 1 byte in the L2 cache, you must "allocate" a full 128-byte line to hold it.

the Sector is the unit of movement. A 128-byte cache line is divided into four 32-byte sectors.
In older GPUs, if you missed the cache, you had to fetch the entire line from HBM.
In modern GPUs, the L2 can be "partially filled." If you only need data in the first 10 bytes, the hardware only fetches Sector 0 (32 bytes) from HBM3. The other three sectors in that line remain "invalid" or empty until they are actually needed. you save tons of HBm bandwith
3. Uncoalesced accesses can cause sector explosion. what does this mean lol?

Imagine a Warp (32 threads) where each thread asks for a 4-byte float. thats 32 threads x 4  bytes = 128  bytes.
These 128 bytes sit perfectly inside one cache line.The hardware performs one L2 request, fetches four sectors, and everyone is happy.

The Bad Scenario
Imagine the same 32 threads, but they are asking for data that is spread out (e.g., thread 0 asks for address 0, thread 1 asks for address 200, thread 2 asks for address 400). Even if each thread only wants 4 bytes, each of those addresses likely falls into a different cache line. The L2 now has to manage 32 different "tags" and fetch at least one 32-byte sector for each thread.
this means instead of moving 128 bytes of data to satisfy the Warp, the hardware moves 32 requests x 32  bytes = 1,024  bytes. You are now using 8x more bandwidth than necessary.


### GPC - Graphics Processing Cluster

1. it a group of 18 SMs
2. Each GPC has its own dedicated chunk of L2 cache
4. Shared Memory or the (SRAM) is private to a single SM. SM-A can directly read from SM-B. 
But with hopper,  Within a GPC, the SMs are connected by a high-speed inter-SM fabric. This allows an SM to directly "reach into" the Shared Memory of another SM in the same cluster.

### NVLink 4.0

PCIe 5.0 is more like a "slow" bridge between the CPU and the GPU, which NVLink is a "super-highway" that connects GPUs directly to other GPUs.

1. 18 NVLink 4.0 lanes that gives 900 GB/s total GPU-to-GPU bandwidth
2. Organized as 9 sub-links, each providing 100 GB/s bidirectional (50 GB/s per direction)
3. Can directly connect up to 8 GPUs
4. With NVSwitch fabric this scales up to 256 GPUs

### SM

the SM (streaming multiprocessor) is the main execution unit. each kernel's thread blocks get assigned to sms, and all the actual compute happens here. each sm contains:

- fp32 cuda cores - for standard floating point ops like add, multiply, fma
- int32 units - integer math and memory address calculations (can run simultaneously with fp32 on hopper)
- fp64 units - double precision math (h100 has full fp64 throughput for hpc workloads)
- 4th gen tensor cores - for matrix multiply operations (wgmma)
- shared memory / l1 data cache - an unified on-chip SRAM which you can split between SMEM and L1
- L1 instruction cache 
- L2 instruction cache 
- warp schedulers - select which warp to issue each cycle
- dispatch units - route the selected instruction to the right execution pipeline
- register file - 64k 32-bit registers per SM, private to each thread

each SM is divided into 4 sub-partitions (smsps). each smsp has its own warp scheduler, dispatch unit, register file partition, and execution pipelines. the 4 smsps share the l1/shared memory.


### special function units (sfus)

16 sfus per SM (4 per smsp), each capable of 1 instruction per cycle , that is 16 ops/cycle per SM.

sfus handle math like sin, cos, log, exp, sqrt, reciprocal, and rsqrt. these are operations you can't build out of a few adds and multiplies. sfus use a combination of approximations to compute these in 1-4 cycles at the cost of slightly reduced precision.



### load/store units (lsus)

32 lsus per sm (8 per smsp). these handle all memory traffic in and out of the sm:

- loads - fetch data from l1 cache, l2 cache, or global memory (hbm)
- stores - write data back to l1/l2/global memory
- atomics - read-modify-write operations like `atomicAdd`

when a warp executes a load instruction, all 32 threads have a memory address they want to read. the LSU doesn't issue 32 separate requests, instead it inspects all 32 addresses and tries to merge them into the minimum number of cache line sized transactions. if the addresses are all within the same 128-byte aligned region, that's one transaction. if they're spread across 32 different cache lines, that's 32 transactions — and you pay 32x the latency and bandwidth.


### shared memory & l1 cache

256 kb of unified on-chip sram per sm, shared between l1 data cache and programmable shared memory. you configure the split at the kernel level.
shared memory bandwidth is 33 tb/s per sm, this is ~10x HBM bandwidth.

shared memory is divided into 32 banks, each 4 bytes wide. consecutive 4-byte words map to consecutive banks. a warp can read 32 different addresses simultaneously as long as they hit 32 different banks that's one cycle, conflict-free.

bank conflicts happen when two or more threads in a warp access different addresses in the same bank. the hardware serializes those accesses  2 threads hitting the same bank = 2 cycles, 8 threads = 8 cycles

TMA async copies bypass the bank conflict problem entirely when loading from global memory into shared memory, since they write to smem directly without going through the warp's execution pipeline.