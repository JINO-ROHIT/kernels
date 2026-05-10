## ptx

a series of experiments to learn to read and write ptx.

ref - https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-isa-version-9-2


### ISA
an instruction set architecture (ISA) is a specification of the instructions a processor can execute, their format, the behavior of these instructions, and their binary encoding. The human-readable form of an ISA is known as the assembly.

On NVIDIA GPUs, the native ISA is called SASS(shader assembly).

PTX is NVIDIA's virtual ISA, an instruction set for an abstract GPU. PTX code is not executed directly; instead, it is compiled by ptxas into the native ISA (SASS).

### how does compilation happen

when you have a CUDA file with both device and host code, and this file is compiled using the NVIDIA CUDA compiler NVCC , the source code is split into GPU code and CPU code. The GPU code is sent to the GPU compiler, and the CPU code is sent to the host compiler. The host compiler is not part of NVCC. NVCC invokes the host compiler passed in on the command line, or the default compiler on the system.

1. The nvcc compiler first translates your CUDA code to PTX. 
2. Then ptxas (the PTX assembler) converts PTX to SASS (Shader Assembly), which is the actual machine code for your GPU architecture. 

```
CUDA C++ --> PTX ---> SAAS --> execute the kernel
```


### Computing power

all NVIDIA GPUs have a version identifier, either a compute capability or a CC number. each compute capability has a major version number and a minor version number. for example, compute capability 8.6 has a major version number of 8 and a minor version number of 6.

like other processors, NVIDIA GPUs also have specific ISAs. different generations of GPUs have different ISAs. these ISAs are identified by a version number that corresponds to the GPU's computing power. when compiling the binary (cubin), it is compiled for that specific computing power.

for example, the GeForce and RTX GPUs of the NVIDIA Ampere generation have a computing power of 8.6, and their cubin version is sm_86. all cubin versions are formatted as sm_XY, where X and Y correspond to the major and minor numbers of the computing power.

different generations of NVIDIA GPUs, and even different products within the same generation, may have different ISAs. this is part of the reason for using PTX.


### GPU code compatibility

NVIDIA GPUs are binary-compatible in their major compute capability versions, provided the minor version is the same or higher. This means that sm_86a compiled cubin can be loaded onto any sm_8xGPU with x greater than or equal to 6.

For example, a cubin compiled for sm_86 (such as the NVIDIA RTX A4000) can also be loaded and run on sm_89 (such as the NVIDIA RTX 4000 Ada Generation). However, it will not load on devices with a compute capability of 8.0 because the minor version of that GPU compute capability is lower than the minor version of cubin.

In major compute capability versions, NVIDIA GPUs are not binary compatible. The sm_86compiled cubin will not load and run on GPUs version 9.0 ( NVIDIA Hopper architecture ) or later.



A PTX statement is either a directive or an instruction. 

