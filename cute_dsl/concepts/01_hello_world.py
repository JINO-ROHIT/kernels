#ref - https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/hello_world.ipynb

import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("Hello world")

@cute.jit
def hello_world():

    kernel().launch(
        grid=(1, 1, 1), 
        block=(32, 1, 1),  
    )


# initialize CUDA context for launching a kernel with error checking
cutlass.cuda.initialize_cuda_context()

# jit - compiles and runs the code immediately
print("Running hello_world()...")
hello_world()

# compile first (useful if you want to run the same code multiple times)
print("Compiling...")
hello_world_compiled = cute.compile(hello_world)

# dump PTX/CUBIN files while compiling
from cutlass.cute import KeepPTX, KeepCUBIN

print("Compiling with PTX/CUBIN dumped...")
hello_world_compiled_ptx_on = cute.compile[KeepPTX, KeepCUBIN](hello_world)

print("Running compiled version...")
hello_world_compiled()