nvcc -arch=sm_89 -std=c++17 -I /home/jrohit/Desktop/projects/kernels/cutlass/include \
-I /home/jrohit/Desktop/projects/kernels/cutlass/tools/util/include \
/home/jrohit/Desktop/projects/kernels/cute_kernels/04_softmax/naive.cu  -o /home/jrohit/Desktop/projects/kernels/cute_kernels/04_softmax/naive \
&& ./naive && rm naive


nvcc -arch=sm_89 -std=c++17 -I /home/jrohit/Desktop/projects/kernels/cutlass/include \
-I /home/jrohit/Desktop/projects/kernels/cutlass/tools/util/include \
/home/jrohit/Desktop/projects/kernels/cute_kernels/04_softmax/online.cu  -o /home/jrohit/Desktop/projects/kernels/cute_kernels/04_softmax/online \
&& ./online && rm online


nvcc -arch=sm_89 -std=c++17 -I /home/jrohit/Desktop/projects/kernels/cutlass/include \
-I /home/jrohit/Desktop/projects/kernels/cutlass/tools/util/include \
/home/jrohit/Desktop/projects/kernels/cute_kernels/04_softmax/warp_reduce.cu  -o /home/jrohit/Desktop/projects/kernels/cute_kernels/04_softmax/warp_reduce \
&& ./warp_reduce && rm warp_reduce
