nvcc -arch=sm_89 -std=c++17 -I /home/jrohit/Desktop/projects/kernels/cutlass/include \
-I /home/jrohit/Desktop/projects/kernels/cutlass/tools/util/include \
/home/jrohit/Desktop/projects/kernels/cute_kernels/03_vector_add/naive.cu  -o /home/jrohit/Desktop/projects/kernels/cute_kernels/03_vector_add/naive \
&& ./naive

nvcc -arch=sm_89 -std=c++17 -I /home/jrohit/Desktop/projects/kernels/cutlass/include \
-I /home/jrohit/Desktop/projects/kernels/cutlass/tools/util/include \
/home/jrohit/Desktop/projects/kernels/cute_kernels/03_vector_add/cute_version.cu  -o /home/jrohit/Desktop/projects/kernels/cute_kernels/03_vector_add/cute_version \
&& ./cute_version

