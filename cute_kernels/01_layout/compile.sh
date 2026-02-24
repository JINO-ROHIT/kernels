nvcc -std=c++17 -I /home/jrohit/Desktop/projects/kernels/cutlass/include \
-I /home/jrohit/Desktop/projects/kernels/cutlass/tools/util/include \
/home/jrohit/Desktop/projects/kernels/cute_kernels/01_layout/layout.cu  -o /home/jrohit/Desktop/projects/kernels/cute_kernels/01_layout/layout \
&& ./layout && rm layout

nvcc -std=c++17 -I /home/jrohit/Desktop/projects/kernels/cutlass/include \
-I /home/jrohit/Desktop/projects/kernels/cutlass/tools/util/include \
/home/jrohit/Desktop/projects/kernels/cute_kernels/01_layout/01_column_major.cu  -o /home/jrohit/Desktop/projects/kernels/cute_kernels/01_layout/test \
&& ./test && rm test

nvcc -std=c++17 -I /home/jrohit/Desktop/projects/kernels/cutlass/include \
-I /home/jrohit/Desktop/projects/kernels/cutlass/tools/util/include \
/home/jrohit/Desktop/projects/kernels/cute_kernels/01_layout/03_index.cu  -o /home/jrohit/Desktop/projects/kernels/cute_kernels/01_layout/03_index \
&& ./03_index && rm 03_index

nvcc -std=c++17 -I /home/jrohit/Desktop/projects/kernels/cutlass/include \
-I /home/jrohit/Desktop/projects/kernels/cutlass/tools/util/include \
/home/jrohit/Desktop/projects/kernels/cute_kernels/01_layout/06_coalesce.cu  -o /home/jrohit/Desktop/projects/kernels/cute_kernels/01_layout/06_coalesce \
&& ./06_coalesce && rm 06_coalesce

nvcc -std=c++17 -I /home/jrohit/Desktop/projects/kernels/cutlass/include \
-I /home/jrohit/Desktop/projects/kernels/cutlass/tools/util/include \
/home/jrohit/Desktop/projects/kernels/cute_kernels/01_layout/07_complement.cu  -o /home/jrohit/Desktop/projects/kernels/cute_kernels/01_layout/07_complement \
&& ./07_complement && rm 07_complement

