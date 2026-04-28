### hopper kernels

this is a series of matmul kernels to learn about hopper instructions.

all kernels benchmarked on H100 SXM.

ref - https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog


1. compile using `nvcc -std=c++17 -arch=compute_90a -code=sm_90a --generate-line-info -Xptxas -v matmul.cu -lcublas -lcuda -o matmul`
2. generate report using `ncu --set full -o matmul_full ./matmul`


currently on pause - switched to blackwell