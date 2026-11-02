# triton kernels

learning triton and gluon kernels.

## how to run

```
modal run triton/modal_runner.py --script my_kernel.py

# Choose the GPU when launching the runner.
MODAL_GPU=A100-80GB modal run triton/modal_runner.py --script 02_my_kernel.py
```

## GPU choices

| GPU argument | 
| --- | 
| `T4` | 
| `L4` | 
| `A10` | 
| `L40S` | 
| `A100`, `A100-40GB`, `A100-80GB` | 
| `H100`, `H100!`, `H200` | 
| `RTX-PRO-6000`, `B200` | 
| `B200+`, `B300` | 
