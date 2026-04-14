1. whats conditions constrain the number of threadsPerBlock and blocks?

```
max threads per block is 1024( we can divide into any dims as long as it doesnt exceed 1024)
max blocks is very very very large.
```

2. how is the theoretical occupancy rate calculated?

```
occupancy  = active warps per SM / max warps per SM
```

3. what is warp, and warp divergence?

```
warp is a group of 32 threads and execute the same instruction, basically all the threads.
warp divergence occurs when threads in the same warp take different execution paths due to a branch like if/else. both paths end up executing sequentially.
```

4. 