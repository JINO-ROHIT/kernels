1. naive vadd - each thread loads an element from gmem and performs vadd and then writes back.

```
Kernel execution time: 3332.2394 us
Memory throughput: 241.67 GB/s
```

2. tiled vadd - create (1,4) tiles and then perform vadd.

```
Kernel execution time: 3267.8214 us
Memory throughput: 246.44 GB/s
```

3. tv layout - use a tv layout with a thread value layout hierarchy to perform vadd

```
Kernel execution time: 3152.6196 us
Memory throughput: 255.44 GB/s
```

4. tranposed - tranposed the 2nd mode to keep coalesced(didnt help?)

```
Kernel execution time: 3129.9765 us
Memory throughput: 257.29 GB/s
```

5. rmem - load elements from gmem to register memory and then perform vadd(this has coalescing issues)

```
Kernel execution time: 27340.9863 us
Memory throughput: 29.45 GB/s
```

6. coalesced rmem - fixed the coalescing issue.

```
Kernel execution time: 4633.4158 us
Memory throughput: 173.80 GB/s
```

7. use the element dtype instead of fixed float

```
Kernel execution time: 4606.7599 us
Memory throughput: 174.81 GB/s
```

8. use a 128 bit copy width ( this involves a lot of manual composition to create the thread layout)

```
Kernel execution time: 3076.3004 us
Memory throughput: 261.78 GB/s
```

9. also use a 128 bit copy width but uses tiled_copy_tv + partition_S that automatically handles the thread value mapping without composition.

```
Kernel execution time: 3080.1920 us
Memory throughput: 261.45 GB/s
```
