# Fast Tensor Core matul on the Ada Architecture

Code for the blog post [spatters.ca/mma-matmul](https://spatters.ca/mma-matmul). 

## Benchmark results
For a M=N=4096 FP16/32 Matrix Multiplication

| Kernel | Execution Time | TFLOP/s &nbsp; &nbsp; | % cuBLAS &nbsp; &nbsp;  | % 4090 peak &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| ---    | ---     | ---                 | ---                  | --- |
| cublasGemmEx | 895 us | 153.6 | 100% |  93.0%  |
| Kernel 1.0: Naive mma | 4680 us | 29.4 | 19.1% | 17.8%   |
| Kernel 1.1: Naive + 2x tiling| 2400 us | 57.3 | 37.3% | 34.7%   |
| Kernel 2.0: Permuted shmem | 1080 us | 127.3 | 82.9% | 77.0% |
| Kernel 2.1: Permuted shmem + register tweak | 1030 us | 133.4 | 86.9%| 80.8% |
| Kernel 3.0: N-stage async pipeline | 1000 us | 137.4 | 89.5% | 83.2% |
| Kernel 3.1: N-stage + 4x tiling | 895 us | 153.6 | 100% | 93.0%   |

## Build instructions
To compile the kernels run the following
```bash
nvcc -c -arch=sm_89 -Xptxas="-v" kernel_1.cu
nvcc -c -arch=sm_89 -Xptxas="-v" kernel_2.cu
nvcc -c -arch=sm_89 -Xptxas="-v" kernel_3.cu
nvcc -c -arch=sm_89 -Xptxas="-v" runner.cu && nvcc -lcublas -o runner runner.o kernel_1.o kernel_2.o kernel_3.o
```

## Running the benchmarks
Benchmarks were run on CUDA Toolkit Version 12.4, CUDA Driver Version 550.67. To run the benchmarks:

```bash
sudo nvidia-smi -pm ENABLED
sudo nvidia-smi --lock-gpu-clocks=2520
sudo nvidia-smi --lock-memory-clocks=10501 
ncu -s 5 -k regex:'^(?!shmem*)' --clock-control none --print-summary per-gpu ./a.out 31 # 31 -> Kernel 3.1
sudo nvidia-smi --reset-gpu-clocks
sudo nvidia-smi --reset-memory-clocks
sudo nvidia-smi -pm DISABLED
```

