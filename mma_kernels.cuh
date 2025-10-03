#include <cuda.h>
__global__ void mma_matmul_1_0(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_1_1(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_2_0(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_2_1(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_3_0(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_3_1(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_3_2(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_3_3(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_3_4(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void wgmma_matmul_4_0(const 	__grid_constant__ CUtensorMap tensor_map_A, const 	__grid_constant__ CUtensorMap tensor_map_B, const half *A, const half *B, float *C, int M, int N, int K);
