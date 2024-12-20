__global__ void mma_matmul_1_0(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_1_1(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_2_0(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_2_1(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_3_0(const half *A, const half *B, float *C, int M, int N, int K);
__global__ void mma_matmul_3_1(const half *A, const half *B, float *C, int M, int N, int K);
