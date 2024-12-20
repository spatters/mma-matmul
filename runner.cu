#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include "mma_kernels.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define ceilDiv(x, y) (((x) + (y) - 1) / (y))

#define SIZE 4096
#define GLOBAL_K SIZE
#define GLOBAL_M SIZE
#define GLOBAL_N SIZE
#define K_BLOCK 32 
#define REPS 50
#define WARMUP_REPS 5
#define TOTAL_REPS REPS + WARMUP_REPS


// Reference fp16/32 CUDA MATMUL Kernel
__global__ void shmem_matmul(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ half As[K_BLOCK][K_BLOCK];
  __shared__ half Bs[K_BLOCK][K_BLOCK];
  int idx = threadIdx.x + blockDim.x*blockIdx.x; 
  int idy = threadIdx.y + blockDim.y*blockIdx.y;

  if ((idx < N) && (idy < M)){
    float temp = 0;
    for (int i = 0; i < K/K_BLOCK; i++) {
      As[threadIdx.y][threadIdx.x] = A[idy*K + i*K_BLOCK + threadIdx.x];
      Bs[threadIdx.y][threadIdx.x] = B[(i*K_BLOCK + threadIdx.y) * N + idx];
      __syncthreads();

      for (int k = 0; k < K_BLOCK; k++)
      	temp += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]); 
      __syncthreads();

    }
    C[idy*N+idx] = temp;
  }
}

float uniform_rand() {
  return (float)rand() / (float)RAND_MAX;
}

void host_transpose(half *src, half *dst, int M, int N) {
  for (int i=0; i < M; i++) {
    for (int j=0; j < N; j++) {
      dst[j*M + i] = src[i*N + j];
    }
  }
}

void init_uniform_half(half **host, half **device, int M, int N) {
  *host = new half[M * N];
  cudaMalloc(device, M * N * sizeof(half));
  for (int i=0; i < M*N; i++) {
    (*host)[i] = __float2half(uniform_rand());
  }
  cudaMemcpy(*device, *host, M * N * sizeof(half), cudaMemcpyHostToDevice);
}

bool in_array(int val, int* vals, int N) {
  for (int i=0; i<N; i++) {
    if (vals[i] == val)
      return true;
  }
  return false;
}

void init_zero_float(float** host, float **device, int M, int N) {
  *host = new float[M * N];
  cudaMalloc(device, M * N * sizeof(float));
  for (int i=0; i < M*N; i++) {
    (*host)[i] = 0.;
  }
  cudaMemcpy(*device, *host, M * N * sizeof(float), cudaMemcpyHostToDevice);
}

void run_cublas_kernel(int numReps, half *A, half *B, float *C, int M, int N, int K) {
  cublasStatus_t stat;   // cuBLAS functions status
  cublasHandle_t handle; // cuBLAS context
  stat = cublasCreate(&handle); // initialize CUBLAS context
  cudaDataType_t aType = CUDA_R_16F;
  cudaDataType_t bType = CUDA_R_16F;
  cudaDataType_t cType = CUDA_R_32F;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  float alpha = 1.0;
  float beta = 0.0;
  for (int i=0; i<numReps; i++) {
    stat = cublasGemmEx(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        N, M, K, 
        (void*)&alpha, 
        (void*)B, bType, K, 
        (void*)A, aType, M, 
        (void*)&beta, 
        (void*)C, cType, M, 
        computeType, CUBLAS_GEMM_DEFAULT);
  }
  const char* statStr = cublasGetStatusName(stat);
  printf("cublasSgemm status %s.\n", statStr);
}

void run_mma_kernel(int kernelNum, int numReps, half *A, half *B, half *B_T, float *C, int M, int N, int K) {
  dim3 mma_block(16, 16);
  dim3 mma_grid;
  for (int i=0; i<numReps; i++) {
    switch (kernelNum) {
      case 10:
        mma_grid.x = ceilDiv(GLOBAL_M, 32);
        mma_grid.y = ceilDiv(GLOBAL_N, 32);
        mma_matmul_1_0<<<mma_grid, mma_block>>>(A, B, C, GLOBAL_M, GLOBAL_N, GLOBAL_K);
        break;
      case 11:
        mma_grid.x = ceilDiv(GLOBAL_M, 64);
        mma_grid.y = ceilDiv(GLOBAL_N, 64);
        mma_matmul_1_1<<<mma_grid, mma_block>>>(A, B, C, GLOBAL_M, GLOBAL_N, GLOBAL_K);
        break;
      case 20:
        mma_grid.x = ceilDiv(GLOBAL_M, 64);
        mma_grid.y = ceilDiv(GLOBAL_N, 64);
        mma_matmul_2_0<<<mma_grid, mma_block>>>(A, B_T, C, GLOBAL_M, GLOBAL_N, GLOBAL_K);
        break;
      case 21:
        mma_grid.x = ceilDiv(GLOBAL_M, 64);
        mma_grid.y = ceilDiv(GLOBAL_N, 64);
        mma_matmul_2_1<<<mma_grid, mma_block>>>(A, B_T, C, GLOBAL_M, GLOBAL_N, GLOBAL_K);
        break;
      case 30:
        mma_grid.x = ceilDiv(GLOBAL_M, 64);
        mma_grid.y = ceilDiv(GLOBAL_N, 64);
        mma_matmul_3_0<<<mma_grid, mma_block>>>(A, B_T, C, GLOBAL_M, GLOBAL_N, GLOBAL_K);
        break;
      case 31:
        mma_grid.x = ceilDiv(GLOBAL_M, 128);
        mma_grid.y = ceilDiv(GLOBAL_N, 128);
        mma_matmul_3_1<<<mma_grid, mma_block>>>(A, B_T, C, GLOBAL_M, GLOBAL_N, GLOBAL_K);
        break;
    }
  }
}

int main(int argc, char **argv){
  if (argc != 2) {
    printf("Please supply kernel number as argument.\n");
    printf("Valid Kernel Numbers: 0, 10, 11, 20, 21, 30, 31.\n");
    exit(EXIT_FAILURE);
  }
  int kernelNum = atoi(argv[1]);
  int validKernels[7] = {0, 10, 11, 20, 21, 30, 31};
  bool validKernelNum = in_array(kernelNum, validKernels, 7);
  if (not validKernelNum) {
    printf("Kernel Num: %d not recognized. Valid Kernel Nums: 0, 10, 11, 20, 21, 30, 31.\n", kernelNum);
    exit(EXIT_FAILURE);
  }
  printf("Running Kernel %d.%d\n", kernelNum/10, kernelNum%10);
  float *h_C1, *h_C2, *d_C1, *d_C2;
  half  *h_A, *h_B, *h_B_T, *d_A, *d_B, *d_B_T;

  init_uniform_half(&h_A, &d_A, GLOBAL_M, GLOBAL_K);
  init_uniform_half(&h_B, &d_B, GLOBAL_K, GLOBAL_N);
  init_zero_float(&h_C1, &d_C1, GLOBAL_M, GLOBAL_N);
  init_zero_float(&h_C2, &d_C2, GLOBAL_M, GLOBAL_N);
  cudaCheckErrors("cudaMemcpy malloc / H2D failure");

  // Launch reference kernel
  dim3 block(K_BLOCK, K_BLOCK);  
  dim3 grid(ceilDiv(GLOBAL_N, block.x), ceilDiv(GLOBAL_M, block.y));
  shmem_matmul<<<grid, block>>>(d_A, d_B, d_C1, GLOBAL_M, GLOBAL_N, GLOBAL_K);
  cudaCheckErrors("reference kernel failure");
  cudaMemcpy(h_C1, d_C1, GLOBAL_M * GLOBAL_N * sizeof(float), cudaMemcpyDeviceToHost);

  // transpose B as Kernel 2.0 on require B to be column-major
  h_B_T = new half[GLOBAL_N * GLOBAL_K];
  host_transpose(h_B, h_B_T, GLOBAL_K, GLOBAL_N);
  cudaMalloc(&d_B_T, GLOBAL_N * GLOBAL_K * sizeof(half));
  cudaMemcpy(d_B_T, h_B_T, GLOBAL_N * GLOBAL_K * sizeof(half), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy malloc / H2D failure");

  // run kernel in loop to profile
  if (kernelNum == 0) {
    run_cublas_kernel(TOTAL_REPS, d_A, d_B, d_C2, GLOBAL_M, GLOBAL_N, GLOBAL_K);
  }
  else {
    run_mma_kernel(kernelNum, TOTAL_REPS, d_A, d_B, d_B_T, d_C2, GLOBAL_M, GLOBAL_N, GLOBAL_K);
  }

  cudaCheckErrors("mma kernal failure");
  cudaMemcpy(h_C2, d_C2, GLOBAL_M * GLOBAL_N *sizeof(float), cudaMemcpyDeviceToHost);

  // check output vs reference kernel
  float diff = 0.;
  float abs_diff = 0.;
  float max_abs_diff = 0.;
  float avg_diff = 0.;
  float avg_abs_diff = 0.;
  float avg_out_val = 0.;

  for (int i = 0; i < GLOBAL_M * GLOBAL_N; i++) {
    diff = ((h_C1[i]) - h_C2[i]);
    abs_diff = abs(diff);
    avg_abs_diff += abs_diff / (float)(GLOBAL_M * GLOBAL_N);
    avg_diff += diff / (float)(GLOBAL_M * GLOBAL_N);
    avg_out_val += h_C1[i] / (float)(GLOBAL_M * GLOBAL_N);
    if (abs_diff > max_abs_diff)
      max_abs_diff = abs_diff;
  }
  printf("max abs diff %f, avg abs diff %f, avg diff: %f, avg output value; %f\n", max_abs_diff, avg_abs_diff, avg_diff, avg_out_val);
  return 0;
}
