#include <cuda.h>
#include <cuda/barrier>
#include <mma.h>
#include <stdio.h>
#include <cuda_fp16.h>
#define N_STAGES 3

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;


__forceinline__
__device__ uint64_t matrix_descriptor_encode(uint32_t x) {
  return (x & 0x3FFFF) >> 4;
}


__forceinline__
__device__ uint64_t update_start_addr(uint64_t base_descriptor, uint32_t shmem_start_addr) {
  base_descriptor &= (~(1ULL << 14));
  base_descriptor |= matrix_descriptor_encode(shmem_start_addr);
  return base_descriptor;
}


__forceinline__
__device__ uint64_t get_matrix_descriptor(uint32_t shmem_start_addr, uint32_t LBO, uint32_t SBO, uint32_t swizzle_mode) {
  uint64_t base_descriptor = 0;
  base_descriptor |= matrix_descriptor_encode(shmem_start_addr);
  base_descriptor |= (matrix_descriptor_encode(LBO) << 16);
  base_descriptor |= (matrix_descriptor_encode(SBO) << 32);
  // BASE OFFSET set to 0
  base_descriptor |= (matrix_descriptor_encode(swizzle_mode) << 62);
  return base_descriptor;
}


//__forceinline__
__device__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::);
}


__forceinline__
__device__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
}


__forceinline__
__device__ void wgmma_wait_all() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(0));
}


__forceinline__
__device__ void cp_async(uint4 *dstAddr, const uint4 *srcAddr) {
  unsigned ptxDstAddr = __cvta_generic_to_shared(dstAddr);
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
      :: "r"(ptxDstAddr),
      "l"(srcAddr),
      "n"(16));
}


__forceinline__ 
__device__ void load_matrix_x4(unsigned *destReg, uint4 *srcAddr) {
  unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(destReg[0]), "=r"(destReg[1]), "=r"(destReg[2]), "=r"(destReg[3])
      :  "r"(ptxSrcAddr)
      );
}

__forceinline__ 
__device__ void load_matrix_x2(unsigned *destReg, uint4 *srcAddr) {
  unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
  asm volatile(
      "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(destReg[0]), "=r"(destReg[1])
      :  "r"(ptxSrcAddr)
      );
}

__forceinline__ 
__device__ void mma_m16n8k16(const unsigned *A, const unsigned *B, float *C, float *D) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
      "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
     );
}

__forceinline__ 
__device__ void mma_m16n8k16_f16(const unsigned *A, const unsigned *B, unsigned *C, unsigned *D) {
  asm (
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(D[0]), "=r"(D[1])
      :
      "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "r"(C[0]), "r"(C[1])
      );
}

__forceinline__ 
__device__ void wgmma_m64n64k16(const uint64_t a_desc, const uint64_t b_desc, float *D) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7,"
       "%8, %9, %10, %11, %12, %13, %14, %15,"
       "%16, %17, %18, %19, %20, %21, %22, %23,"
       "%24, %25, %26, %27, %28, %29, %30, %31},"
       "%32, %33,"
       "%34,"
       "%35, %36,"
       "%37, %38;\n"
      : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]), "+f"(D[6]), "+f"(D[7]),
        "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]),
        "+f"(D[16]), "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]), "+f"(D[22]), "+f"(D[23]),
        "+f"(D[24]), "+f"(D[25]), "+f"(D[26]), "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31])
      :
      "l"(a_desc), "l"(b_desc),
      "n"(1),
      "n"(1), "n"(1)
      "n"(0), "n"(0)
      );
}

// Kernel 4.0: WAGMI
__global__ void wgmma_matmul_4_0(const 	__grid_constant__ CUtensorMap tensor_map_A, const 	__grid_constant__ CUtensorMap tensor_map_B, const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ __align__(1024) half As[64*32];
  __shared__ __align__(1024) half Bs[64*32];
  float dReg[32] = {0.0f};

  // Initialize shared memory barrier
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier barA;
  __shared__ barrier barB;

  int blockRowStart = blockIdx.y*64;
  int blockColStart = blockIdx.x*64;

  // warp layout is 2 x 4
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int warpOffsetA = 16 * (warpID / 4);
  int warpOffsetB = 8 * (warpID % 4);
  uint64_t a_desc, b_desc;
  //constexpr int LBO = 1;
  //constexpr int SBO = 512;
  //constexpr int swizzle_mode = 2; //  64B swizzle
  constexpr int LBO = 128;
  constexpr int SBO = 256;
  constexpr int swizzle_mode = 0; //  64B swizzle



  if (threadIdx.x == 0) {
    init(&barA, blockDim.x * blockDim.y);
    init(&barB, blockDim.x * blockDim.y);
    cde::fence_proxy_async_shared_cta();
  }

   __syncthreads();
  barrier::arrival_token tokenA;
  barrier::arrival_token tokenB;
  for (int k=0; k<K; k+=32) {
    if (threadIdx.x == 0) {
      // Initiate bulk tensor copy from global to shared memory,
      //cde::cp_async_bulk_tensor_2d_global_to_shared(&As, &tensor_map_A, k, blockRowStart, barA);
      cde::cp_async_bulk_tensor_5d_global_to_shared(&As, &tensor_map_A, 0,0,0, blockRowStart/8, K/32, barA);
      tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(As));
      //cde::cp_async_bulk_tensor_2d_global_to_shared(&Bs, &tensor_map_B, k, blockColStart, barB);
      cde::cp_async_bulk_tensor_5d_global_to_shared(&Bs, &tensor_map_B,  0,0,0, blockColStart/8, K/32, barB);
      tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(Bs));
    } else {
      tokenA = barA.arrive();
      tokenB = barB.arrive();
    }
    barA.wait(std::move(tokenA));
    barB.wait(std::move(tokenB));

    wgmma_fence();
    a_desc = get_matrix_descriptor(__cvta_generic_to_shared(As), LBO, SBO, swizzle_mode);
    b_desc = get_matrix_descriptor(__cvta_generic_to_shared(Bs), LBO, SBO, swizzle_mode);
    wgmma_m64n64k16(a_desc, b_desc, dReg);
    a_desc = get_matrix_descriptor(__cvta_generic_to_shared(As+16), LBO, SBO, swizzle_mode);
    b_desc = get_matrix_descriptor(__cvta_generic_to_shared(Bs+16), LBO, SBO, swizzle_mode);
    wgmma_m64n64k16(a_desc, b_desc, dReg);
    wgmma_commit_group();
    wgmma_wait_all();
  }


  if ((blockIdx.x==0) && (blockIdx.y==0) && (threadID==0)) {
    for (int i=0; i<64; i++) {
      for (int j=0; j<32; j++) {
        if (j%8==0)
          printf("%d, %d: ", i, j);
        half shared_a_val = As[i*32 + j];
        half global_a_val = A[i*K + K-32 + j];
        //printf("A[%d][%d]: %f, %f\n", i, j, global_a_val, shared_a_val);
        printf("(%.3f, %.3f), ", __half2float(global_a_val), __half2float(shared_a_val));
        if (j%8==7)
          printf("\n");
      }
    }
  }

  // Store from accum D registers to global memory
  int warpGroupRow = warpID * 16;
  int groupID     = laneID >> 2;
  int groupLaneID = (laneID % 4);
  float* cBlock = C + (blockRowStart + warpGroupRow + groupID) * N + blockColStart + 2 * groupLaneID; 
  for (int col=0;col<64;col+=16) {
      int regCol = col/2;
      float2 d0 = make_float2(dReg[regCol+0], dReg[regCol+1]);
      float2 d1 = make_float2(dReg[regCol+2], dReg[regCol+3]);
      float2 d2 = make_float2(dReg[regCol+4], dReg[regCol+5]);
      float2 d3 = make_float2(dReg[regCol+6], dReg[regCol+7]);
      float2 *cOut0 = reinterpret_cast<float2 *>(cBlock + col);
      float2 *cOut1 = reinterpret_cast<float2 *>(cBlock + 8*N + col);
      float2 *cOut2 = reinterpret_cast<float2 *>(cBlock + col + 8);
      float2 *cOut3 = reinterpret_cast<float2 *>(cBlock + 8*N + col + 8);
      *cOut0 = d0;
      *cOut1 = d1;
      *cOut2 = d2;
      *cOut3 = d3;
  }
}
