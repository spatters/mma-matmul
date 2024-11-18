#include <mma.h>
#include <cuda_fp16.h>

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

// Kernel 2.0: Permuted shmem layout
__launch_bounds__(16 * 16)
__global__ void mma_matmul_2_0(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ uint4 As[32][8];
  __shared__ uint4 Bs[32][8];
  int mBlock = blockIdx.y*64;
  int nBlock = blockIdx.x*64;
  const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + mBlock * K);
  const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + nBlock * K);

  // warp layout is 2 x 4
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int mWarp = 16 * (warpID / 4);
  int nWarp = 8 * (warpID % 4);

  unsigned aReg[4];
  unsigned bReg[2];
  float dReg[2][2][4] = {0.};

  // row / column indices when storing to shared memory
  int storeRow = warpID * 4 + laneID / 8;
  int storeCol = (laneID % 8) ^ (laneID / 8);

  // row/column indices when loading from permuted shmem layout to registers
  int loadRowA = (laneID % 16) / 2;
  int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
  int loadRowB = (laneID % 8) / 2;
  int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);

  for (int k = 0; k < K/8; k += 4) {
    As[storeRow][storeCol] = globalTileA[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
    Bs[storeRow][storeCol] = globalTileB[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
    __syncthreads();

    // loop over the two (M/N=16, K=4) tiles of a and b
    for (int m = 0; m < 2; m++) {
      int mTile = m * 8;
      for (int n = 0; n < 2; n++) {
        int nTile = n * 4;
        load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + loadColA));
        load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + loadColB));
        mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);
        load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + (loadColA^2)));
        load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + (loadColB^2)));
        mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);
      }
    }
    __syncthreads();
  }

  int groupID     = laneID >> 2;
  int groupLaneID = laneID % 4;
  for (int m = 0; m < 2; m++) {
    for (int n = 0; n <  2; n++) {
      int mTile = m * 16;
      int nTile = n * 8;
      float2 d0 = make_float2(dReg[m][n][0], dReg[m][n][1]);
      float2 d2 = make_float2(dReg[m][n][2], dReg[m][n][3]);
      float2 *cOut0 = reinterpret_cast<float2 *>(C + (mBlock + mTile + 2*mWarp + groupID    )*N + nBlock + nTile + 2*nWarp + 2 * groupLaneID);
      float2 *cOut2 = reinterpret_cast<float2 *>(C + (mBlock + mTile + 2*mWarp + groupID + 8)*N + nBlock + nTile + 2*nWarp + 2 * groupLaneID);
      *cOut0 = d0;
      *cOut2 = d2;
    }
  }
  __syncthreads();
}

// Kernel 2.1: Permuted shmem layout, only load a to registers once
__launch_bounds__(16 * 16)
__global__ void mma_matmul_2_1(const half *A, const half *B, float *C, int M, int N, int K) {
  __shared__ uint4 As[32][8];
  __shared__ uint4 Bs[32][8];
  int blockRowStart = blockIdx.y*64;
  int blockColStart = blockIdx.x*64;
  const uint4 *globalTileA = reinterpret_cast<const uint4 *>(A + blockRowStart * K);
  const uint4 *globalTileB = reinterpret_cast<const uint4 *>(B + blockColStart * K);

  // warp layout is 2 x 4
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;
  int warpOffsetA = 16 * (warpID / 4);
  int warpOffsetB = 8 * (warpID % 4);

  unsigned aReg[8];
  unsigned bReg[2];
  float dReg[2][2][4] = {0.};

  // row / column indices when storing to shared memory
  int storeRow = warpID * 4 + laneID / 8;
  int storeCol = (laneID % 8) ^ (laneID / 8);

  // row / column indices when loading from permuted shmem layout to registers
  int loadRowA = (laneID % 16) / 2;
  int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
  int loadRowB = (laneID % 8) / 2;
  int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);

  for (int k = 0; k < K/8; k += 4) {
    As[storeRow][storeCol] = globalTileA[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
    Bs[storeRow][storeCol] = globalTileB[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
    __syncthreads();

    // loop over the two 16x8 tiles of a and b
    for (int m = 0; m < 2; m++) {
      int mTile = m * 8;
      load_matrix_x4(aReg, (As[mTile + warpOffsetA + loadRowA] + loadColA));
      load_matrix_x4(aReg + 4, (As[mTile + warpOffsetA + loadRowA] + (loadColA^2)));
      for (int n = 0; n < 2; n++) {
        int nTile = n * 4;
        load_matrix_x2(bReg, (Bs[nTile + warpOffsetB + loadRowB] + loadColB));
        mma_m16n8k16(aReg, bReg, dReg[m][n], dReg[m][n]);
        load_matrix_x2(bReg, (Bs[nTile + warpOffsetB + loadRowB] + (loadColB^2)));
        mma_m16n8k16(aReg+4, bReg, dReg[m][n], dReg[m][n]);
      }
    }
    __syncthreads();
  }

  int groupID     = laneID >> 2;
  int groupLaneID = laneID % 4;
  for (int m = 0; m < 2; m++) {
    for (int n = 0; n <  2; n++) {
      int mTile = m * 16;
      int nTile = n * 8;
      float2 d0 = make_float2(dReg[m][n][0], dReg[m][n][1]);
      float2 d2 = make_float2(dReg[m][n][2], dReg[m][n][3]);
      float2 *cOut0 = reinterpret_cast<float2 *>(&C[(blockRowStart + mTile + 2*warpOffsetA + groupID    )*N + blockColStart + nTile + 2*warpOffsetB + 2 * groupLaneID]);
      float2 *cOut2 = reinterpret_cast<float2 *>(&C[(blockRowStart + mTile + 2*warpOffsetA + groupID + 8)*N + blockColStart + nTile + 2*warpOffsetB + 2 * groupLaneID]);
      *cOut0 = d0;
      *cOut2 = d2;
    }
  }
}
