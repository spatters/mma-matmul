#include <mma.h>
#include <cuda_fp16.h>

#define M_TILE 2
#define N_TILE 2

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


// Kernel 1.0: Naive mma 
__launch_bounds__(16 * 16)
__global__ void mma_matmul_1_0(const half *A, const half *B, float *C, int M, int N, int K) {
  // declare cache in shared memory
  __shared__ half As[32][16];
  __shared__ half Bs[16][32];

  int mBlock = 32 * blockIdx.y;
  int nBlock = 32 * blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // warps arranged in 2x4 grid:
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;

  // warp offsets in threadblock shmem tiles
  int nWarp = 8 * (warpID % 4);
  int mWarp = 16 * (warpID / 4);

  // warps are split into 8 groups of 4 threads each
  int groupID     = laneID / 4;
  int groupLaneID = laneID % 4;

  half  aReg[8];
  half  bReg[4]; 
  float dReg[4] = {0.};

  for (int kStart=0; kStart < K; kStart += 16) {
    As[ty     ][tx] = A[(mBlock + ty     )*K + kStart + tx];
    As[ty + 16][tx] = A[(mBlock + ty + 16)*K + kStart + tx];
    Bs[ty][tx     ] = B[(kStart + ty)*K + nBlock      + tx];
    Bs[ty][tx + 16] = B[(kStart + ty)*K + nBlock + 16 + tx];
    __syncthreads();

    // set up the registers for mma call
    aReg[0] = As[mWarp + groupID    ][groupLaneID*2    ];
    aReg[1] = As[mWarp + groupID    ][groupLaneID*2 + 1];
    aReg[2] = As[mWarp + groupID + 8][groupLaneID*2    ];
    aReg[3] = As[mWarp + groupID + 8][groupLaneID*2 + 1];
    aReg[4] = As[mWarp + groupID    ][groupLaneID*2 + 8];
    aReg[5] = As[mWarp + groupID    ][groupLaneID*2 + 9];
    aReg[6] = As[mWarp + groupID + 8][groupLaneID*2 + 8];
    aReg[7] = As[mWarp + groupID + 8][groupLaneID*2 + 9];

    bReg[0] = Bs[groupLaneID*2 + 0][nWarp + groupID];
    bReg[1] = Bs[groupLaneID*2 + 1][nWarp + groupID];
    bReg[2] = Bs[groupLaneID*2 + 8][nWarp + groupID];
    bReg[3] = Bs[groupLaneID*2 + 9][nWarp + groupID];
    unsigned const *aPtr = reinterpret_cast<unsigned const *>(&aReg);
    unsigned const *bPtr = reinterpret_cast<unsigned const *>(&bReg);
    mma_m16n8k16(aPtr, bPtr, dReg, dReg);
    __syncthreads();
  }
  // Write results to global memory
  C[(mBlock + mWarp + groupID)*N + nBlock + nWarp + 2*groupLaneID] = dReg[0];
  C[(mBlock + mWarp + groupID)*N + nBlock + nWarp + 2*groupLaneID+1] = dReg[1];
  C[(mBlock + mWarp + groupID+8)*N + nBlock + nWarp + 2*groupLaneID] = dReg[2];
  C[(mBlock + mWarp + groupID+8)*N + nBlock + nWarp + 2*groupLaneID+1] = dReg[3];
}


// Kernel 1.1: Naive mma with 2x M/N tiling
__launch_bounds__(16 * 16)
__global__ void mma_matmul_1_1(const half *A, const half *B, float *C, int M, int N, int K) {
  // declare cache in shared memory
  __shared__ half As[M_TILE * 32][16];
  __shared__ half Bs[16][N_TILE * 32];

  int mBlock = M_TILE * 32 * blockIdx.y;
  int nBlock = N_TILE * 32 * blockIdx.x;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int threadID = threadIdx.y * blockDim.x + threadIdx.x;
  int warpID = threadID / 32;
  int laneID = threadID % 32;

  // tile warps as follows
  // (warp_0 | warp_1 | warp_2 | warp_3)
  // (warp_4 | warp_5 | warp_6 | warp_7)
  int nWarp = 8 * (warpID % 4);
  int mWarp = 16 * (warpID / 4);

  int groupID     = laneID / 4;
  int groupLaneID = laneID % 4;

  half  aReg[8];
  half  bReg[4];
  float dReg[M_TILE][N_TILE][4] = {0.};

  for (int kStart=0; kStart < K; kStart += 16) {
    for (int m=0; m < M_TILE; ++m) {
      int mTile = m * 32;
      As[mTile      + ty][tx] = A[(mBlock + mTile      + ty)*K + kStart + tx];
      As[mTile + 16 + ty][tx] = A[(mBlock + mTile + 16 + ty)*K + kStart + tx];
    }
    for (int n=0; n < N_TILE; ++n) {
      int nTile = n * 32;
      Bs[ty][nTile      + tx] = B[(kStart + ty)*K + nBlock + nTile      + tx];
      Bs[ty][nTile + 16 + tx] = B[(kStart + ty)*K + nBlock + nTile + 16 + tx];
    }
    __syncthreads();
    for (int m=0; m < M_TILE; m++) {
      int mTile = m * 32;
      // set up the registers for mma call
      aReg[0] = As[mTile + mWarp + groupID    ][groupLaneID*2    ];
      aReg[1] = As[mTile + mWarp + groupID    ][groupLaneID*2 + 1];
      aReg[2] = As[mTile + mWarp + groupID + 8][groupLaneID*2    ];
      aReg[3] = As[mTile + mWarp + groupID + 8][groupLaneID*2 + 1];
      aReg[4] = As[mTile + mWarp + groupID    ][groupLaneID*2 + 8];
      aReg[5] = As[mTile + mWarp + groupID    ][groupLaneID*2 + 9];
      aReg[6] = As[mTile + mWarp + groupID + 8][groupLaneID*2 + 8];
      aReg[7] = As[mTile + mWarp + groupID + 8][groupLaneID*2 + 9];
      for (int n=0; n < N_TILE; n++) {
        int nTile = n * 32;
        bReg[0] = Bs[groupLaneID*2 + 0][nTile + nWarp + groupID];
        bReg[1] = Bs[groupLaneID*2 + 1][nTile + nWarp + groupID];
        bReg[2] = Bs[groupLaneID*2 + 8][nTile + nWarp + groupID];
        bReg[3] = Bs[groupLaneID*2 + 9][nTile + nWarp + groupID];

        unsigned const *aPtr = reinterpret_cast<unsigned const *>(&aReg);
        unsigned const *bPtr = reinterpret_cast<unsigned const *>(&bReg);
        mma_m16n8k16(aPtr, bPtr, dReg[m][n], dReg[m][n]);
      }
    }
    __syncthreads();
  }
  // Copy dReg to global memory
  for (int m=0; m < M_TILE; m++) {
    int mTile = m * 32;
    for (int n=0; n < N_TILE; n++) {
      int nTile = n * 32;
      C[(mBlock + mTile + mWarp + groupID  )*N + nBlock + nTile + nWarp + 2*groupLaneID  ] = dReg[m][n][0];
      C[(mBlock + mTile + mWarp + groupID  )*N + nBlock + nTile + nWarp + 2*groupLaneID+1] = dReg[m][n][1];
      C[(mBlock + mTile + mWarp + groupID+8)*N + nBlock + nTile + nWarp + 2*groupLaneID  ] = dReg[m][n][2];
      C[(mBlock + mTile + mWarp + groupID+8)*N + nBlock + nTile + nWarp + 2*groupLaneID+1] = dReg[m][n][3];
    }
  }
}
