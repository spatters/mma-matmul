#include <iostream>
#include <thrust/host_vector.h>                
#include <thrust/device_vector.h>             

#include <curand.h>
#include <cuda_bf16.h>  // for __nv_bfloat16

#define ceilDiv(x, y) (((x) + (y) - 1) / (y))
int main()
{
  constexpr int M = 4096;
  constexpr int N = 4096;
  constexpr int K = 4096;

  thrust::device_vector<float> A_float(M*K);
  thrust::device_vector<float> B_float(M*K);
  thrust::device_vector<__nv_bfloat16> A(M*N);
  thrust::device_vector<__nv_bfloat16> B(M*N);

  thrust::host_vector<float> h_C(M*N, 0);
  thrust::device_vector<float> C(M*N, 0);

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curandSetPseudoRandomGeneratorSeed(gen, 1234);

	curandGenerateNormal(gen, thrust::raw_pointer_cast(A_float.data()), M*N, 0.0f, 1.0f);
	curandGenerateNormal(gen, thrust::raw_pointer_cast(B_float.data()), M*N, 0.0f, 1.0f);
  thrust::transform(A_float.begin(), A_float.end(), A.begin(), __float2bfloat16);
  thrust::transform(B_float.begin(), B_float.end(), B.begin(), __float2bfloat16);

  __nv_bfloat16  *d_A_ptr = thrust::raw_pointer_cast(A.data());
  __nv_bfloat16  *d_B_ptr = thrust::raw_pointer_cast(B.data());
  float *d_C_ptr = thrust::raw_pointer_cast(C.data());
  float *h_C_ptr = thrust::raw_pointer_cast(h_C.data());
}
