#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <hip/hip_runtime.h>

#include "selective_scan_fwd_kernel.cuh"

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase& params, hipStream_t stream)
{
}

template void selective_scan_fwd_cuda<float, float>(SSMParamsBase& params, hipStream_t stream);

#define CHECK_HIP(expr) do {              \
  hipError_t result = (expr);             \
  if (result != hipSuccess) {             \
    fprintf(stderr, "%s:%d: %s (%d)\n",   \
      __FILE__, __LINE__,                 \
      hipGetErrorString(result), result); \
    exit(EXIT_FAILURE);                   \
  }                                       \
} while(0)

__global__ void sq_arr(float *arr, int n)
{
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if (tid < n) {
    arr[tid] = arr[tid] * arr[tid];
  }
}

int main() {
  const int N = 10000000;
  std::vector<float> hArr(N);
  for (int i = 0; i < 10; i++)
    hArr[i] = i+1;
  float *dArr;
  CHECK_HIP(hipMalloc(&dArr, sizeof(float) * N));
  for (int j = 0; j < 1000; j++)
  {
    CHECK_HIP(hipMemcpy(dArr, hArr.data(), sizeof(float) * N, hipMemcpyHostToDevice));
    sq_arr<<<dim3(1), dim3(32,1,1), 0, 0>>>(dArr, N);
  }
  CHECK_HIP(hipMemcpy(hArr.data(), dArr, sizeof(float) * N, hipMemcpyDeviceToHost));
  for (int i = 0; i < 10; ++i)
    printf("%f\n", hArr[i]);
  CHECK_HIP(hipFree(dArr));
  return 0;
}
