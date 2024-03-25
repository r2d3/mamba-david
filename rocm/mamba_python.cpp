#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>
using namespace std;

#include <hip/hip_runtime.h>
#include <c10/hip/HIPException.h>  // For C10_HIP_CHECK and C10_HIP_KERNEL_LAUNCH_CHECK

__global__ void sq_arr(float *arr, int n)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if (tid < n) {
        arr[tid] = arr[tid] * arr[tid];
    }
}

void test()
{
  const int N = 10000000;
  std::vector<float> hArr(N);
  for (int i = 0; i < 10; i++)
    hArr[i] = i+1;
  float *dArr;
  C10_HIP_CHECK(hipMalloc(&dArr, sizeof(float) * N));
  for (int j = 0; j < 1000; j++)
  {
    C10_HIP_CHECK(hipMemcpy(dArr, hArr.data(), sizeof(float) * N, hipMemcpyHostToDevice));
    sq_arr<<<dim3(1), dim3(32,1,1), 0, 0>>>(dArr, N);
    C10_HIP_CHECK(hipDeviceSynchronize());
    C10_HIP_KERNEL_LAUNCH_CHECK();
  }
  C10_HIP_CHECK(hipMemcpy(hArr.data(), dArr, sizeof(float) * N, hipMemcpyDeviceToHost));
  for (int i = 0; i < 10; ++i)
    printf("%f\n", hArr[i]);
  C10_HIP_CHECK(hipFree(dArr));
}


PYBIND11_MODULE(mamba, m) {
    m.doc() = "ROCm Mamba module";
    m.def("test", &test, "Test");
}
