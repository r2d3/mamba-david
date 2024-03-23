#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>
using namespace std;

#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <c10/util/complex.h>  // For scalar_value_type
using complex_t = c10::complex<float>;

#include "selective_scan.h"

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase& params, hipStream_t stream);

void test()
{
    cout << "Test" << endl;
    complex_t c;
    at::Tensor a;
    at::Tensor b;
    cout << a.stride(0) << endl;
}

PYBIND11_MODULE(mamba_ssm, m) {
    m.doc() = "ROCm Mamba module";
//    m.def("fwd", &selective_scan_fwd, "Selective scan forward");
    m.def("test", &test, "Test");
}
