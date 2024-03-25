#include "selective_scan_fwd_kernel.cuh"

#include <c10/util/complex.h>
using complex_t = c10::complex<float>;

template void selective_scan_fwd_cuda<at::BFloat16, float>(SSMParamsBase &params, hipStream_t stream);
template void selective_scan_fwd_cuda<at::BFloat16, complex_t>(SSMParamsBase &params, hipStream_t stream);

template void selective_scan_fwd_cuda<at::Half, float>(SSMParamsBase &params, hipStream_t stream);
template void selective_scan_fwd_cuda<at::Half, complex_t>(SSMParamsBase &params, hipStream_t stream);

template void selective_scan_fwd_cuda<float, float>(SSMParamsBase& params, hipStream_t stream);
template void selective_scan_fwd_cuda<float, complex_t>(SSMParamsBase& params, hipStream_t stream);

