#include "selective_scan_bwd_kernel.cuh"

#include <c10/util/complex.h>
using complex_t = c10::complex<float>;

template void selective_scan_bwd_cuda<at::BFloat16, float>(SSMParamsBwd &params, hipStream_t stream);
#if 0
template void selective_scan_bwd_cuda<at::BFloat16, complex_t>(SSMParamsBwd &params, hipStream_t stream);

template void selective_scan_bwd_cuda<at::Half, float>(SSMParamsBwd &params, hipStream_t stream);
template void selective_scan_bwd_cuda<at::Half, complex_t>(SSMParamsBwd &params, hipStream_t stream);

template void selective_scan_bwd_cuda<float, float>(SSMParamsBwd &params, hipStream_t stream);
template void selective_scan_bwd_cuda<float, complex_t>(SSMParamsBwd &params, hipStream_t stream);
#endif
