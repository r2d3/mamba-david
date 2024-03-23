#include "selective_scan_fwd_kernel.cuh"

#include <c10/util/complex.h>
using complex_t = c10::complex<float>;

template void selective_scan_fwd_cuda<float, float>(SSMParamsBase& params, hipStream_t stream);
template void selective_scan_fwd_cuda<float, complex_t>(SSMParamsBase& params, hipStream_t stream);

int main()
{
    return 0;
}
