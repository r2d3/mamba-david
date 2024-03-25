#include <hipcub/block/block_load.hpp>

template<
    class Type,
    rocprim::block_load_method LoadMethod,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
void load_valid_kernel(Type* device_input, Type* device_output, size_t valid)
{
#if __HIP_DEVICE_COMPILE__
    Type _items[ItemsPerThread];
    auto offset = blockIdx.x * BlockSize * ItemsPerThread;
    rocprim::block_load<Type, BlockSize, ItemsPerThread, LoadMethod> load;
    load.load(device_input + offset, _items, (unsigned int)valid);
    hipcub::BlockLoad<Type, BlockSize, ItemsPerThread, hipcub::BLOCK_LOAD_WARP_TRANSPOSE> a;
    a.Load(device_input, _items);
#endif
}

template __global__ void load_valid_kernel<float, rocprim::block_load_method::block_load_warp_transpose, 32, 4>(float*, float*, size_t);

int main()
{
    return 0;
}
