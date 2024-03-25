# Port of `selective_scan` to `ROCm`/`HIP`

## Porting to ROCm

`selective_scan` files from `mamba/csrc/selective_scan` has been copied
to `mamba/rocm` and modified to work with ROCm/HIP.

- `static_switch.hpp`: unchanged
- `selective_scan.h`: unchanged, Params structure
- `selective_scan_common.h`: modified the way `Ktraits:Block*` were used to comply with HIP compiler
- `selective_scan.cpp`: use HIP equivalent functions; Python bindings, fill Params structure from input Tensors
- `selective_scan_fwd_kernel.cuh`: use hipCUB, use `#if __HIP_DEVICE_COMPILE__` to workaround a HIP compiler bug, correct `Ktraits::Block` compilation issue, harcode shared memory size (`kSmemSize`) as compiler does not allow to use class that partially use device code on the host side
- `mamba.hip`: instantiate `selective_scan_fwd_cuda` for `at::BFloat16`, `at::Half` and `float`

A `CMakeLists.txt` has been created in order to compile our code and the Python module as `torch.utils.cpp_extension` does not (yet) support ROCm/HIP.

It download pybind11, find the location of the `torch` Python as it contains include/lib used to interface C++ with `torch` objects.

## Compiling/Running the code

We use `rocm/pytorch` Docker image to do compilation/execution

~~~ bash
git clone -b feature/rocm git@github.com:r2d3/mamba-david.git mamba-david

docker run -it --mount type=bind,source=/home/a2labs,target=/host \
 --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
 --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
 --shm-size 8G \
 rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1 /bin/bash
 
mkdir ~/build_mamba && cd ~/build_mamba
cmake -DCMAKE_PREFIX_PATH=/opt/rocm /host/mamba-david/rocm

make -j 10
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  python -c "import mamba; mamba.test()"

# Install packages needed to run the test
pip install einops
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env" 
pip install transformers

# Run test
cd /host/mamba-david
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH=.:~/build_mamba pytest -v \
  tests/ops/test_selective_scan.py::test_selective_scan
~~~

Use our Docker image ('mamba') with pre-installed Python modules needed to test

~~~ bash
docker run -it --mount type=bind,source=/home/a2labs,target=/host \
 --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
 --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
 --shm-size 8G \
 mamba /bin/bash

mkdir ~/build_mamba && cd ~/build_mamba
cmake -DCMAKE_PREFIX_PATH=/opt/rocm /host/mamba-david/rocm

make -j 10
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  python -c "import mamba; mamba.test()"

# Run test
cd /host/mamba-david
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH=.:~/build_mamba pytest -v \
  tests/ops/test_selective_scan.py::test_selective_scan
~~~

If Docker image is not already built, build it with: `docker build -t mamba ~/mamba-david/rocm`
