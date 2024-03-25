FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1 
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install einops transformers
