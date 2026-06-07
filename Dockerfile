FROM docker.io/nvidia/cuda:12.6.3-devel-ubuntu24.04

# Set noninteractive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and necessary tools
RUN apt-get update && apt-get install -y \
    lsb-release software-properties-common gnupg \
    git \
    ninja-build \
    libblas-dev \
    wget \
    curl \
    liblapack-dev \
    libfftw3-dev \
    lld \
    libtbb-dev \
    clang \
    libc++-dev \
    libc++abi-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# install clang-18
#RUN wget -qO- https://apt.llvm.org/llvm.sh | bash -s -- 18 clang-18 libomp-18-dev

# Install newer CMake version
RUN wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-x86_64.sh && \
    sh cmake.sh --prefix=/usr/local --skip-license && \
    rm cmake.sh

# Add library paths to ld.conf and update ld cache
RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/local.conf && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/local.conf && \
    ldconfig

# Install finufft
RUN git clone https://github.com/flatironinstitute/finufft.git && \
    cd finufft && \
    cmake -S . -B build -GNinja \
        -DCMAKE_INSTALL_PREFIX=/usr/local  \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS:BOOL=OFF \
        -DBUILD_TESTING:BOOL=OFF \
        -DBUILD_EXAMPLES:BOOL=OFF \
        -DFINUFFT_USE_CUDA:BOOL=ON \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DFINUFFT_STATIC_LINKING:BOOL=OFF \
        -DFINUFFT_BUILD_PYTHON:BOOL=OFF && \
    cmake --build build && cmake --install build && \
    cp -r include/finufft_common /usr/local/include/
RUN rm -rf finufft && ldconfig

# Install tomocam from GitHub (perlmutter branch)
RUN git clone -b master https://github.com/aryabhatt/laminocam.git && \
    cd laminocam && \
    cmake -S . -B build -G Ninja \
    -DUSE_GPU:BOOL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CUDA_HOST_COMPILER=clang++ \
    -DENABLE_TESTS:BOOL=OFF && \
    cmake --build build && cmake --install build
RUN rm -rf laminocam && ldconfig 

# Create mount points for input and output data
RUN mkdir -p /data/input /data/output
VOLUME ["/data/input", "/data/output"]

WORKDIR /data

ENTRYPOINT ["recon_scalar"]
