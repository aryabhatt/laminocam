#! /bin/bash

function module_load() {
    local mod="$1"

    # if loaded do nothing
    if module is-loaded "$mod" 2> /dev/null; then
        return 0
    fi
    module load "$mod"
}

module_load PrgEnv-gnu
module_load cmake/3.30.2
module_load cray-fftw/3.3.10.8


cmake -S . -B build -G Ninja \
    -DCMAKE_INSTALL_PREFIX=/global/common/software/als/camera \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DUSE_GPU:BOOL=ON \
    -DENABLE_TESTS:BOOL=OFF \
    -Dfinufft_DIR=/global/common/software/als/camera && \
    cmake --build build && \
    cmake --install build
