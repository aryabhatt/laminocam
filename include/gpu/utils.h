/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals from the
 * U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 * IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */


#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdexcept>
#include <tuple>

#define SAFE_CALL(ans)                                                              \
    { checkCuda((ans), __FILE__, __LINE__); }
inline void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s at %s:%d\n",
                cudaGetErrorString(result), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUFFT_CHECK(ans)                                                            \
    { checkCufft((ans), __FILE__, __LINE__); }
inline void checkCufft(cufftResult result, const char *file, int line) {
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT error %d at %s:%d\n", result, file, line);
        exit(EXIT_FAILURE);
    }
}

namespace tomocam::gpu {

#ifdef __NVCC__
    inline dim3 make_dim3(dims_t dims) { return dim3(dims.n1, dims.n2, dims.n3); }

    // Utility function to calculate block dimensions for a given problem
    inline dim3 make_grid(dim3 dimensions, dim3 threads = {1, 8, 32}) {
        dim3 blocks;
        blocks.x = (dimensions.x + threads.x - 1) / threads.x;
        blocks.y = (dimensions.y + threads.y - 1) / threads.y;
        blocks.z = (dimensions.z + threads.z - 1) / threads.z;
        return blocks;
    }

    inline dim3 make_grid(dims_t dimensions, dim3 threads = {1, 8, 32}) {
        return make_grid(make_dim3(dimensions), threads);
    }

    // Utility function to get the 3D index of the current thread in the grid
    __device__ __inline__ int3 Index3D() {
        return make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                         blockIdx.y * blockDim.y + threadIdx.y,
                         blockIdx.z * blockDim.z + threadIdx.z);
    }

    __device__ __inline__ bool operator<(const int3 &a, const dims_t &b) {
        return ((size_t)a.x < b.n1) && ((size_t)a.y < b.n2) && ((size_t)a.z < b.n3);
    }

#endif // __NVCC__
    // Utility function to ensure that the current GPU device matches the specified
    // GPU ID
    inline bool ensure_device(int gpu_id) {
        int current_gpu;
        SAFE_CALL(cudaGetDevice(&current_gpu));
        if (current_gpu != gpu_id) {
            throw std::runtime_error(
                "Current GPU does not match the specified GPU ID.");
        }
        return true;
    }

} // namespace tomocam::gpu

#endif // CUDA_UTILS_H
