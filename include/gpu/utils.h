

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cufft.h>
#include <cuda_runtime.h>

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
    // Utility function to calculate block dimensions for a given problem
    inline dim3 make_grid(dim3 dimensions, dim3 threads = {1, 16, 16}) {
        dim3 blocks;
        blocks.x = (dimensions.x + threads.x - 1) / threads.x;
        blocks.y = (dimensions.y + threads.y - 1) / threads.y;
        blocks.z = (dimensions.z + threads.z - 1) / threads.z;
        return blocks;
    }

    // Utility function to get the 3D index of the current thread in the grid
    __device__ __inline__ int3 Index3D() {
        return make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                         blockIdx.y * blockDim.y + threadIdx.y,
                         blockIdx.z * blockDim.z + threadIdx.z);
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
