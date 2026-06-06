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
#ifndef GPUMEMORY__H
#define GPUMEMORY__H

#include <iostream>
#include <memory>
#include <stdexcept>

#include <cuda_runtime.h>
#include <string>

#include "utils.h"

namespace tomocam::gpu {

    namespace memory {
        struct cudaDelete {
            void operator()(void *ptr) const noexcept {
                if (ptr != nullptr) {
                    auto err = cudaFree(ptr);
                    if (err != cudaSuccess) {
                        std::cerr << "cudaFree failed: " << cudaGetErrorString(err)
                                  << '\n';
                    }
                }
            }
        };

        template <typename T>
        using cunique_ptr = std::unique_ptr<T, cudaDelete>;

        template <typename T>
        cunique_ptr<T> make_cunique_ptr(std::size_t count) {
            T *raw = nullptr;
            auto err = cudaMalloc(&raw, sizeof(T) * count);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("failed to allocated gpu memory"));
            }
            return cunique_ptr<T>(raw);
        }
    } // namespace memory

    /**
     * copy helper function for device memroy
     * @param dst destination pointer
     * @param src source pointer
     * @param size number of bytes to copy
     * throws std::runtime_error if cudaMemcpy fails
     */
    // device-to-device
    inline void copyD2D(void *dst, const void *src, size_t size) {
        auto err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("failed to copy data: ") +
                                     cudaGetErrorString(err));
        }
    }

    // host-to-device
    inline void copyH2D(void *d_ptr, const void *src, size_t size) {
        auto err = cudaMemcpy(d_ptr, src, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("failed to copy data to device: ") +
                                     cudaGetErrorString(err));
        }
    }
    inline void copyD2H(void *ptr, const void *d_ptr, size_t size) {
        auto err = cudaMemcpy(ptr, d_ptr, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("failed to copy data to host: ") +
                                     cudaGetErrorString(err));
        }
    }

    // Aliases used by nufft.h
    inline void copy_to_device(const void *src, void *d_dst, size_t size) {
        copyH2D(d_dst, src, size);
    }
    inline void copy_to_host(const void *d_src, void *dst, size_t size) {
        copyD2H(dst, d_src, size);
    }

    /* Shallow wrapper to provide 3D indexing for shared memory */
    template <typename T>
    class SharedMemory {
      private:
        dim3 dims_;
        T *ptr_;

        __forceinline__ __device__ unsigned get_dynamic_smem_size() {
            unsigned size;
            asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(size));
            return size;
        }

      public:
        __device__ __inline__ SharedMemory(dim3 dims) : dims_(dims) {
            extern __shared__ unsigned char shared_mem[];
#ifdef DEBUG
            unsigned expected_size = dims_.x * dims_.y * dims_.z * sizeof(T);
            unsigned actual_size = get_dynamic_smem_size();
            assert(expected_size <= actual_size);
#endif
            ptr_ = reinterpret_cast<T *>(shared_mem);
        }
        __device__ __inline__ T &operator()(unsigned x, unsigned y, unsigned z) {
            return ptr_[x * dims_.y * dims_.z + y * dims_.z + z];
        }
        __device__ __inline__ const T &operator()(unsigned x, unsigned y,
                                                  unsigned z) const {
            return ptr_[x * dims_.y * dims_.z + y * dims_.z + z];
        }

        __device__ __inline__ T &operator()(unsigned idx) { return ptr_[idx]; }
        __device__ __inline__ const T &operator()(unsigned idx) const {
            return ptr_[idx];
        }
    };

} // namespace tomocam::gpu

#endif // GPUMEMORY__H
