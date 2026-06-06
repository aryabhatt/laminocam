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

#ifndef DEVICE_PTR__H
#define DEVICE_PTR__H

#include <cuda_runtime.h>

#include "dtypes.h"

namespace tomocam::gpu {

    template <typename T>
    class DevicePtr {

      private:
        T *dev_ptr_;
        dims_t dims_;

        __device__ size_t flat_idx(int i, int j, int k) const {
            return (static_cast<size_t>(i) * dims_.n2 * dims_.n3) +
                   (static_cast<size_t>(j) * dims_.n3) + static_cast<size_t>(k);
        }

      public:
        DevicePtr(T *dev_ptr, dims_t dims) : dev_ptr_(dev_ptr), dims_(dims) {}

        __host__ __device__ [[nodiscard]] auto dims() const { return dims_; }
        __host__ __device__ [[nodiscard]] size_t size() const {
            return (dims_.n1 * dims_.n2 * dims_.n3);
        }

        // obj indexing
        __device__ T &operator[](int3 idx3) {
            auto idx = flat_idx(idx3.x, idx3.y, idx3.z);
            return dev_ptr_[idx];
        }

        // const obj indexing
        __device__ const T &operator[](int3 idx3) const {
            auto idx = flat_idx(idx3.x, idx3.y, idx3.z);
            return dev_ptr_[idx];
        }

        // linear indexing
        __device__ T &operator[](size_t idx) { return dev_ptr_[idx]; }

        // const linear indexing
        __device__ const T &operator[](size_t idx) const { return dev_ptr_[idx]; }

        // three-dim indexing
        __device__ T &operator()(int i, int j, int k) {
            auto idx = flat_idx(i, j, k);
            return dev_ptr_[idx];
        }

        // const three-dim indexing
        __device__ const T &operator()(int i, int j, int k) const {
            auto idx = flat_idx(i, j, k);
            return dev_ptr_[idx];
        }
    };

} // namespace tomocam::gpu

#endif // DEVICE_PTR__H
