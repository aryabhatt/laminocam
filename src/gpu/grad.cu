#include <cuda_runtime.h>

#include "gpu/device_array.h"
#include "gpu/device_ptr.h"
#include "gpu/utils.cuh"

namespace tomocam::gpu {

    template <typename T>
    __global__ void grad_ux_kernel(DevicePtr<T> u, DevicePtr<T> u_x) {

        __shared__ extern char *_shamem;
        T *shared_u = reinterpret_cast<T *>(_shamem);
        auto idx = Index3D();
        if (idx < u.dims()) {
            // Load data into shared memory

            // calculate gradients using central differences
            u_x[idx] = shared_u[{idx.x, idx.y, idx.z + 1}] -
                       shared_u[{idx.x, idx.y, idx.z - 1}];
        }
    }

    template <typename T>
    __global__ void grad_uy_kernel(DevicePtr<T> u, DevicePtr<T> u_y) {

        __shared__ extern char *_shamem;
        T *shared_u = reinterpret_cast<T *>(_shamem);
        auto idx = Index3D();
        if (idx < u.dims()) {
            // Load data into shared memory

            // calculate gradients using central differences
            u_y[idx] = shared_u[{idx.x, idx.y + 1, idx.z}] -
                       shared_u[{idx.x, idx.y - 1, idx.z}];
        }
    }

    template <typename T>
    __global__ void grad_uz_kernel(DevicePtr<T> u, DevicePtr<T> u_z) {
        __shared__ extern char *_shamem;
        T *shared_u = reinterpret_cast<T *>(_shamem);
        auto idx = Index3D();
        if (idx < u.dims()) {
            // Load data into shared memory

            // calculate gradients using central differences
            u_z[idx] = shared_u[{idx.x + 1, idx.y, idx.z}] -
                       shared_u[{idx.x - 1, idx.y, idx.z}];
        }
    }
} // namespace tomocam::gpu
