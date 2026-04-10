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
