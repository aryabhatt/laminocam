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
#include <thrust/device_vector.h>

#include "gpu/device_array.h"
#include "gpu/device_ptr.h"
#include "gpu/polar_grid.h"
#include "gpu/utils.h"

constexpr double PI = 3.14159265358979323846;

namespace tomocam::gpu {

    template <typename T>
    __global__ void make_polar_grid_kernel(T *theta, T gamma, DevicePtr<T> x,
                                           DevicePtr<T> y, DevicePtr<T> z) {

        auto dims = x.dims();
        auto idx = Index3D();

        T dX = 2 * PI / (T)dims.n3;
        T dY = 2 * PI / (T)dims.n2;
        if (idx < dims) {
            // qX, qY frequency coordinates in the plane of the detector
            T qX = (idx.z + 0.5) * dX - PI;
            T qY = (idx.y + 0.5) * dY - PI;
            // qZ frequency coordinate along the beam direction

            x[idx] = qX * cos(gamma) - qY * sin(gamma) * cos(theta[idx.x]);
            y[idx] = qX * sin(gamma) + qY * cos(gamma) * cos(theta[idx.x]);
            z[idx] = qY * sin(theta[idx.x]);
        }
    }

    template <typename T>
    void make_polar_grid(const std::vector<T> &angles, T gamma,
                         DeviceArray<T> &x, DeviceArray<T> &y, DeviceArray<T> &z) {

        // move angles to device
        thrust::device_vector<T> d_angles = angles;

        auto dims = x.dims();
        dim3 blockSize(1, 16, 16);
        dim3 gridSize;
        gridSize.x = (dims.n1 + blockSize.x - 1) / blockSize.x;
        gridSize.y = (dims.n2 + blockSize.y - 1) / blockSize.y;
        gridSize.z = (dims.n3 + blockSize.z - 1) / blockSize.z;
        make_polar_grid_kernel<T><<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_angles.data()), gamma, x, y, z);
        SAFE_CALL(cudaGetLastError());
    }

    template <typename T>
    PolarGrid<T>::PolarGrid(const std::vector<T> &theta, T gamma, size_t nrows,
                            size_t ncols) {
        auto dims = dims_t{theta.size(), nrows, ncols};
        npts = dims.n1 * dims.n2 * dims.n3;
        x = DeviceArray<T>(dims);
        y = DeviceArray<T>(dims);
        z = DeviceArray<T>(dims);
        make_polar_grid(theta, gamma, x, y, z);
    }

    template struct PolarGrid<float>;
    template struct PolarGrid<double>;

} // namespace tomocam::gpu
