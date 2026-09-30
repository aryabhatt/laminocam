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

#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "gpu/device_array.h"
#include "gpu/device_ptr.h"
#include "gpu/polar_grid2d.h"
#include "gpu/utils.h"

constexpr double PI2D = 3.14159265358979323846;

namespace tomocam::gpu {

    template <typename T>
    __global__ void make_polar_grid2d_kernel(T *theta, DevicePtr<T> x,
                                             DevicePtr<T> y) {

        auto dims = x.dims();
        auto idx = Index3D();

        T dr = 2 * PI2D / (T)dims.n3;
        if (idx < dims) {
            T r = (idx.z + 0.5) * dr - PI2D;
            x[idx] = cos(theta[idx.y]) * r;
            y[idx] = sin(theta[idx.y]) * r;
        }
    }

    template <typename T>
    void make_polar_grid2d(const std::vector<T> &theta, DeviceArray<T> &x,
                           DeviceArray<T> &y) {

        auto dims = x.dims();
        dim3 blockSize(1, 16, 16);
        dim3 gridSize;
        gridSize.x = (dims.n1 + blockSize.x - 1) / blockSize.x;
        gridSize.y = (dims.n2 + blockSize.y - 1) / blockSize.y;
        gridSize.z = (dims.n3 + blockSize.z - 1) / blockSize.z;

        thrust::device_vector<T> d_theta = theta;
        T *d_theta_ptr = thrust::raw_pointer_cast(d_theta.data());

        make_polar_grid2d_kernel<T><<<gridSize, blockSize>>>(d_theta_ptr, x, y);
        SAFE_CALL(cudaGetLastError());
    }

    template <typename T>
    PolarGrid2D<T>::PolarGrid2D(const std::vector<T> &theta, size_t ncols) {
        auto dims = dims_t{1, theta.size(), ncols};
        npts = dims.n1 * dims.n2 * dims.n3;
        x = DeviceArray<T>(dims);
        y = DeviceArray<T>(dims);
        make_polar_grid2d(theta, x, y);
    }

    template struct PolarGrid2D<float>;
    template struct PolarGrid2D<double>;

} // namespace tomocam::gpu
