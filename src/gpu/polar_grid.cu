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

// std headers
#include <cmath>
#include <stdexcept>
#include <vector>

// cuda headers
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

// local headers
#include "gpu/device_array.h"
#include "gpu/device_ptr.h"
#include "gpu/polar_grid.h"
#include "gpu/utils.h"

constexpr double PI = 3.14159265358979323846;

namespace tomocam::gpu {

    template <typename T>
    __global__ void make_polar_grid_kernel(T *theta, T *gamma, T *beta,
                                           DevicePtr<T> x, DevicePtr<T> y,
                                           DevicePtr<T> z, DevicePtr<T> w) {

        auto dims = x.dims();
        auto idx = Index3D();

        T dX = 2 * PI / (T)dims.n3;
        T dY = 2 * PI / (T)dims.n2;
        if (idx < dims) {

            T ct = cos(theta[idx.x]);
            T st = sin(theta[idx.x]);
            T cg = cos(gamma[idx.x]);
            T sg = sin(gamma[idx.x]);
            T cb = cos(beta[idx.x]);
            T sb = sin(beta[idx.x]);

            // frequency coordinates in the detector plane
            T qX = (idx.z + 0.5) * dX - PI;
            T qY = (idx.y + 0.5) * dY - PI;

            // transpose(Rx(theta) * Ry(beta) * Rz(gamma)) applied to (qX, qY, 0)
            T rx =  cb * cg * qX + (ct * sg + st * sb * cg) * qY;
            T ry = -cb * sg * qX + (ct * cg - st * sb * sg) * qY;
            T rz =  sb * qX      - st * cb * qY;

            x[idx] = rx;
            y[idx] = ry;
            z[idx] = rz;

            // zero weight for points outside the [-pi, pi]^3 cube
            w[idx] = (fabs(rx) > PI || fabs(ry) > PI || fabs(rz) > PI)
                         ? (T)0
                         : (T)1;
        }
    }

    template <typename T>
    void make_polar_grid(const std::vector<T> &theta, const std::vector<T> &gamma,
                         const std::vector<T> &beta, DeviceArray<T> &x,
                         DeviceArray<T> &y, DeviceArray<T> &z, DeviceArray<T> &w) {

        auto dims = x.dims();
        dim3 blockSize(1, 16, 16);
        dim3 gridSize;
        gridSize.x = (dims.n1 + blockSize.x - 1) / blockSize.x;
        gridSize.y = (dims.n2 + blockSize.y - 1) / blockSize.y;
        gridSize.z = (dims.n3 + blockSize.z - 1) / blockSize.z;

        thrust::device_vector<T> d_theta = theta;
        thrust::device_vector<T> d_gamma = gamma;
        thrust::device_vector<T> d_beta  = beta;
        T *d_theta_ptr = thrust::raw_pointer_cast(d_theta.data());
        T *d_gamma_ptr = thrust::raw_pointer_cast(d_gamma.data());
        T *d_beta_ptr  = thrust::raw_pointer_cast(d_beta.data());

        make_polar_grid_kernel<T>
            <<<gridSize, blockSize>>>(d_theta_ptr, d_gamma_ptr, d_beta_ptr, x, y, z, w);
        SAFE_CALL(cudaGetLastError());
    }

    template <typename T>
    PolarGrid<T>::PolarGrid(const std::vector<T> &theta, const std::vector<T> &gamma,
                            const std::vector<T> &beta, size_t nrows, size_t ncols) {
        if (theta.size() != gamma.size() || theta.size() != beta.size()) {
            throw std::invalid_argument(
                "theta, gamma, and beta must have the same size");
        }
        auto dims = dims_t{theta.size(), nrows, ncols};
        npts = dims.n1 * dims.n2 * dims.n3;
        x = DeviceArray<T>(dims);
        y = DeviceArray<T>(dims);
        z = DeviceArray<T>(dims);
        w = DeviceArray<T>(dims);
        make_polar_grid(theta, gamma, beta, x, y, z, w);
    }

    template struct PolarGrid<float>;
    template struct PolarGrid<double>;

} // namespace tomocam::gpu
