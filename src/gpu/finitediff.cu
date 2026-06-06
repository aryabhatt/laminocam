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
#include "gpu/utils.h"

namespace tomocam::gpu::opt {

    template <typename T, bool update = false>
    __global__ void grad_ux_kernel(DevicePtr<const T> u, DevicePtr<T> du_x) {

        // Allocate shared memory for the current block
        dim3 dims = blockDim;
        dims.z += 2;
        SharedMemory<T> shared_u(dims);

        auto idx = Index3D();
        if (idx < u.dims()) {
            // calculate chunk size

            // Load data into shared memory
            for (unsigned i = 0; i < 3; ++i) {
                unsigned z_idx = idx.z - 1 + i;
                if (z_idx < u.dims().n3) {
                    shared_u(threadIdx.x, threadIdx.y, threadIdx.z + i) =
                        u(idx.x, idx.y, z_idx);
                } else {
                    shared_u(threadIdx.x, threadIdx.y, threadIdx.z + i) = 0;
                }
            }
            __syncthreads();

            // calculate gradients using central differences
            if (update) {
                du_x[idx] +=
                    0.5 * (shared_u(threadIdx.x, threadIdx.y, threadIdx.z + 2) -
                           shared_u(threadIdx.x, threadIdx.y, threadIdx.z));
            } else {
                du_x[idx] =
                    0.5 * (shared_u(threadIdx.x, threadIdx.y, threadIdx.z + 2) -
                           shared_u(threadIdx.x, threadIdx.y, threadIdx.z));
            }
        }
    }

    template <typename T, bool update = false>
    __global__ void grad_uy_kernel(DevicePtr<const T> u, DevicePtr<T> du_y) {

        // Allocate shared memory for the current block
        dim3 dims = blockDim;
        dims.y += 2;
        SharedMemory<T> shared_u(dims);

        auto idx = Index3D();
        if (idx < u.dims()) {
            // Load data into shared memory
            for (unsigned i = 0; i < 3; ++i) {
                unsigned y_idx = idx.y - 1 + i;
                if (y_idx < u.dims().n2) {
                    shared_u(threadIdx.x, threadIdx.y + i, threadIdx.z) =
                        u(idx.x, y_idx, idx.z);
                } else {
                    shared_u(threadIdx.x, threadIdx.y + i, threadIdx.z) = 0;
                }
            }
            __syncthreads();

            // calculate gradients using central differences
            if (update) {
                du_y[idx] +=
                    0.5 * (shared_u(threadIdx.x, threadIdx.y + 2, threadIdx.z) -
                           shared_u(threadIdx.x, threadIdx.y, threadIdx.z));
            } else {
                du_y[idx] =
                    0.5 * (shared_u(threadIdx.x, threadIdx.y + 2, threadIdx.z) -
                           shared_u(threadIdx.x, threadIdx.y, threadIdx.z));
            }
        }
    }

    template <typename T, bool update = false>
    __global__ void grad_uz_kernel(DevicePtr<const T> u, DevicePtr<T> du_z) {

        // Allocate shared memory for the current block
        dim3 dims = blockDim;
        dims.x += 2;
        SharedMemory<T> shared_u(dims);

        auto idx = Index3D();
        if (idx < u.dims()) {
            // Load data into shared memory
            for (unsigned i = 0; i < 3; ++i) {
                unsigned x_idx = idx.x + i - 1;
                if (x_idx < u.dims().n1) {
                    shared_u(threadIdx.x + i, threadIdx.y, threadIdx.z) =
                        u(x_idx, idx.y, idx.z);
                } else {
                    shared_u(threadIdx.x + i, threadIdx.y, threadIdx.z) = 0;
                }
            }
            __syncthreads();

            // calculate gradients using central differences
            if (update) {
                du_z[idx] +=
                    0.5 * (shared_u(threadIdx.x + 2, threadIdx.y, threadIdx.z) -
                           shared_u(threadIdx.x, threadIdx.y, threadIdx.z));
            } else {
                du_z[idx] =
                    0.5 * (shared_u(threadIdx.x + 2, threadIdx.y, threadIdx.z) -
                           shared_u(threadIdx.x, threadIdx.y, threadIdx.z));
            }
        }
    }

    template <typename T>
    std::array<DeviceArray<T>, 3> grad(const DeviceArray<T> &u) {
        DeviceArray<T> du_x(u.dims());
        DeviceArray<T> du_y(u.dims());
        DeviceArray<T> du_z(u.dims());

        dim3 blockSize(1, 8, 32);
        dim3 gridSize = make_grid(u.dims(), blockSize);
        size_t smemSizeX = blockSize.x * blockSize.y * (blockSize.z + 2) * sizeof(T);
        size_t smemSizeY = blockSize.x * (blockSize.y + 2) * blockSize.z * sizeof(T);
        size_t smemSizeZ = (blockSize.x + 2) * blockSize.y * blockSize.z * sizeof(T);

        cudaStream_t sx, sy, sz;
        SAFE_CALL(cudaStreamCreate(&sx));
        SAFE_CALL(cudaStreamCreate(&sy));
        SAFE_CALL(cudaStreamCreate(&sz));

        // du/dx
        grad_ux_kernel<T, false><<<gridSize, blockSize, smemSizeX, sx>>>(u, du_x);
        // du/dy
        grad_uy_kernel<T, false><<<gridSize, blockSize, smemSizeY, sy>>>(u, du_y);
        // du/dz
        grad_uz_kernel<T, false><<<gridSize, blockSize, smemSizeZ, sz>>>(u, du_z);
        SAFE_CALL(cudaGetLastError());

        SAFE_CALL(cudaStreamSynchronize(sx));
        SAFE_CALL(cudaStreamSynchronize(sy));
        SAFE_CALL(cudaStreamSynchronize(sz));

        SAFE_CALL(cudaStreamDestroy(sx));
        SAFE_CALL(cudaStreamDestroy(sy));
        SAFE_CALL(cudaStreamDestroy(sz));
        std::array<DeviceArray<T>, 3> du;
        du[0] = std::move(du_x);
        du[1] = std::move(du_y);
        du[2] = std::move(du_z);
        return du;
    }
    // Explicit template instantiations
    template std::array<DeviceArray<float>, 3> grad(const DeviceArray<float> &u);
    template std::array<DeviceArray<double>, 3> grad(const DeviceArray<double> &u);

    // divergence
    template <typename T>
    DeviceArray<T> divergence(const std::array<DeviceArray<T>, 3> &u) {

        auto dims = u[0].dims();
        // output array for divergence
        DeviceArray<T> div_u(dims);

        dim3 blockSize(1, 8, 32);
        dim3 gridSize = make_grid(dims, blockSize);

        //  du[0]/dx
        size_t smemSize = blockSize.x * blockSize.y * (blockSize.z + 2) * sizeof(T);
        grad_ux_kernel<T, false><<<gridSize, blockSize, smemSize>>>(u[0], div_u);
        SAFE_CALL(cudaGetLastError());

        // du[1]/dy
        smemSize = blockSize.x * (blockSize.y + 2) * blockSize.z * sizeof(T);
        grad_uy_kernel<T, true><<<gridSize, blockSize, smemSize>>>(u[1], div_u);
        SAFE_CALL(cudaGetLastError());

        // du[2]/dz
        smemSize = (blockSize.x + 2) * blockSize.y * blockSize.z * sizeof(T);
        grad_uz_kernel<T, true><<<gridSize, blockSize, smemSize>>>(u[2], div_u);
        SAFE_CALL(cudaGetLastError());
        return div_u;
    }
    // Explicit template instantiations
    template DeviceArray<float>
    divergence(const std::array<DeviceArray<float>, 3> &u);
    template DeviceArray<double>
    divergence(const std::array<DeviceArray<double>, 3> &u);

    // laplacian
    template <typename T>
    DeviceArray<T> laplacian(const DeviceArray<T> &u) {

        // output array for laplacian
        auto dims = u.dims();

        // compute gradient
        auto temp = grad(u);
        // compute divergence of the gradient
        auto lap_u = divergence(temp);
        return lap_u;
    }
    // Explicit template instantiations
    template DeviceArray<float> laplacian(const DeviceArray<float> &u);
    template DeviceArray<double> laplacian(const DeviceArray<double> &u);

} // namespace tomocam::gpu::opt
