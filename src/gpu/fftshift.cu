/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 *IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 *the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#include <array>
#include <vector>

#include <cuda/std/complex>
#include <thrust/device_vector.h>

#include "gpu/device_array.h"
#include "gpu/device_ptr.h"
#include "gpu/memory.h"
#include "gpu/utils.h"

namespace tomocam::gpu {

    template <typename T>
    __global__ void phase_shift2d_kernel(DevicePtr<cuda::std::complex<T>> C,
                                         const T *dx, const T *dy) {
        auto dims = C.dims();
        auto idx = Index3D();
        if (!(idx < dims)) return;

        T dqx = 2 * T(M_PI) / static_cast<T>(dims.n3);
        T dqy = 2 * T(M_PI) / static_cast<T>(dims.n2);
        T qx = (idx.z + T(0.5)) * dqx - T(M_PI);
        T qy = (idx.y + T(0.5)) * dqy - T(M_PI);
        T phase = qx * dx[idx.x] + qy * dy[idx.x];
        C[idx] *= cuda::std::complex<T>(cos(-phase), sin(-phase));
    }

    template <typename T>
    void phase_shift2d(DeviceArray<cuda::std::complex<T>> &C,
                       const std::vector<std::array<T, 2>> &shifts) {

        size_t n = shifts.size();
        std::vector<T> dx(n), dy(n);
        for (size_t i = 0; i < n; ++i) {
            dx[i] = shifts[i][0];
            dy[i] = shifts[i][1];
        }
        thrust::device_vector<T> d_dx(dx);
        thrust::device_vector<T> d_dy(dy);

        auto dims = C.dims();
        dim3 threads(1, 16, 16);
        dim3 blocks((dims.n3 + threads.x - 1) / threads.x,
                    (dims.n2 + threads.y - 1) / threads.y,
                    (dims.n1 + threads.z - 1) / threads.z);
        phase_shift2d_kernel<T>
            <<<blocks, threads>>>(C, thrust::raw_pointer_cast(d_dx.data()),
                                  thrust::raw_pointer_cast(d_dy.data()));
        SAFE_CALL(cudaGetLastError());
    }

    template <typename T>
    __global__ void roll_kernel(DevicePtr<T> in, DevicePtr<T> out, int3 delta,
                                int nx, int ny, int nz) {
        int ix = blockIdx.z * blockDim.z + threadIdx.z;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;
        int iz = blockIdx.x * blockDim.x + threadIdx.x;

        if (ix < nx && iy < ny && iz < nz) {
            int ix2 = (ix + delta.x + nx) % nx;
            int iy2 = (iy + delta.y + ny) % ny;
            int iz2 = (iz + delta.z + nz) % nz;
            out(ix, iy, iz) = in(ix2, iy2, iz2);
        }
    }

    template <typename T>
    DeviceArray<T> roll(const DeviceArray<T> &arr, int3 delta) {

        auto d = arr.dims();
        int nx = (int)d.n1;
        int ny = (int)d.n2;
        int nz = (int)d.n3;

        DeviceArray<T> out(d);
        dim3 threads(16, 16, 1);
        dim3 blocks((nz + 15) / 16, (ny + 15) / 16, nx);
        roll_kernel<T><<<blocks, threads>>>(const_cast<DeviceArray<T> &>(arr), out,
                                            delta, nx, ny, nz);
        return out;
    }

    template void
    phase_shift2d(DeviceArray<cuda::std::complex<float>> &,
                  const std::vector<std::array<float, 2>> &);
    template void
    phase_shift2d(DeviceArray<cuda::std::complex<double>> &,
                  const std::vector<std::array<double, 2>> &);

    // explicit instantiation
    template DeviceArray<float> roll(const DeviceArray<float> &arr, int3 delta);
    template DeviceArray<double> roll(const DeviceArray<double> &arr, int3 delta);
    template DeviceArray<cuda::std::complex<float>>
    roll(const DeviceArray<cuda::std::complex<float>> &arr, int3 delta);
    template DeviceArray<cuda::std::complex<double>>
    roll(const DeviceArray<cuda::std::complex<double>> &arr, int3 delta);

} // namespace tomocam::gpu
