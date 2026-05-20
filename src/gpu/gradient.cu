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

#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu/device_array.h"
#include "gpu/device_ptr.h"
#include "gpu/utils.h"

namespace tomocam::gpu::opt {

    constexpr int Nx = 1;
    constexpr int Ny = 16;
    constexpr int Nz = 16;

    // Computes the central-difference gradient of u.
    //   gz = du/dz  (slice direction)
    //   gy = du/dy  (row direction)
    //   gx = du/dx  (col direction)
    template <typename T>
    __global__ void gradient_kernel(DevicePtr<T> u, DevicePtr<T> gz,
                                    DevicePtr<T> gy, DevicePtr<T> gx) {
        auto idx = Index3D();
        int i = idx.x;
        int j = idx.y;
        int k = idx.z;

        int n1 = (int)u.dims().n1;
        int n2 = (int)u.dims().n2;
        int n3 = (int)u.dims().n3;

        if (i < n1 && j < n2 && k < n3) {
            if (i > 0 && i < n1 - 1)
                gz(i, j, k) = (u(i + 1, j, k) - u(i - 1, j, k)) / T(2);
            if (j > 0 && j < n2 - 1)
                gy(i, j, k) = (u(i, j + 1, k) - u(i, j - 1, k)) / T(2);
            if (k > 0 && k < n3 - 1)
                gx(i, j, k) = (u(i, j, k + 1) - u(i, j, k - 1)) / T(2);
        }
    }

    template <typename T>
    void gradient(const DeviceArray<T> &u, DeviceArray<T> &gz,
                  DeviceArray<T> &gy, DeviceArray<T> &gx) {
        auto d = u.dims();
        dim3 block(Nx, Ny, Nz);
        dim3 dims((unsigned)d.n1, (unsigned)d.n2, (unsigned)d.n3);
        auto grid = make_grid(dims, block);

        gradient_kernel<T><<<grid, block>>>(u, gz, gy, gx);
        SAFE_CALL(cudaGetLastError());
    }

    // instantiate the template
    template void gradient(const DeviceArray<float> &, DeviceArray<float> &,
                           DeviceArray<float> &, DeviceArray<float> &);
    template void gradient(const DeviceArray<double> &, DeviceArray<double> &,
                           DeviceArray<double> &, DeviceArray<double> &);

} // namespace tomocam::gpu::opt
