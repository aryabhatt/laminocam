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

    // Computes the negative Laplacian of u using a 7-point stencil:
    //   lap(i,j,k) = -(u(i+1,j,k) + u(i-1,j,k)
    //                + u(i,j+1,k) + u(i,j-1,k)
    //                + u(i,j,k+1) + u(i,j,k-1) - 6*u(i,j,k))
    // Interior elements only; boundary elements are left at their current value
    // (caller is responsible for pre-zeroing output array).
    template <typename T>
    __global__ void laplacian_kernel(DevicePtr<T> u, DevicePtr<T> lap) {
        auto idx = Index3D();
        // With block(Nx=1, Ny=16, Nz=16) and grid dims (n1, n2, n3):
        //   idx.x -> slice (n1), idx.y -> row (n2), idx.z -> col (n3)
        int i = idx.x;
        int j = idx.y;
        int k = idx.z;

        int n1 = (int)u.dims().n1;
        int n2 = (int)u.dims().n2;
        int n3 = (int)u.dims().n3;

        if (i > 0 && i < n1 - 1 && j > 0 && j < n2 - 1 && k > 0 &&
            k < n3 - 1) {
            lap(i, j, k) = -(u(i + 1, j, k) + u(i - 1, j, k) +
                             u(i, j + 1, k) + u(i, j - 1, k) +
                             u(i, j, k + 1) + u(i, j, k - 1) -
                             T(6) * u(i, j, k));
        }
    }

    template <typename T>
    void laplacian(const DeviceArray<T> &u, DeviceArray<T> &lap) {
        auto d = u.dims();
        dim3 block(Nx, Ny, Nz);
        dim3 dims((unsigned)d.n1, (unsigned)d.n2, (unsigned)d.n3);
        auto grid = make_grid(dims, block);

        laplacian_kernel<T><<<grid, block>>>(u, lap);
        SAFE_CALL(cudaGetLastError());
    }

    // instantiate the template
    template void laplacian(const DeviceArray<float> &, DeviceArray<float> &);
    template void laplacian(const DeviceArray<double> &, DeviceArray<double> &);

} // namespace tomocam::gpu::opt
