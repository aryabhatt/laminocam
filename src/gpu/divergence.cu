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

    // Computes the negative divergence of vector field (vz, vy, vx):
    //   div(i,j,k) = -[ d(vz)/dn1 + d(vy)/dn2 + d(vx)/dn3 ]
    // using central differences. Interior elements only; boundary elements are
    // left at their current value (caller is responsible for pre-zeroing output).
    template <typename T>
    __global__ void divergence_kernel(DevicePtr<T> vz, DevicePtr<T> vy,
                                      DevicePtr<T> vx, DevicePtr<T> div) {
        auto idx = Index3D();
        int i = idx.x;
        int j = idx.y;
        int k = idx.z;

        int n1 = (int)vz.dims().n1;
        int n2 = (int)vz.dims().n2;
        int n3 = (int)vz.dims().n3;

        SharedMemory<T> s_mem(dim3(Nx + 2, Ny + 2, Nz + 2));

        if (i > 0 && i < n1 - 1 && j > 0 && j < n2 - 1 && k > 0 && k < n3 - 1) {
            div(i, j, k) = -(vz(i + 1, j, k) - vz(i - 1, j, k)) / T(2) -
                           (vy(i, j + 1, k) - vy(i, j - 1, k)) / T(2) -
                           (vx(i, j, k + 1) - vx(i, j, k - 1)) / T(2);
        }
    }

    template <typename T>
    void divergence(const DeviceArray<T> &vz, const DeviceArray<T> &vy,
                    const DeviceArray<T> &vx, DeviceArray<T> &div) {
        auto d = vz.dims();
        dim3 block(Nx, Ny, Nz);
        dim3 dims((unsigned)d.n1, (unsigned)d.n2, (unsigned)d.n3);
        auto grid = make_grid(dims, block);
        size_t shared_mem_size =
            (size_t) * ((Nx + 2) * (Ny + 2) * (Nz + 2) * sizeof(T));

        divergence_kernel<T><<<grid, block, shared_mem_size>>>(vz, vy, vx, div);
        SAFE_CALL(cudaGetLastError());
    }

    // instantiate the template
    template void divergence(const DeviceArray<float> &, const DeviceArray<float> &,
                             const DeviceArray<float> &, DeviceArray<float> &);
    template void divergence(const DeviceArray<double> &,
                             const DeviceArray<double> &,
                             const DeviceArray<double> &, DeviceArray<double> &);

} // namespace tomocam::gpu::opt
