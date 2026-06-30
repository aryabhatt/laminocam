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

#include <cuda/std/complex>

#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/nufft.h"
#include "gpu/polar_grid.h"

namespace tomocam::gpu {

    template <typename T>
    DeviceArray<T> sysmat(const DeviceArray<T> &x, const gpu::PolarGrid<T> &grid) {

        // scale
        T scale = static_cast<T>(grid.dims().n2 * grid.dims().n3);

        auto d_fz = gpu::array::to_complex(x);
        auto d_cz = DeviceArray<cuda::std::complex<T>>(grid.dims());
        // type-2 non-uniform FFT
        gpu::nufft::nufft3d2(d_cz, d_fz, grid);
        // type-1 non-uniform FFT
        gpu::nufft::nufft3d1(d_cz, d_fz, grid);
        return gpu::array::to_real(d_fz) / scale;
    }
    // explicit instantiations
    template DeviceArray<float> sysmat(const DeviceArray<float> &x,
                                       const gpu::PolarGrid<float> &grid);
    template DeviceArray<double> sysmat(const DeviceArray<double> &x,
                                        const gpu::PolarGrid<double> &grid);
} // namespace tomocam::gpu
