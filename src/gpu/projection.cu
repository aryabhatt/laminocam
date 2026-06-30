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

#include <cuda/std/complex>

#include "dtypes.h"
#include "polar_grid.h"

#include "gpu/cufinufft_plan_cache.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/fft.h"
#include "gpu/fftshift.h"
#include "gpu/nufft.h"
#include "gpu/polar_grid.h"
#include "gpu/utils.h"

namespace tomocam::gpu {

    template <typename T>
    using complex = cuda::std::complex<T>;

    template <typename T>
    DeviceArray<T> forward(const DeviceArray<T> &volume,
                           const gpu::PolarGrid<T> &pg) {

        auto dims = pg.dims();
        T scale = static_cast<T>(dims.n2 * dims.n3);

        // cast to complex
        auto Ft = gpu::array::to_complex(volume);
        DeviceArray<complex<T>> C(pg.dims());

        // NUFFT type-2: uniform -> nonuniform
        nufft::nufft3d2(C, Ft, pg);

        // ifft with fftshift
        C = gpu::fftshift2(C);
        C = gpu::fft::ifft2d(C);
        C = gpu::ifftshift2(C);

        return gpu::array::to_real(C) / scale;
    }

    // Explicit instantiations for forward
    template DeviceArray<float> forward(const DeviceArray<float> &,
                                        const gpu::PolarGrid<float> &);
    template DeviceArray<double> forward(const DeviceArray<double> &,
                                         const gpu::PolarGrid<double> &);

    // -------------------------------------------------------------------------
    // backward: projections -> volume
    // -------------------------------------------------------------------------

    template <typename T>
    DeviceArray<T> backward(const DeviceArray<T> &proj, const gpu::PolarGrid<T> &pg,
                            const dims_t &recon_dims) {

        // scale
        T scale = static_cast<T>(proj.nrows() * proj.ncols());
        // cast to complex
        auto C = array::to_complex(proj);

        // 2D Fourier transform with fftshift
        C = gpu::fftshift2(C);
        C = gpu::fft::fft2d(C);
        C = gpu::ifftshift2(C);

        DeviceArray<complex<T>> F(recon_dims);
        nufft::nufft3d1(C, F, pg);

        // scale and return real part
        return array::to_real(F) / scale;
    }

    // Explicit instantiations for backward
    template DeviceArray<float> backward(const DeviceArray<float> &,
                                         const gpu::PolarGrid<float> &,
                                         const dims_t &);
    template DeviceArray<double> backward(const DeviceArray<double> &,
                                          const gpu::PolarGrid<double> &,
                                          const dims_t &);

} // namespace tomocam::gpu
