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
    DeviceArray<T> forward(const DeviceArray<T> &volume, const PolarGrid<T> &pg) {

        // cast to complex
        auto Ft = array::to_complex(volume);
        DeviceArray<complex<T>> C(pg.dims());

        // NUFFT type-2: uniform -> nonuniform
        nufft::nufft3d2(C, Ft, pg);

        // ifft with fftshift
        C = fftshift2(C);
        C = fft::ifft2d(C);
        C = ifftshift2(C);

        return array::to_real(C);
    }

    // Explicit instantiations for forward
    template DeviceArray<float> forward(const DeviceArray<float> &,
                                        const PolarGrid<float> &);
    template DeviceArray<double> forward(const DeviceArray<double> &,
                                         const PolarGrid<double> &);

    // -------------------------------------------------------------------------
    // backward: projections -> volume
    // -------------------------------------------------------------------------

    template <typename T>
    DeviceArray<T> backward(const DeviceArray<T> &proj, const PolarGrid<T> &pg,
                            const dims_t &recon_dims) {
        // cast to complex
        auto C = array::to_complex(proj);

        // 2D Fourier transform with fftshift
        C = fftshift2(C);
        C = fft::fft2d(C);
        C = ifftshift2(C);

        DeviceArray<complex<T>> F(recon_dims);
        nufft::nufft3d1(C, F, pg);

        // scale and return real part
        auto result = array::to_real(F);
        T scale = static_cast<T>(proj.ncols()) * static_cast<T>(proj.size());
        array::scale_inplace(result, T(1) / scale);
        return result;
    }

    // Explicit instantiations for backward
    template DeviceArray<float> backward(const DeviceArray<float> &,
                                         const PolarGrid<float> &,
                                         const dims_t &);
    template DeviceArray<double> backward(const DeviceArray<double> &,
                                          const PolarGrid<double> &,
                                          const dims_t &);

} // namespace tomocam::gpu
