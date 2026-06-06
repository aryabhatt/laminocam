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

#include <execution>

#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "fft.h"
#include "fftutils.h"
#include "filter.h"
#include "nufft.h"
#include "padding.h"
#include "polar_grid.h"
#include "tomocam.h"
#include <cstdint>

constexpr double factor = 1.4142135624;

namespace tomocam {

    template <typename T>
    Array<T> forward(const Array<T> &volume, const PolarGrid<T> &pg, T gamma) {

        // cast double array to complex
        auto Ft = array::to_complex(volume);
        Array<std::complex<T>> C(pg.dims());

        // call NUFFT3d type-2
        nufft::nufft3d2<T>(C, Ft, pg);

        // ifft
        C = fft::fftshift2(C);
        C = fft::ifft2(C);
        C = fft::ifftshift2(C);

        return array::to_real<T>(C);
    }
    // Explicit instantiation forward
    template Array<float> forward(const Array<float> &, const PolarGrid<float> &,
                                  float);
    template Array<double> forward(const Array<double> &, const PolarGrid<double> &,
                                   double);

    template <typename T>
    Array<T> backward(const Array<T> &proj, const PolarGrid<T> &pg,
                      const dims_t &recon_dims, T gamma, bool use_filter,
                      const std::string &filter_type) {

        // cast to complex
        auto C = array::to_complex(proj);

        // 2-D Fourier transforms
        C = fft::fftshift2(C);
        C = fft::fft2(C);
        C = fft::ifftshift2(C);

        // apply density compensation
        std::transform(std::execution::par_unseq, C.begin(), C.end(), pg.w.begin(),
                       C.begin(), [](std::complex<T> c, T w) { return c * w; });

        // nufft
        Array<std::complex<T>> F(recon_dims);
        nufft::nufft3d1<T>(C, F, pg);

        // crop image
        // scale
        T scale = static_cast<T>(proj.ncols() * proj.nrows());
        return array::to_real<T>(F) / scale;
    }

    // Explicit instantiation backward
    template Array<float> backward(const Array<float> &, const PolarGrid<float> &,
                                   const dims_t &, float, bool, const std::string &);
    template Array<double> backward(const Array<double> &, const PolarGrid<double> &,
                                    const dims_t &, double, bool,
                                    const std::string &);

} // namespace tomocam
