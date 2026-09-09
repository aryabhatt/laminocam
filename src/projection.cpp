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

#include <algorithm>
#include <array>
#include <cstdint>
#include <execution>
#include <stdexcept>

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

constexpr double factor = 1.4142135624;

namespace tomocam {

    template <typename T>
    Array<T> forward(const Array<T> &volume, const cpu::PolarGrid<T> &pg) {

        // cast double array to complex
        auto Ft = array::to_complex(volume);
        Array<std::complex<T>> C(pg.dims());

        // call NUFFT3d type-2
        nufft::nufft3d2<T>(C, Ft, pg);

        // exclude Fourier coefficients outside |q| < pi
        std::transform(std::execution::par_unseq, pg.w.begin(), pg.w.end(), C.data(),
                       C.data(), std::multiplies<std::complex<T>>());

        // ifft
        C = fft::fftshift2(C);
        C = fft::ifft2(C);
        C = fft::ifftshift2(C);

        // scale by image size
        T scale = static_cast<T>(pg.dims().n2 * pg.dims().n3);
        return array::to_real<T>(C) / scale;
    }
    // Explicit instantiation forward
    template Array<float> forward(const Array<float> &,
                                  const cpu::PolarGrid<float> &);
    template Array<double> forward(const Array<double> &,
                                   const cpu::PolarGrid<double> &);

    template <typename T>
    Array<T> backproj(const Array<T> &proj, const cpu::PolarGrid<T> &pg,
                      const dims_t &recon_dims) {
        // cast to complex
        auto C = array::to_complex(proj);

        // 2-D Fourier transforms
        C = fft::fftshift2(C);
        C = fft::fft2(C);
        C = fft::ifftshift2(C);

        // exclude Fourier coefficients outside |q| < pi
        std::transform(std::execution::par_unseq, pg.w.begin(), pg.w.end(), C.data(),
                       C.data(), std::multiplies<std::complex<T>>());

        // nufft
        Array<std::complex<T>> F(recon_dims);
        nufft::nufft3d1<T>(C, F, pg);

        // scale by image size
        T scale = static_cast<T>(pg.dims().n2 * pg.dims().n3);
        return array::to_real<T>(F) / scale;
    }
    // Explicit instantiation backproj
    template Array<float> backproj(const Array<float> &,
                                   const cpu::PolarGrid<float> &, const dims_t &);
    template Array<double> backproj(const Array<double> &,
                                    const cpu::PolarGrid<double> &, const dims_t &);

    template <typename T>
    Array<T> backproj(const Array<T> &proj, const cpu::PolarGrid<T> &pg,
                      const dims_t &recon_dims,
                      const std::vector<std::array<T, 2>> &shifts) {

        if (shifts.size() != proj.nslices()) {
            throw std::invalid_argument(
                "shifts size must match the number of slices in the projection");
        }
        // cast to complex
        auto C = array::to_complex(proj);

        // 2-D Fourier transforms
        C = fft::fftshift2(C);
        C = fft::fft2(C);
        C = fft::ifftshift2(C);

        // center of rotation corrections: multiply C by exp(i*(kx*dx + ky*dy))
        fft::phase_shift2d(C, shifts);

        // multiply by pg.w exclude coefficients outside | q | < pi
        std::transform(std::execution::par_unseq, pg.w.begin(), pg.w.end(), C.data(),
                       C.data(), std::multiplies<std::complex<T>>());

        // nufft
        Array<std::complex<T>> F(recon_dims);
        nufft::nufft3d1<T>(C, F, pg);

        // scale by image size
        T scale = static_cast<T>(pg.dims().n2 * pg.dims().n3);
        return array::to_real<T>(F) / scale;
    }

    // Explicit instantiation backproj
    template Array<float> backproj(const Array<float> &,
                                   const cpu::PolarGrid<float> &, const dims_t &,
                                   const std::vector<std::array<float, 2>> &);
    template Array<double> backproj(const Array<double> &,
                                    const cpu::PolarGrid<double> &, const dims_t &,
                                    const std::vector<std::array<double, 2>> &);

    // 2D slice-by-slice tomography (vertical rotation axis): forward
    // projection. Every slice shares the same polar grid, so nufft2d2
    // broadcasts the NUFFT over the outer (slice) axis; fft1/fftshift1
    // already treat that axis (and the angle axis) as a batch and apply
    // the 1-D radial transform independently to every row.
    template <typename T>
    Array<T> forward2d(const Array<T> &volume, const cpu::PolarGrid2D<T> &pg) {
        auto Ft = array::to_complex(volume);
        Array<std::complex<T>> C(
            dims_t{volume.nslices(), pg.nangles(), pg.nradial()});

        nufft::nufft2d2<T>(C, Ft, pg);

        C = fft::ifftshift1(C);
        C = fft::ifft1(C);
        C = fft::fftshift1(C);

        T scale = static_cast<T>(pg.nradial());
        return array::to_real<T>(C) / scale;
    }
    // Explicit instantiation forward2d
    template Array<float> forward2d(const Array<float> &,
                                    const cpu::PolarGrid2D<float> &);
    template Array<double> forward2d(const Array<double> &,
                                     const cpu::PolarGrid2D<double> &);

    // 2D slice-by-slice tomography: unfiltered backprojection.
    template <typename T>
    Array<T> backproj2d(const Array<T> &sino, const cpu::PolarGrid2D<T> &pg,
                        const dims_t &recon_dims) {
        auto C = array::to_complex(sino);

        C = fft::fftshift1(C);
        C = fft::fft1(C);
        C = fft::ifftshift1(C);

        Array<std::complex<T>> F(recon_dims);
        nufft::nufft2d1<T>(C, F, pg);

        T scale = static_cast<T>(pg.nradial());
        return array::to_real<T>(F) / scale;
    }
    // Explicit instantiation backproj2d
    template Array<float> backproj2d(const Array<float> &,
                                     const cpu::PolarGrid2D<float> &,
                                     const dims_t &);
    template Array<double> backproj2d(const Array<double> &,
                                      const cpu::PolarGrid2D<double> &,
                                      const dims_t &);

} // namespace tomocam
