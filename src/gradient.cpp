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
#include <complex>
#include <execution>

#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "nufft.h"
#include "polar_grid.h"
#include "toeplitz.h"
#include "tomocam.h"

namespace tomocam {
    template <typename T>
    Array<T> sysmat(const Array<T> &x, const cpu::PolarGrid<T> &grid) {

        T scale = static_cast<T>(grid.dims().n2 * grid.dims().n3);
        auto xcmplx = array::to_complex(x);
        auto ccmplx = Array<std::complex<T>>(grid.dims());
        nufft::nufft3d2(ccmplx, xcmplx, grid);
        nufft::nufft3d1(ccmplx, xcmplx, grid);
        return array::to_real(xcmplx) / scale;
    }
    // Explicit instantiations
    template Array<float> sysmat(const Array<float> &,
                                 const cpu::PolarGrid<float> &);
    template Array<double> sysmat(const Array<double> &,
                                  const cpu::PolarGrid<double> &);

    // use Toeplitz method to compute the system matrix
    template <typename T>
    Array<T> sysmat(const Array<T> &x, const cpu::PointSpreadFunction<T> &psf) {
        return psf.convolve(x);
    }
    template Array<float> sysmat(const Array<float> &,
                                 const cpu::PointSpreadFunction<float> &);
    template Array<double> sysmat(const Array<double> &,
                                  const cpu::PointSpreadFunction<double> &);

    // 2D slice-by-slice normal operator (vertical rotation axis): direct NUFFT path
    template <typename T>
    Array<T> sysmat2d(const Array<T> &x, const cpu::PolarGrid2D<T> &grid) {

        T scale = static_cast<T>(grid.nradial());
        auto xcmplx = array::to_complex(x);
        auto ccmplx =
            Array<std::complex<T>>(dims_t{x.nslices(), grid.nangles(), grid.nradial()});
        nufft::nufft2d2(ccmplx, xcmplx, grid);
        nufft::nufft2d1(ccmplx, xcmplx, grid);
        return array::to_real(xcmplx) / scale;
    }
    template Array<float> sysmat2d(const Array<float> &,
                                   const cpu::PolarGrid2D<float> &);
    template Array<double> sysmat2d(const Array<double> &,
                                    const cpu::PolarGrid2D<double> &);

    // 2D slice-by-slice normal operator: Toeplitz (PSF convolution) path
    template <typename T>
    Array<T> sysmat2d(const Array<T> &x, const cpu::PointSpreadFunction2D<T> &psf) {
        return psf.convolve(x);
    }
    template Array<float> sysmat2d(const Array<float> &,
                                   const cpu::PointSpreadFunction2D<float> &);
    template Array<double> sysmat2d(const Array<double> &,
                                    const cpu::PointSpreadFunction2D<double> &);
} // namespace tomocam
