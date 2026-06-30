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
    Array<T> sysmat(const Array<T> &x, const PolarGrid<T> &grid) {

        T scale = static_cast<T>(grid.dims().n2 * grid.dims().n3);
        auto xcmplx = array::to_complex(x);
        auto ccmplx = Array<std::complex<T>>(grid.dims());
        nufft::nufft3d2(ccmplx, xcmplx, grid);
        std::transform(std::execution::par_unseq, ccmplx.begin(), ccmplx.end(),
                       grid.w.begin(), ccmplx.begin(),
                       [](const std::complex<T> &c, T w) { return c * w; });
        nufft::nufft3d1(ccmplx, xcmplx, grid);
        return array::to_real(xcmplx) / scale;
    }
    // Explicit instantiations
    template Array<float> sysmat(const Array<float> &, const PolarGrid<float> &);
    template Array<double> sysmat(const Array<double> &, const PolarGrid<double> &);

    // use Toeplitz method to compute the system matrix
    template <typename T>
    Array<T> sysmat(const Array<T> &x, const PointSpreadFunction<T> &psf) {
        return psf.convolve(x);
    }
    template Array<float> sysmat(const Array<float> &,
                                 const PointSpreadFunction<float> &);
    template Array<double> sysmat(const Array<double> &,
                                  const PointSpreadFunction<double> &);
} // namespace tomocam
