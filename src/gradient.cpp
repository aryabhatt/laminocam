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
#include <format>

#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "nufft.h"
#include "polar_grid.h"
#include "tomocam.h"

namespace tomocam {
    template <typename T>
    Array<T> sysmat(const Array<T> &x, const PolarGrid<T> &grid) {
        // cast to complex
        T scale = static_cast<T>(grid.x.nslices() * grid.x.ncols());
        auto xcmplx = array::to_complex(x);
        auto ccmplx = Array<std::complex<T>>(grid.dims());
        nufft::nufft3d2(ccmplx, xcmplx, grid);
        nufft::nufft3d1(ccmplx, xcmplx, grid);
        // return array::to_real(xcmplx) / scale;
        return array::to_real(xcmplx);
    }
    // Explicit instantiations
    template Array<float> sysmat(const Array<float> &, const PolarGrid<float> &);
    template Array<double> sysmat(const Array<double> &, const PolarGrid<double> &);

    // Compute gradient of ||R^T R f - yT||^2
    template <typename T>
    Array<T> gradient(const Array<T> &f, const Array<T> &yT,
                      const PolarGrid<T> &grid) {
        auto AAx = sysmat(f, grid);
        return (AAx - yT);
    }
    // Explicit instantiations
    template Array<float> gradient(const Array<float> &, const Array<float> &,
                                   const PolarGrid<float> &);
    template Array<double> gradient(const Array<double> &, const Array<double> &,
                                    const PolarGrid<double> &);

    // Compute residual ||R^T R f - yT||^2
    template <typename T>
    T residual(const Array<T> &f, const Array<T> &yT, const PolarGrid<T> &grid,
               T yTy) {
        auto AAx = sysmat(f, grid);
        auto xAAx = array::dot(f, AAx);
        auto yTx = array::dot(f, yT);
#ifdef DEBUG
        std::cout << std::format("residual: xAAx = {}, yTx = {}, yTy = {}\n", xAAx,
                                 yTx, yTy);
#endif
        return xAAx - 2.0 * yTx + yTy;
    }

    // Explicit instantiations
    template float residual(const Array<float> &, const Array<float> &,
                            const PolarGrid<float> &, float);
    template double residual(const Array<double> &, const Array<double> &,
                             const PolarGrid<double> &, double);
} // namespace tomocam
