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
#ifndef TOMOCAM_PRECOND__H
#define TOMOCAM_PRECOND__H

#include <algorithm>
#include <execution>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "fft.h"

namespace tomocam::opt {
    template <typename T>
    using complex_t = std::complex<T>;

    template <typename T>
    class DensityComp {
      private:
        Array<T> w_;

      public:
        DensityComp(const PolarGrid<T> &grid, dims_t recon_dims, size_t n_iters = 10,
                    tol = 1e-6) {
            size_t M = grid.npts;
            dims_t dims{1, 1, M};
            auto w = array::ones<complex_t<T>>(dims);
            auto u = array::zeros<complex_t<T>>(recon_dims);
            w_(recon_dims);

            for (size_t i = 0; i < n_iters; i++) {
                nufft::nufft3d1(w, u, grid);
                nufft::nufft3d2(w, u, grid);
                auto d = array::max(array::abs(u), 1.0e-12);
                w = array::div(u, d);
            }
            nufft::nufft3d1(w, u, grid);
            auto d = array::real(u);
            auto floor = 1.0e-03 * array::max(d);
            std::transform(d.begin(), d.end(), w_.begin(),
                           [floor](T x) { return std::max(x, floor); });
        }

        void apply(Array<T> &a) const {
            std::transform(std::execution::par_unseq, a.begin(), a.end(), w_.begin(),
                           a.begin(), [](T x, T y) { return x / y; });
        }
    };

    template <typename T>
    class ToeplitzPrecond {
      private:
        Array<T> w_;

      public:
        ToeplitzPrecond(const PolarGrid<T> &grid, dims_t recon_dims) {
            size_t M = grid.npts;
            dims_t dims{1, 1, M};
            auto w = array::ones<complex_t<T>>(dims);
            auto u = array::zeros<complex_t<T>>(recon_dims);

            // compute point spread function
            nufft::nufft3d1(w, u, grid);
            nufft::nufft3d2(w, u, grid);

            auto d = array::max(array::abs(u), 1.0e-12);
            w = array::div(u, d);
            nufft::nufft3d1(w, u, grid);
            auto d = array::real(u);
            auto floor = 1.0e-03 * array::max(d);
            std::transform(d.begin(), d.end(), w_.begin(),
                           [floor](T x) { return std::max(x, floor); });
        }
    };
} // namespace tomocam::opt

#endif // TOMOCAM_PRECOND__H
