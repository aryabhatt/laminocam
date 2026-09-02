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
#include <complex>
#include <execution>

#include "array.h"
#include "array_ops.h"
#include "nufft.h"

namespace tomocam::opt {

    // -------------------------------------------------------------------------
    // Base interface: z = P * r  (out-of-place, r unchanged)
    // update_rho() supports ADMM outer loop without rebuilding the spectrum.
    // -------------------------------------------------------------------------
    template <typename T>
    class IPrecond {
      public:
        virtual ~IPrecond() = default;
        virtual void apply(const Array<T> &r, Array<T> &z) const = 0;
        virtual void update_rho(T rho) { (void)rho; }
    };

    template <typename T>
    class IdentityPrecond : public IPrecond<T> {
      public:
        void apply(const Array<T> &r, Array<T> &z) const override { z = r.clone(); }
    };

    // -------------------------------------------------------------------------
    // Density-compensation (Jacobi) preconditioner
    //   w_[i] ~ diag(A^T A)_i, built via Pipe-Menon iterations.
    //   apply: z[i] = r[i] / w_[i]   — O(N), no FFT/NUFFT in hot path.
    // -------------------------------------------------------------------------
    template <typename T>
    class DensityComp : public IPrecond<T> {
      private:
        Array<T> w_;

      public:
        DensityComp(const cpu::PolarGrid<T> &grid, dims_t recon_dims,
                    size_t n_iters = 10, T tol = T(1e-6)) {
            (void)tol; // convergence criterion not yet implemented; use n_iters

            auto w = Array<std::complex<T>>::ones(grid.dims());
            auto u = Array<std::complex<T>>::zeros(recon_dims);
            auto v = Array<std::complex<T>>(grid.dims());

            for (size_t iter = 0; iter < n_iters; iter++) {
                nufft::nufft3d1(w, u, grid); // u = type-1(w): non-uniform → uniform
                nufft::nufft3d2(v, u, grid); // v = type-2(u): uniform → non-uniform
                std::transform(std::execution::par_unseq, w.begin(), w.end(),
                               v.begin(), w.begin(),
                               [](std::complex<T> wj, std::complex<T> vj) {
                                   return wj / std::max(std::abs(vj), T(1e-12));
                               });
            }
            nufft::nufft3d1(w, u, grid);
            auto d = array::to_real(u);
            T floor_val = T(1e-3) * array::max(d);
            w_ = Array<T>(recon_dims);
            std::transform(std::execution::par_unseq, d.begin(), d.end(), w_.begin(),
                           [floor_val](T x) { return std::max(x, floor_val); });
        }

        void apply(const Array<T> &r, Array<T> &z) const override { z = r / w_; }
    };


} // namespace tomocam::opt

#endif // TOMOCAM_PRECOND__H
