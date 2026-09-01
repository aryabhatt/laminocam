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
#include "toeplitz.h"

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
                std::transform(std::execution::par_unseq,
                               w.begin(), w.end(), v.begin(), w.begin(),
                               [](std::complex<T> wj, std::complex<T> vj) {
                                   return wj / std::max(std::abs(vj), T(1e-12));
                               });
            }
            nufft::nufft3d1(w, u, grid);
            auto d = array::to_real(u);
            T floor_val = T(1e-3) * array::max(d);
            w_ = Array<T>(recon_dims);
            std::transform(std::execution::par_unseq,
                           d.begin(), d.end(), w_.begin(),
                           [floor_val](T x) { return std::max(x, floor_val); });
        }

        void apply(const Array<T> &r, Array<T> &z) const override { z = r / w_; }
    };

    // -------------------------------------------------------------------------
    // Toeplitz spectral preconditioner
    //   P^{-1}(r) = IFFT(r_hat / t_hat)  where t_hat = FFT(PSF).
    //   Follows the same pad/FFT/crop/norm pattern as PointSpreadFunction::convolve.
    //   update_rho() handles the ADMM augmented operator A + rho*I by updating
    //   the spectral inverse to 1/(t_hat + rho + reg) — no FFT recomputation.
    // -------------------------------------------------------------------------
    template <typename T>
    class ToeplitzPrecond : public IPrecond<T> {
      private:
        using complex_t = std::complex<T>;
        dims_t vol_dims_;
        dims_t pad_dims_;
        Array<complex_t> t_hat_;     // PSF spectrum, R2C shape {n1, n2, n3/2+1}
        Array<complex_t> t_hat_inv_; // current 1/(t_hat + rho + reg)
        T reg_;

        void build_inverse(T rho) {
            std::transform(
                std::execution::par_unseq,
                t_hat_.begin(), t_hat_.end(), t_hat_inv_.begin(),
                [rho, reg = reg_](const complex_t &h) -> complex_t {
                    return complex_t(T(1) / (std::real(h) + rho + reg), T(0));
                });
        }

      public:
        // Preferred: reuse an existing PointSpreadFunction (no second NUFFT call).
        ToeplitzPrecond(const cpu::PointSpreadFunction<T> &psf, dims_t vol_dims,
                        T reg = T(1e-2))
            : vol_dims_(vol_dims), pad_dims_(psf.dims()),
              t_hat_(psf.kernel_hat().clone()), reg_(reg) {
            dims_t fft_dims = {pad_dims_.n1, pad_dims_.n2, pad_dims_.n3 / 2 + 1};
            t_hat_inv_ = Array<complex_t>(fft_dims);
            build_inverse(T(0));
        }

        // Fallback: build the PSF internally (one extra NUFFT call).
        ToeplitzPrecond(const cpu::PolarGrid<T> &grid, dims_t vol_dims,
                        T reg = T(1e-2))
            : vol_dims_(vol_dims), reg_(reg) {
            cpu::PointSpreadFunction<T> psf(grid, vol_dims);
            pad_dims_ = psf.dims();
            t_hat_ = psf.kernel_hat().clone();
            dims_t fft_dims = {pad_dims_.n1, pad_dims_.n2, pad_dims_.n3 / 2 + 1};
            t_hat_inv_ = Array<complex_t>(fft_dims);
            build_inverse(T(0));
        }

        void update_rho(T rho) override { build_inverse(rho); }

        // Apply: z = P^{-1}(r)  via pad → R2C FFT → multiply t_hat_inv → C2R IFFT → crop.
        // Normalization matches PointSpreadFunction::convolve so that convolve(apply(r)) ≈ r.
        void apply(const Array<T> &r, Array<T> &z) const override {
            auto r_pad = pad3d(r, pad_dims_, PadType::RIGHT);

            dims_t fft_dims = {pad_dims_.n1, pad_dims_.n2, pad_dims_.n3 / 2 + 1};
            Array<complex_t> r_hat(fft_dims);
            std::array<int, 3> n = {(int)pad_dims_.n1, (int)pad_dims_.n2,
                                    (int)pad_dims_.n3};
            auto &fft_plan =
                fft::plans::cache<T>.get_plan(3, n, 1, fft::FftwType::R2C);
            fft_plan.execute(r_pad.begin(), r_hat.begin());

            r_hat *= t_hat_inv_;

            Array<T> result(pad_dims_);
            auto &ifft_plan =
                fft::plans::cache<T>.get_plan(3, n, 1, fft::FftwType::C2R);
            ifft_plan.execute(r_hat.begin(), result.begin());

            // same normalization as PointSpreadFunction::convolve
            T norm = T(pad_dims_.n1 * pad_dims_.n2 * pad_dims_.n3) *
                     T(vol_dims_.n2 * vol_dims_.n3);
            dims_t center{pad_dims_.n1 / 2, pad_dims_.n2 / 2, pad_dims_.n3 / 2};
            z = crop3d(result, vol_dims_, center) / norm;
        }
    };

} // namespace tomocam::opt

#endif // TOMOCAM_PRECOND__H
