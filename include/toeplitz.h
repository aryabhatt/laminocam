#ifndef TOEPLITZ_H
#define TOEPLITZ_H

#include <array>
#include <complex>
#include <cstdint>

#include "array.h"
#include "fft.h"
#include "finufft_plan.h"
#include "padding.h"
#include "polar_grid.h"

namespace tomocam {

    template <typename T>
    class PointSpreadFunction {
      private:
        using complex_t = std::complex<T>;
        dims_t dims_; // full real-space dims of the oversampled kernel
        Array<complex_t> kernel_hat_; // R2C output: shape {n1, n2, n3/2+1}

      public:
        PointSpreadFunction() = default;

        PointSpreadFunction(const PolarGrid<T> &grid, dims_t proj_dims,
                            dims_t recon_dims) {

            dims_ = {2 * recon_dims.n1 - 1, 2 * recon_dims.n2 - 1,
                     2 * recon_dims.n3 - 1};
            dims_t fft_dims = {dims_.n1, dims_.n2, dims_.n3 / 2 + 1};

            // unit weights at non-uniform grid points
            auto ones = Array<complex_t>::ones(proj_dims);

            // allocate output array for NUFFT
            Array<complex_t> nufft_out(dims_);

            // one-shot NUFFT (not cached): non-uniform → oversampled uniform grid
            std::array<int64_t, 3> n_modes = {static_cast<int64_t>(dims_.n3),
                                              static_cast<int64_t>(dims_.n2),
                                              static_cast<int64_t>(dims_.n1)};

            nufft::FinufftPlanWrapper<T> nufft_plan(1, 3, n_modes, 1);
            nufft_plan.set_points(grid);
            nufft_plan.execute(ones.begin(), nufft_out.begin());

            // PSF is real for a symmetric grid; extract real part before R2C FFT
            auto kernel_real = array::to_real(nufft_out);

            // 3D R2C FFT → half-complex kernel stored as member
            kernel_hat_ = Array<complex_t>(fft_dims);
            std::array<int, 3> n = {(int)dims_.n1, (int)dims_.n2, (int)dims_.n3};
            auto &fft_plan =
                fft::plans::cache<T>.get_plan(3, n, 1, fft::FftwType::R2C);
            fft_plan.execute(kernel_real.begin(), kernel_hat_.begin());
        }

        Array<T> convolve(const Array<T> &input) const {
            auto orig_dims = input.dims();

            // zero-pad input to match PSF size
            dims_t pad_dims = dims_;
            auto padded = pad3d(input, pad_dims, PadType::RIGHT);

            // 3D R2C forward FFT
            dims_t fft_dims = {pad_dims.n1, pad_dims.n2, pad_dims.n3 / 2 + 1};
            Array<complex_t> output_hat(fft_dims);
            std::array<int, 3> n = {(int)pad_dims.n1, (int)pad_dims.n2,
                                    (int)pad_dims.n3};
            auto &fft_plan =
                fft::plans::cache<T>.get_plan(3, n, 1, fft::FftwType::R2C);
            fft_plan.execute(padded.begin(), output_hat.begin());

            // multiply in Fourier space
            output_hat *= kernel_hat_;

            // 3D C2R inverse FFT
            Array<T> result(pad_dims);
            auto &ifft_plan =
                fft::plans::cache<T>.get_plan(3, n, 1, fft::FftwType::C2R);
            ifft_plan.execute(output_hat.begin(), result.begin());

            // FFTW is unnormalized; NUFFT sysmat (gradient.cpp) divides by n2*n3
            T norm = static_cast<T>(pad_dims.n1 * pad_dims.n2 * pad_dims.n3) *
                     static_cast<T>(orig_dims.n2 * orig_dims.n3);

            // FFTW circular conv: z[k+(N-1)] = N_pad*scale * out_nufft[k],
            // so the valid output starts at offset (N1-1, N2-1, N3-1) = crop_size.
            return crop3d(result, orig_dims, PadType::RIGHT) / norm;
        }
    };
} // namespace tomocam

#endif // TOEPLITZ_H
