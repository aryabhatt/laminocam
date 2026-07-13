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

#ifndef GPU_TOEPLITZ_H
#define GPU_TOEPLITZ_H

#include <array>
#include <cuda/std/complex>
#include <thrust/fill.h>

#include "dtypes.h"
#include "gpu/cufinufft_plan.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/fft.h"
#include "gpu/padding.h"
#include "gpu/polar_grid.h"
#include "padding.h"
#include "toeplitz.h"

namespace tomocam::gpu {

    template <typename T>
    class PointSpreadFunction {
      private:
        using complex_t = cuda::std::complex<T>;
        dims_t dims_;
        DeviceArray<complex_t> kernel_hat_;

        // Returns the smallest n' >= n of the form 2^p * 3^q (fast for cuFFT).
        static size_t next_fast_dim(size_t n) {
            size_t best = 1;
            while (best < n) best <<= 1;
            for (size_t q3 = 3; q3 < best; q3 *= 3) {
                size_t x = q3;
                while (x < n) x <<= 1;
                if (x < best) best = x;
            }
            return best;
        }

      public:
        PointSpreadFunction() = default;

        // copy CPU computed PSF to GPU for repeated use in convolve()
        PointSpreadFunction(dims_t dims, const DeviceArray<complex_t> &kernel_hat)
            : dims_(dims), kernel_hat_(kernel_hat) {}

        // upload a CPU-computed kernel hat (from FINUFFT) to GPU to save device
        // memory; std::complex<T> and cuda::std::complex<T> are ABI-compatible
        PointSpreadFunction(dims_t dims, const Array<std::complex<T>> &cpu_kernel)
            : dims_(dims) {
            static_assert(sizeof(std::complex<T>) == sizeof(complex_t),
                          "complex type ABI mismatch");
            kernel_hat_ = DeviceArray<complex_t>(cpu_kernel.dims());
            SAFE_CALL(cudaMemcpy(kernel_hat_.data(), cpu_kernel.data(),
                                 cpu_kernel.size() * sizeof(complex_t),
                                 cudaMemcpyHostToDevice));
        }

        // construct PSF from a polar grid and store its R2C FFT for repeated use in
        // convolve()
        PointSpreadFunction(const gpu::PolarGrid<T> &grid, dims_t recon_dims) {

            dims_t proj_dims = grid.dims();
            dims_ = {next_fast_dim(2 * recon_dims.n1 - 1),
                     next_fast_dim(2 * recon_dims.n2 - 1),
                     next_fast_dim(2 * recon_dims.n3 - 1)};
#ifdef DEBUG
            std::cout << std::format(
                "Toeplitz PSF: recon_dims={} -> proj_dims={} -> padded dims={}\n",
                recon_dims.to_string(), proj_dims.to_string(), dims_.to_string());
#endif

            // One-shot plan (not cached): used exactly once for PSF construction.
            auto ones = DeviceArray<complex_t>(proj_dims);
            thrust::fill(ones.begin(), ones.end(), complex_t{T(1), T(0)});

            DeviceArray<complex_t> nufft_out(dims_);
            {
                int gpu_id;
                SAFE_CALL(cudaGetDevice(&gpu_id));
                std::array<int64_t, 3> n_modes = {static_cast<int64_t>(dims_.n3),
                                                  static_cast<int64_t>(dims_.n2),
                                                  static_cast<int64_t>(dims_.n1)};
                nufft::cuFinfftPlanWrapper<T> plan(1, 3, n_modes, 1, gpu_id);
                plan.set_points(grid);
                plan.execute(ones.data(), nufft_out.data());
            }

            // PSF is real for a symmetric grid; extract real part before R2C FFT
            auto kernel_real = gpu::array::to_real(nufft_out);

            // store R2C FFT of kernel for repeated use in convolve()
            kernel_hat_ = gpu::fft::rfft3d(kernel_real);
        }

        /// Compute A^T A x via Toeplitz convolution with the precomputed PSF.
        DeviceArray<T> convolve(const DeviceArray<T> &input) const {
            auto orig_dims = input.dims();

            auto padded = gpu::pad3d(input, dims_, PadType::RIGHT);
            auto output_hat = gpu::fft::rfft3d(padded);
            output_hat *= kernel_hat_;
            auto result = gpu::fft::irfft3d(output_hat, dims_);

            // cuFFT is unnormalized; match the NUFFT sysmat scale (n2*n3)
            T norm = static_cast<T>(dims_.n1 * dims_.n2 * dims_.n3) *
                     static_cast<T>(orig_dims.n2 * orig_dims.n3);

            // The FINUFFT type-1 PSF is in CMCL order: center at dims_/2.
            // Valid linear-conv output starts at that same offset.
            dims_t center{dims_.n1 / 2, dims_.n2 / 2, dims_.n3 / 2};
            return gpu::crop3d(result, orig_dims, center) / norm;
        }
    };

} // namespace tomocam::gpu

#endif // GPU_TOEPLITZ_H
