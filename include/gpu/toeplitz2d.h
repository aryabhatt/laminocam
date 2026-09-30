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

#ifndef GPU_TOEPLITZ2D_H
#define GPU_TOEPLITZ2D_H

#include <algorithm>
#include <array>
#include <cuda/std/complex>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include "dtypes.h"
#include "gpu/cufinufft_plan.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/fft.h"
#include "gpu/padding.h"
#include "gpu/polar_grid2d.h"
#include "padding.h"

namespace tomocam::gpu {

    // Toeplitz normal-operator kernel for 2D (vertical rotation axis)
    // tomography. A single 2D PSF is built once from the polar grid and
    // broadcast-convolved independently across every slice of a 3D volume.
    template <typename T>
    class PointSpreadFunction2D {
      private:
        using complex_t = cuda::std::complex<T>;
        size_t psf_n_;
        DeviceArray<complex_t> kernel_hat_; // one 2D kernel: {1, psf_n_, psf_n_/2+1}
        // Upper bound on slices convolved at once. Fixed for the lifetime of
        // the PSF so the batched cuFFT plans (keyed on batch size) are reused
        // from the plan cache instead of re-created on every call.
        size_t max_partition_ = 0;

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

        // multiply every slice of `data` by the single shared kernel_hat_
        void broadcast_multiply(DeviceArray<complex_t> &data) const {
            size_t slice_sz = data.nrows() * data.ncols();
            for (size_t i = 0; i < data.nslices(); ++i) {
                auto *p = data.data() + i * slice_sz;
                thrust::transform(thrust::device, p, p + slice_sz,
                                  kernel_hat_.data(), p,
                                  thrust::multiplies<complex_t>());
            }
        }

      public:
        PointSpreadFunction2D() = default;

        // grid:       the 2D polar grid the PSF is built from.
        // image_size: square in-plane size (nrows == ncols) of the slices
        //             that will later be passed to convolve().
        PointSpreadFunction2D(const gpu::PolarGrid2D<T> &grid, size_t image_size) {
            psf_n_ = next_fast_dim(2 * image_size - 1);
            dims_t psf_dims{1, psf_n_, psf_n_};

            // unit weights at all non-uniform grid points
            auto ones = DeviceArray<complex_t>(grid.dims());
            thrust::fill(ones.begin(), ones.end(), complex_t{T(1), T(0)});

            // one-shot plan (not cached): non-uniform -> oversampled uniform grid
            DeviceArray<complex_t> nufft_out(psf_dims);
            {
                int gpu_id;
                SAFE_CALL(cudaGetDevice(&gpu_id));
                std::array<int64_t, 3> n_modes = {static_cast<int64_t>(psf_n_),
                                                  static_cast<int64_t>(psf_n_), 1};
                nufft::cuFinfftPlanWrapper<T> plan(1, 2, n_modes, 1, gpu_id);
                plan.set_points(grid);
                plan.execute(ones.data(), nufft_out.data());
            }

            // PSF is real for a symmetric grid; extract real part before R2C FFT
            auto kernel_real = gpu::array::to_real(nufft_out);
            kernel_hat_ = gpu::fft::rfft2d(kernel_real);

            // Size the partition once, from the memory free now. Per slice the
            // working set is the padded real slice, its half spectrum, the C2R
            // result and the R2C/C2R cuFFT work areas: ~5 * psf_n_^2 elements.
            // Budget a quarter of free memory; the solver owns the rest.
            size_t free_b = 0, total_b = 0;
            SAFE_CALL(cudaMemGetInfo(&free_b, &total_b));
            const size_t per_slice = 5 * psf_n_ * psf_n_ * sizeof(T);
            max_partition_ = std::max<size_t>(1, (free_b / 4) / per_slice);
        }

        // Override the maximum partition size (slices convolved at once).
        // Mainly for tests that need to exercise the multi-partition path on
        // small inputs, and for a scheduler that sizes partitions itself.
        void set_max_partition(size_t n) { max_partition_ = std::max<size_t>(1, n); }

        // Slices are independent, so the volume is convolved in equal-sized
        // partitions; the last one is zero-padded with empty slices (zeros in,
        // zeros out) so every call uses a single batch size and hence a
        // single pair of cached cuFFT plans.
        DeviceArray<T> convolve(const DeviceArray<T> &input) const {
            const dims_t dims = input.dims();
            const size_t chunk = partition_size(dims.n1);
            if (chunk >= dims.n1) return convolve_chunk(input);

            DeviceArray<T> output(dims);
            const size_t plane = dims.n2 * dims.n3;
            for (size_t s = 0; s < dims.n1; s += chunk) {
                const size_t nb = std::min(chunk, dims.n1 - s);
                DeviceArray<T> slab(dims_t{chunk, dims.n2, dims.n3}); // zeroed
                copyD2D(slab.data(), input.data() + s * plane,
                        nb * plane * sizeof(T));
                auto res = convolve_chunk(slab);
                copyD2D(output.data() + s * plane, res.data(),
                        nb * plane * sizeof(T));
            }
            return output;
        }

      private:
        // Balanced partition size: the fewest partitions of at most
        // max_partition_ slices, spread evenly (e.g. 745 with a cap of 403
        // gives 2 x 373) to minimise zero-padding. Depends only on nslices.
        size_t partition_size(size_t nslices) const {
            const size_t nparts = (nslices + max_partition_ - 1) / max_partition_;
            return (nslices + nparts - 1) / nparts;
        }

        DeviceArray<T> convolve_chunk(const DeviceArray<T> &input) const {
            auto orig_dims = input.dims();
            dims_t pad_dims{orig_dims.n1, psf_n_, psf_n_};

            // zero-pad every slice to the PSF size, one batched 2D R2C FFT,
            // multiply by the shared kernel, one batched 2D C2R FFT. Scoped so
            // `padded` and the spectrum are released as soon as they are used.
            DeviceArray<T> result;
            {
                DeviceArray<complex_t> output_hat;
                {
                    auto padded = gpu::pad3d(input, pad_dims, PadType::RIGHT);
                    output_hat = gpu::fft::rfft2d(padded);
                }
                broadcast_multiply(output_hat);
                result = gpu::fft::irfft2d(output_hat, pad_dims);
            }

            // cuFFT is unnormalized; match sysmat2d's NUFFT scale (nradial)
            T norm = static_cast<T>(pad_dims.n2) * static_cast<T>(pad_dims.n3) *
                     static_cast<T>(orig_dims.n3);

            // FINUFFT type-1 output is in CMCL ordering: the PSF center (k=0)
            // sits at index psf_n_/2 in the padded array.
            dims_t center{0, psf_n_ / 2, psf_n_ / 2};
            return gpu::crop3d(result, orig_dims, center) / norm;
        }
    };

} // namespace tomocam::gpu

#endif // GPU_TOEPLITZ2D_H
