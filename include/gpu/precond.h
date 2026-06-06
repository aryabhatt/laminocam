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

#include <cmath>
#include <type_traits>
#include <vector>

#include <cuda/std/complex>
#include <cufft.h>

#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/utils.h"

namespace tomocam::gpu::opt {

    // -------------------------------------------------------------------------
    // GPU Ramp Preconditioner
    // Mirrors CPU RampPreconditioner (include/precond.h) using cuFFT batched
    // 2D plans.  The filter is the same sqrt(fx^2 + fy^2) ramp in frequency
    // space, applied independently to each slice of the 3D volume.
    // -------------------------------------------------------------------------
    template <typename T>
    class GpuRampPreconditioner {
      private:
        dims_t dims_;
        size_t nc_;             // n3/2 + 1 (half-spectrum column count)
        DeviceArray<T> filter_; // ramp values, shape (1, n2, nc)

      public:
        explicit GpuRampPreconditioner(dims_t dims)
            : dims_(dims), nc_(dims.n3 / 2 + 1),
              filter_(dims_t{1, dims.n2, dims.n3 / 2 + 1}) {

            size_t n2 = dims.n2, n3 = dims.n3;

            // Build ramp filter on host (same formula as CPU RampPreconditioner)
            std::vector<T> xfreq(nc_), yfreq(n2);
            for (size_t i = 0; i < nc_; ++i) xfreq[i] = (T)i / (T)n3;
            for (size_t i = 0; i < n2; ++i) {
                yfreq[i] =
                    (i <= n2 / 2) ? (T)i / (T)n2 : (T)((int)i - (int)n2) / (T)n2;
            }
            std::vector<T> h_filt(n2 * nc_);
            for (size_t j = 0; j < n2; ++j) {
                for (size_t k = 0; k < nc_; ++k) {
                    h_filt[j * nc_ + k] =
                        std::sqrt(xfreq[k] * xfreq[k] + yfreq[j] * yfreq[j]);
                }
            }
            copyH2D(filter_.data(), h_filt.data(), n2 * nc_ * sizeof(T));
        }

        GpuRampPreconditioner(const GpuRampPreconditioner &) = delete;
        GpuRampPreconditioner &operator=(const GpuRampPreconditioner &) = delete;

        DeviceArray<T> apply(const DeviceArray<T> &in) const {

            size_t n1 = dims_.n1, n2 = dims_.n2, n3 = dims_.n3;

            // Fourier transform R2C2D (n2, n3) with batch size n1
            auto in_hat = tomocam::gpu::fft::fft_r2c2d<T>(in);

            // Apply ramp filter in frequency domain (in-place)
            broadcast_multiply_inplace(in_hat, filter_);

            // Inverse Fourier transform C2R2D (n2, n3) with batch size n1
            auto out = tomocam::gpu::fft::ifft_c2r2d<T>(in_hat, in.dims());

            // Scale by 1/(n2*n3) to normalize the round-trip FFT
            array::scale(out, T(1) / static_cast<T>(n2 * n3));
            return out;
        }
    };

} // namespace tomocam::gpu::opt
