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
    class RampPreconditioner {
      private:
        Array<T> filter_;

      public:
        RampPreconditioner(dims_t dims) {

            dims_t filter_dims = {1, dims.n2, dims.n3 / 2 + 1};
            filter_ = Array<T>(filter_dims);

            // Create 1-d ramp filter in x-direction
            int n3 = static_cast<int>(dims.n3);
            std::vector<T> xfreq(dims.n3 / 2 + 1, 0);
            for (int i = 0; i < n3 / 2 + 1; ++i) { xfreq[i] = (T)i / (T)n3; }

            // Create 1-d ramp filter in y-direction
            int n2 = static_cast<int>(dims.n2);
            std::vector<T> yfreq(dims.n2, 0);
            for (int i = 0; i < n2; ++i) {
                yfreq[i] = (i <= n2 / 2) ? (T)i / (T)n2 : (T)(i - n2) / (T)n2;
            }

            // create a 2D ramp filter by outer addition
            for (size_t j = 0; j < dims.n2; ++j) {
                for (size_t i = 0; i < dims.n3 / 2 + 1; ++i) {
                    auto f = std::sqrt(xfreq[i] * xfreq[i] + yfreq[j] * yfreq[j]);
                    filter_[{0, j, i}] = f;
                }
            }
#ifdef DEBUG
            for (auto f : filter_) {
                if (std::isnan(f) || std::isinf(f)) {
                    throw std::runtime_error(
                        "Error: Ramp filter contains NaN/Inf values.\n");
                }
            }
#endif
        }

        Array<T> apply(const Array<T> &input) const {

            // Apply the ramp filter in frequency domain
            auto dims = input.dims();
            auto scale = 1.0 / (dims.n2 * dims.n3);
            auto fft_input = fft::fft2_r2c(input);
            for (size_t i = 0; i < dims.n1; ++i) {
                auto slice = fft_input.slice(i);
                std::transform(std::execution::par_unseq, slice.begin(), slice.end(),
                               filter_.begin(), slice.begin(),
                               std::multiplies<std::complex<T>>());
            }
            auto filtered = fft::fft2_c2r(fft_input, dims);
            return filtered * scale;
        }
    };
} // namespace tomocam::opt

#endif // TOMOCAM_PRECOND__H
