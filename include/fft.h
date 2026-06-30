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

#ifndef FFTDEFS__H
#define FFTDEFS__H

#include <array>
#include <complex>

#include "array.h"
#include "dtypes.h"
#include "fftw_plan_cache.h"

namespace tomocam::fft {

    template <typename T>
    Array<std::complex<T>> fft1(const Array<std::complex<T>> &input) {
        using complex_t = std::complex<T>;
        dims_t dims = input.dims();
        Array<complex_t> output(dims);

        int howmany = static_cast<int>(dims.x() * dims.y());
        std::array<int, 3> n = {static_cast<int>(dims.z()), 0, 0};
        auto &plan = plans::cache<T>.get_plan(1, n, howmany, FftwType::C2C_FORWARD);
        plan.execute(const_cast<complex_t *>(input.begin()), output.begin());
        return output;
    }

    template <typename T>
    Array<std::complex<T>> ifft1(const Array<std::complex<T>> &input) {
        using complex_t = std::complex<T>;
        dims_t dims = input.dims();
        Array<complex_t> output(dims);

        int howmany = static_cast<int>(dims.x() * dims.y());
        std::array<int, 3> n = {static_cast<int>(dims.z()), 0, 0};
        auto &plan = plans::cache<T>.get_plan(1, n, howmany, FftwType::C2C_BACKWARD);
        plan.execute(const_cast<complex_t *>(input.begin()), output.begin());
        return output;
    }

    template <typename T>
    Array<std::complex<T>> fft2(const Array<std::complex<T>> &input) {
        using complex_t = std::complex<T>;
        dims_t dims = input.dims();
        Array<complex_t> output(dims);

        int howmany = static_cast<int>(dims.x());
        std::array<int, 3> n = {static_cast<int>(dims.y()), static_cast<int>(dims.z()), 0};
        auto &plan = plans::cache<T>.get_plan(2, n, howmany, FftwType::C2C_FORWARD);
        plan.execute(const_cast<complex_t *>(input.begin()), output.begin());
        return output;
    }

    template <typename T>
    Array<std::complex<T>> ifft2(const Array<std::complex<T>> &input) {
        using complex_t = std::complex<T>;
        dims_t dims = input.dims();
        Array<complex_t> output(dims);

        int howmany = static_cast<int>(dims.x());
        std::array<int, 3> n = {static_cast<int>(dims.y()), static_cast<int>(dims.z()), 0};
        auto &plan = plans::cache<T>.get_plan(2, n, howmany, FftwType::C2C_BACKWARD);
        plan.execute(const_cast<complex_t *>(input.begin()), output.begin());
        return output;
    }

    template <typename T>
    Array<std::complex<T>> fft2_r2c(const Array<T> &input) {
        using complex_t = std::complex<T>;
        dims_t dims = input.dims();
        dims_t out_dims = {dims.n1, dims.n2, dims.n3 / 2 + 1};
        Array<complex_t> output(out_dims);

        int howmany = static_cast<int>(dims.n1);
        std::array<int, 3> n = {static_cast<int>(dims.n2), static_cast<int>(dims.n3), 0};
        auto &plan = plans::cache<T>.get_plan(2, n, howmany, FftwType::R2C);
        plan.execute(const_cast<T *>(input.begin()), output.begin());
        return output;
    }

    template <typename T>
    Array<T> fft2_c2r(const Array<std::complex<T>> &input, dims_t out_dims) {
        using complex_t = std::complex<T>;
        Array<T> output(out_dims);

        int howmany = static_cast<int>(out_dims.n1);
        std::array<int, 3> n = {static_cast<int>(out_dims.n2), static_cast<int>(out_dims.n3), 0};
        auto &plan = plans::cache<T>.get_plan(2, n, howmany, FftwType::C2R);
        plan.execute(const_cast<complex_t *>(input.begin()), output.begin());
        return output;
    }

    template <typename T>
    Array<std::complex<T>> fft3(const Array<std::complex<T>> &input) {
        using complex_t = std::complex<T>;
        dims_t dims = input.dims();
        Array<complex_t> output(dims);

        std::array<int, 3> n = {static_cast<int>(dims.x()), static_cast<int>(dims.y()),
                                 static_cast<int>(dims.z())};
        auto &plan = plans::cache<T>.get_plan(3, n, 1, FftwType::C2C_FORWARD);
        plan.execute(const_cast<complex_t *>(input.begin()), output.begin());
        return output;
    }

    template <typename T>
    Array<std::complex<T>> ifft3(const Array<std::complex<T>> &input) {
        using complex_t = std::complex<T>;
        dims_t dims = input.dims();
        Array<complex_t> output(dims);

        std::array<int, 3> n = {static_cast<int>(dims.x()), static_cast<int>(dims.y()),
                                 static_cast<int>(dims.z())};
        auto &plan = plans::cache<T>.get_plan(3, n, 1, FftwType::C2C_BACKWARD);
        plan.execute(const_cast<complex_t *>(input.begin()), output.begin());
        return output;
    }

} // namespace tomocam::fft

#endif // FFTDEFS__H
