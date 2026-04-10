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
#include <fftw3.h>
#include <new>
#include <type_traits>

#include "array.h"
#include "dtypes.h"

#ifndef FFTDEFS__H
    #define FFTDEFS__H

namespace tomocam::fft {

    template <typename T>
    Array<std::complex<T>> fft1(const Array<std::complex<T>> &input) {

        // double or float
        bool is_double = std::is_same_v<T, double>;

        // create return array
        dims_t dims = input.dims();
        Array<std::complex<T>> output(dims);

        int rank = 1;
        int n[] = {static_cast<int>(dims.z())};
        int idist = static_cast<int>(dims.z());
        int odist = static_cast<int>(dims.z());
        int istride = 1;
        int ostride = 1;
        int *inembed = NULL;
        int *onembed = NULL;
        int batches = static_cast<int>(dims.x() * dims.y());

        if (is_double) {
            auto *idata = (fftw_complex *)input.begin();
            auto *odata = (fftw_complex *)output.begin();
            fftw_plan plan = fftw_plan_many_dft(
                rank, n, batches, idata, inembed, istride, idist, odata, onembed,
                ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
            if (!plan) { throw std::bad_alloc(); }
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            auto *idata = (fftwf_complex *)input.begin();
            auto *odata = (fftwf_complex *)output.begin();
            fftwf_plan plan = fftwf_plan_many_dft(
                rank, n, batches, idata, inembed, istride, idist, odata, onembed,
                ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        return output;
    }

    template <typename T>
    Array<std::complex<T>> ifft1(const Array<std::complex<T>> &arr) {

        // double or float
        bool is_double = std::is_same_v<T, double>;

        // create return array
        dims_t dims = arr.dims();
        Array<std::complex<T>> output(arr.dims());

        int rank = 1;
        int n[] = {static_cast<int>(dims.z())};
        int idist = static_cast<int>(dims.z());
        int odist = static_cast<int>(dims.z());
        int istride = 1;
        int ostride = 1;
        int *inembed = NULL;
        int *onembed = NULL;
        int batches = static_cast<int>(dims.x() * dims.y());

        if (is_double) {
            auto *idata = (fftw_complex *)arr.begin();
            auto *odata = (fftw_complex *)output.begin();
            fftw_plan plan = fftw_plan_many_dft(
                rank, n, batches, idata, inembed, istride, idist, odata, onembed,
                ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            auto *idata = (fftwf_complex *)arr.begin();
            auto *odata = (fftwf_complex *)output.begin();
            fftwf_plan plan = fftwf_plan_many_dft(
                rank, n, batches, idata, inembed, istride, idist, odata, onembed,
                ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        return output;
    }

    template <typename T>
    Array<std::complex<T>> fft2(const Array<std::complex<T>> &input) {

        // double or float
        bool is_double = std::is_same_v<T, double>;

        // create return array
        dims_t dims = input.dims();
        Array<std::complex<T>> output(input.dims());

        int rank = 2;
        int n[] = {static_cast<int>(dims.y()), static_cast<int>(dims.z())};
        int idist = static_cast<int>(dims.y() * dims.z());
        int odist = static_cast<int>(dims.y() * dims.z());
        int istride = 1;
        int ostride = 1;
        int *inembed = NULL;
        int *onembed = NULL;
        int batches = static_cast<int>(dims.x());

        if (is_double) {
            auto *idata = (fftw_complex *)input.begin();
            auto *odata = (fftw_complex *)output.begin();
            fftw_plan plan = fftw_plan_many_dft(
                rank, n, batches, idata, inembed, istride, idist, odata, onembed,
                ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
            if (!plan) { throw std::bad_alloc(); }
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            auto *idata = (fftwf_complex *)input.begin();
            auto *odata = (fftwf_complex *)output.begin();
            fftwf_plan plan = fftwf_plan_many_dft(
                rank, n, batches, idata, inembed, istride, idist, odata, onembed,
                ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        return output;
    }

    template <typename T>
    Array<std::complex<T>> ifft2(const Array<std::complex<T>> &input) {
        // double or float
        bool is_double = std::is_same_v<T, double>;

        // create return array
        dims_t dims = input.dims();
        Array<std::complex<T>> output(input.dims());

        int rank = 2;
        int n[] = {static_cast<int>(dims.y()), static_cast<int>(dims.z())};
        int idist = static_cast<int>(dims.y() * dims.z());
        int odist = static_cast<int>(dims.y() * dims.z());
        int istride = 1;
        int ostride = 1;
        int *inembed = NULL;
        int *onembed = NULL;
        int batches = static_cast<int>(dims.x());

        if (is_double) {
            auto *idata = (fftw_complex *)input.begin();
            auto *odata = (fftw_complex *)output.begin();
            fftw_plan plan = fftw_plan_many_dft(
                rank, n, batches, idata, inembed, istride, idist, odata, onembed,
                ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            auto *idata = (fftwf_complex *)input.begin();
            auto *odata = (fftwf_complex *)output.begin();
            fftwf_plan plan = fftwf_plan_many_dft(
                rank, n, batches, idata, inembed, istride, idist, odata, onembed,
                ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        return output;
    }

    template <typename T>
    Array<std::complex<T>> fft2_r2c(const Array<T> &input) {

        // double or float
        bool is_double = std::is_same_v<T, double>;

        dims_t dims = input.dims();
        // create return array
        dims_t out_dims = {dims.n1, dims.n2, dims.n3 / 2 + 1};

        Array<std::complex<T>> output(out_dims);

        int rank = 2;
        int n[] = {static_cast<int>(dims.n2), static_cast<int>(dims.n3)};
        int idist = static_cast<int>(dims.n2 * dims.n3);
        int odist = static_cast<int>(out_dims.n2 * out_dims.n3);
        int istride = 1;
        int ostride = 1;
        int *inembed = nullptr;
        int *onembed = nullptr;
        int batches = static_cast<int>(dims.n1);

        if (is_double) {
            auto *idata = (double *)input.begin();
            auto *odata = (fftw_complex *)output.begin();
            fftw_plan plan = fftw_plan_many_dft_r2c(rank, n, batches, idata, inembed,
                                                    istride, idist, odata, onembed,
                                                    ostride, odist, FFTW_ESTIMATE);
            if (!plan) { throw std::bad_alloc(); }
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            auto *idata = (float *)input.begin();
            auto *odata = (fftwf_complex *)output.begin();
            fftwf_plan plan = fftwf_plan_many_dft_r2c(
                rank, n, batches, idata, inembed, istride, idist, odata, onembed,
                ostride, odist, FFTW_ESTIMATE);
            if (!plan) { throw std::bad_alloc(); }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        return output;
    }

    template <typename T>
    Array<T> fft2_c2r(const Array<std::complex<T>> &input, dims_t out_dims) {

        // double or float
        bool is_double = std::is_same_v<T, double>;

        dims_t dims = input.dims();
        // create return array
        Array<T> output(out_dims);

        int rank = 2;
        int n[] = {static_cast<int>(out_dims.n2), static_cast<int>(out_dims.n3)};
        int idist = static_cast<int>(dims.n2 * dims.n3);
        int odist = static_cast<int>(out_dims.n2 * out_dims.n3);
        int istride = 1;
        int ostride = 1;
        int *inembed = nullptr;
        int *onembed = nullptr;
        int batches = static_cast<int>(dims.n1);

        if (is_double) {
            auto *idata = (fftw_complex *)input.begin();
            auto *odata = (double *)output.begin();
            fftw_plan plan = fftw_plan_many_dft_c2r(rank, n, batches, idata, inembed,
                                                    istride, idist, odata, onembed,
                                                    ostride, odist, FFTW_ESTIMATE);
            if (!plan) { throw std::bad_alloc(); }
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            auto *idata = (fftwf_complex *)input.begin();
            auto *odata = (float *)output.begin();
            fftwf_plan plan = fftwf_plan_many_dft_c2r(
                rank, n, batches, idata, inembed, istride, idist, odata, onembed,
                ostride, odist, FFTW_ESTIMATE);
            if (!plan) { throw std::bad_alloc(); }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        return output;
    }

    template <typename T>
    Array<std::complex<T>> fft3(const Array<std::complex<T>> &input) {

        // double or float
        bool is_double = std::is_same_v<T, double>;

        // create return array
        dims_t dims = input.dims();
        Array<std::complex<T>> output(input.dims());

        if (is_double) {
            auto *idata = (fftw_complex *)input.begin();
            auto *odata = (fftw_complex *)output.begin();
            fftw_plan plan = fftw_plan_dft_3d(dims.x(), dims.y(), dims.z(), idata,
                                              odata, -1, FFTW_ESTIMATE);
            if (!plan) { throw std::bad_alloc(); }
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            auto *idata = (fftwf_complex *)input.begin();
            auto *odata = (fftwf_complex *)output.begin();
            fftwf_plan plan = fftwf_plan_dft_3d(dims.x(), dims.y(), dims.z(), idata,
                                                odata, -1, FFTW_ESTIMATE);
            if (!plan) { throw std::bad_alloc(); }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
        return output;
    }

    template <typename T>
    Array<std::complex<T>> ifft3(const Array<std::complex<T>> &input) {

        // double or float
        bool is_double = std::is_same_v<T, double>;

        dims_t dims = input.dims();
        Array<std::complex<T>> output(dims);

        if (is_double) {
            auto *idata = (fftw_complex *)input.begin();
            auto *odata = (fftw_complex *)output.begin();
            fftw_plan plan = fftw_plan_dft_3d(dims.x(), dims.y(), dims.z(), idata,
                                              odata, 1, FFTW_ESTIMATE);
            if (!plan) { throw std::bad_alloc(); }
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        } else {
            auto *idata = (fftwf_complex *)input.begin();
            auto *odata = (fftwf_complex *)output.begin();
            fftwf_plan plan = fftwf_plan_dft_3d(dims.x(), dims.y(), dims.z(), idata,
                                                odata, 1, FFTW_ESTIMATE);
            if (!plan) { throw std::bad_alloc(); }
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }

        return output;
    }

} // namespace tomocam::fft

#endif // FFTDEFS__H
