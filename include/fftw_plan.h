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

#ifndef FFTW_PLAN_H
#define FFTW_PLAN_H

#include <array>
#include <complex>
#include <cstdint>
#include <stdexcept>

#include <fftw3.h>

namespace tomocam::fft {

    enum class FftwType : uint8_t {
        C2C_FORWARD = 0,
        C2C_BACKWARD = 1,
        R2C = 2,
        C2R = 3
    };

    template <typename T>
    struct FftwTraits;

    template <>
    struct FftwTraits<double> {
        using plan_type = fftw_plan;
        using complex_t = std::complex<double>;

        static void destroy_plan(plan_type p) { fftw_destroy_plan(p); }

        static void make_plan(int dims, int *n, int howmany, int sign,
                              plan_type *plan) {
            size_t total = howmany;
            for (int i = 0; i < dims; ++i) total *= n[i];
            int dist = static_cast<int>(total / howmany);

            fftw_complex *in =
                (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * total);
            fftw_complex *out =
                (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * total);
            *plan = fftw_plan_many_dft(dims, n, howmany, in, nullptr, 1, dist, out,
                                       nullptr, 1, dist, sign, FFTW_MEASURE);
            fftw_free(in);
            fftw_free(out);
            if (*plan == nullptr)
                throw std::runtime_error("Error in fftw_plan_many_dft");
        }

        static void make_plan_c2r(int dims, int *n, int howmany, plan_type *plan) {
            size_t out_size = howmany;
            for (int i = 0; i < dims; ++i) out_size *= n[i];
            size_t in_size = (out_size / n[dims - 1]) * (n[dims - 1] / 2 + 1);

            int idist = static_cast<int>(in_size / howmany);
            int odist = static_cast<int>(out_size / howmany);

            fftw_complex *in =
                (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * in_size);
            double *out = (double *)fftw_malloc(sizeof(double) * out_size);
            *plan = fftw_plan_many_dft_c2r(dims, n, howmany, in, nullptr, 1, idist,
                                           out, nullptr, 1, odist, FFTW_MEASURE);
            fftw_free(in);
            fftw_free(out);
            if (*plan == nullptr)
                throw std::runtime_error("Error in fftw_plan_many_dft_c2r");
        }

        static void make_plan_r2c(int dims, int *n, int howmany, plan_type *plan) {
            size_t in_size = howmany;
            for (int i = 0; i < dims; ++i) in_size *= n[i];
            size_t out_size = (in_size / n[dims - 1]) * (n[dims - 1] / 2 + 1);

            int idist = static_cast<int>(in_size / howmany);
            int odist = static_cast<int>(out_size / howmany);

            double *in = (double *)fftw_malloc(sizeof(double) * in_size);
            fftw_complex *out =
                (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * out_size);
            *plan = fftw_plan_many_dft_r2c(dims, n, howmany, in, nullptr, 1, idist,
                                           out, nullptr, 1, odist, FFTW_MEASURE);
            fftw_free(in);
            fftw_free(out);
            if (*plan == nullptr)
                throw std::runtime_error("Error in fftw_plan_many_dft_r2c");
        }

        static void execute_c2c(plan_type plan, complex_t *in, complex_t *out) {
            fftw_execute_dft(plan, reinterpret_cast<fftw_complex *>(in),
                             reinterpret_cast<fftw_complex *>(out));
        }

        static void execute_r2c(plan_type plan, double *in, complex_t *out) {
            fftw_execute_dft_r2c(plan, in,
                                 reinterpret_cast<fftw_complex *>(out));
        }

        static void execute_c2r(plan_type plan, complex_t *in, double *out) {
            fftw_execute_dft_c2r(plan, reinterpret_cast<fftw_complex *>(in), out);
        }
    };

    template <>
    struct FftwTraits<float> {
        using plan_type = fftwf_plan;
        using complex_t = std::complex<float>;

        static void destroy_plan(plan_type p) { fftwf_destroy_plan(p); }

        static void make_plan(int dims, int *n, int howmany, int sign,
                              plan_type *plan) {
            size_t total = howmany;
            for (int i = 0; i < dims; ++i) total *= n[i];
            int dist = static_cast<int>(total / howmany);

            fftwf_complex *in =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * total);
            fftwf_complex *out =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * total);
            *plan = fftwf_plan_many_dft(dims, n, howmany, in, nullptr, 1, dist, out,
                                        nullptr, 1, dist, sign, FFTW_MEASURE);
            fftwf_free(in);
            fftwf_free(out);
            if (*plan == nullptr)
                throw std::runtime_error("Error in fftwf_plan_many_dft");
        }

        static void make_plan_c2r(int dims, int *n, int howmany, plan_type *plan) {
            size_t out_size = howmany;
            for (int i = 0; i < dims; ++i) out_size *= n[i];
            size_t in_size = (out_size / n[dims - 1]) * (n[dims - 1] / 2 + 1);

            int idist = static_cast<int>(in_size / howmany);
            int odist = static_cast<int>(out_size / howmany);

            fftwf_complex *in =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * in_size);
            float *out = (float *)fftwf_malloc(sizeof(float) * out_size);
            *plan = fftwf_plan_many_dft_c2r(dims, n, howmany, in, nullptr, 1, idist,
                                            out, nullptr, 1, odist, FFTW_MEASURE);
            fftwf_free(in);
            fftwf_free(out);
            if (*plan == nullptr)
                throw std::runtime_error("Error in fftwf_plan_many_dft_c2r");
        }

        static void make_plan_r2c(int dims, int *n, int howmany, plan_type *plan) {
            size_t in_size = howmany;
            for (int i = 0; i < dims; ++i) in_size *= n[i];
            size_t out_size = (in_size / n[dims - 1]) * (n[dims - 1] / 2 + 1);

            int idist = static_cast<int>(in_size / howmany);
            int odist = static_cast<int>(out_size / howmany);

            float *in = (float *)fftwf_malloc(sizeof(float) * in_size);
            fftwf_complex *out =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * out_size);
            *plan = fftwf_plan_many_dft_r2c(dims, n, howmany, in, nullptr, 1, idist,
                                            out, nullptr, 1, odist, FFTW_MEASURE);
            fftwf_free(in);
            fftwf_free(out);
            if (*plan == nullptr)
                throw std::runtime_error("Error in fftwf_plan_many_dft_r2c");
        }

        static void execute_c2c(plan_type plan, complex_t *in, complex_t *out) {
            fftwf_execute_dft(plan, reinterpret_cast<fftwf_complex *>(in),
                              reinterpret_cast<fftwf_complex *>(out));
        }

        static void execute_r2c(plan_type plan, float *in, complex_t *out) {
            fftwf_execute_dft_r2c(plan, in,
                                  reinterpret_cast<fftwf_complex *>(out));
        }

        static void execute_c2r(plan_type plan, complex_t *in, float *out) {
            fftwf_execute_dft_c2r(plan, reinterpret_cast<fftwf_complex *>(in), out);
        }
    };

    template <typename T>
    class FftwPlanWrapper {
      private:
        using Traits = FftwTraits<T>;
        typename Traits::plan_type plan_ = nullptr;
        FftwType type_;

      public:
        FftwPlanWrapper() = default;

        FftwPlanWrapper(int ndim, std::array<int, 3> n, int howmany, FftwType type)
            : type_(type) {
            int *dims = n.data();
            switch (type) {
                case FftwType::C2C_FORWARD:
                    Traits::make_plan(ndim, dims, howmany, FFTW_FORWARD, &plan_);
                    break;
                case FftwType::C2C_BACKWARD:
                    Traits::make_plan(ndim, dims, howmany, FFTW_BACKWARD, &plan_);
                    break;
                case FftwType::R2C:
                    Traits::make_plan_r2c(ndim, dims, howmany, &plan_);
                    break;
                case FftwType::C2R:
                    Traits::make_plan_c2r(ndim, dims, howmany, &plan_);
                    break;
            }
        }

        FftwPlanWrapper(const FftwPlanWrapper &) = delete;
        FftwPlanWrapper &operator=(const FftwPlanWrapper &) = delete;

        FftwPlanWrapper(FftwPlanWrapper &&o) noexcept
            : plan_(o.plan_), type_(o.type_) {
            o.plan_ = nullptr;
        }

        FftwPlanWrapper &operator=(FftwPlanWrapper &&o) noexcept {
            if (this != &o) {
                if (plan_) Traits::destroy_plan(plan_);
                plan_ = o.plan_;
                type_ = o.type_;
                o.plan_ = nullptr;
            }
            return *this;
        }

        ~FftwPlanWrapper() {
            if (plan_) Traits::destroy_plan(plan_);
        }

        bool valid() const { return plan_ != nullptr; }

        void execute(typename Traits::complex_t *in, typename Traits::complex_t *out) {
            if (!plan_)
                throw std::runtime_error(
                    "FftwPlanWrapper::execute: plan not initialized");
            Traits::execute_c2c(plan_, in, out);
        }

        void execute(T *in, typename Traits::complex_t *out) {
            if (!plan_)
                throw std::runtime_error(
                    "FftwPlanWrapper::execute: plan not initialized");
            Traits::execute_r2c(plan_, in, out);
        }

        void execute(typename Traits::complex_t *in, T *out) {
            if (!plan_)
                throw std::runtime_error(
                    "FftwPlanWrapper::execute: plan not initialized");
            Traits::execute_c2r(plan_, in, out);
        }
    };

} // namespace tomocam::fft

#endif // FFTW_PLAN_H
