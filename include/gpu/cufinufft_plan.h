

#ifndef CUFINUFFT_PLAN_H
#define CUFINUFFT_PLAN_H

#include <array>
#include <cuda/std/complex>
#include <stdexcept>

#include <cufinufft.h>
#include <thrust/device_vector.h>

#include "array.h"
#include "polar_grid.h"
#include "gpu/polar_grid.h"

namespace tomocam::gpu::nufft {
    template <typename T>
    struct cuFinfftTraits;

    template <>
    struct cuFinfftTraits<double> {
        using plan_type = cufinufft_plan;
        using complex_type = cuda::std::complex<double>;

        // make plan with double precision, with a default tolerance of 1e-14
        static int makeplan(int type, int dim, int64_t *n_modes, int iflag,
                            int ntrans, plan_type *plan, cufinufft_opts *opts) {
            constexpr double TOL = 1e-14;
            return cufinufft_makeplan(type, dim, n_modes, iflag, ntrans, TOL, plan,
                                     opts);
        }
        // set points for double precision
        static int setpts(plan_type plan, int64_t npts, double *x, double *y,
                          double *z, int nk, double *s, double *t, double *u) {
            return cufinufft_setpts(plan, npts, x, y, z, nk, s, t, u);
        }
        // execute the plan for double precision
        static int execute(plan_type plan, complex_type *cz, complex_type *fz) {
            return cufinufft_execute(plan, reinterpret_cast<cuDoubleComplex*>(cz), 
                                    reinterpret_cast<cuDoubleComplex*>(fz));
        }
        // destroy the plan for double precision
        static void destroy(plan_type plan) { cufinufft_destroy(plan); }
    };

    template <>
    struct cuFinfftTraits<float> {
        using plan_type = cufinufftf_plan;
        using complex_type = cuda::std::complex<float>;

        // make plan with single precision, with a default tolerance of 1.2e-6
        static int makeplan(int type, int dim, int64_t *n_modes, int iflag,
                            int ntrans, plan_type *plan, cufinufft_opts *opts) {
            constexpr float TOL = 1.2e-06f;
            return cufinufftf_makeplan(type, dim, n_modes, iflag, ntrans, TOL, plan,
                                      opts);
        }
        // set points for single precision
        static int setpts(plan_type plan, int64_t npts, float *x, float *y, float *z,
                          int nk, float *s, float *t, float *u) {
            return cufinufftf_setpts(plan, npts, x, y, z, nk, s, t, u);
        }
        // execute the plan for single precision
        static int execute(plan_type plan, complex_type *cz, complex_type *fz) {
            return cufinufftf_execute(plan, reinterpret_cast<cuFloatComplex*>(cz), 
                                     reinterpret_cast<cuFloatComplex*>(fz));
        }
        // destroy the plan for single precision
        static void destroy(plan_type plan) { cufinufftf_destroy(plan); }
    };

    template <typename T>
    class cuFinfftPlanWrapper {
      private:
        using Traits = cuFinfftTraits<T>;
        typename Traits::plan_type plan;
        bool initialized = false;

      public:
        cuFinfftPlanWrapper() = default;

        cuFinfftPlanWrapper(int type, int dim, std::array<int64_t, 3> n_modes,
                            int iflag, int gpu_id) {

            if (dim != n_modes.size()) {
                throw std::runtime_error("cuFinfftPlanWrapper constructor: dim does "
                                         "not match size of n_modes");
            }
            cufinufft_opts opts;
            cufinufft_default_opts(&opts);
            opts.upsampfac = 2.0;
            opts.gpu_device_id = gpu_id;
            int ierr =
                Traits::makeplan(type, dim, n_modes.data(), iflag, 1, &plan, &opts);
            if (ierr != 0) {
                throw std::runtime_error("Error in cufinufft_makeplan");
            }
            initialized = true;
        }

        // Overload for GPU-resident PolarGrid (coordinates already on device)
        void set_points(const tomocam::gpu::PolarGrid<T> &pg) {
            if (!initialized) {
                throw std::runtime_error(
                    "cuFinfftPlanWrapper::set_points called before make_plan");
            }
            T *x = (T *)pg.x.begin();
            T *y = (T *)pg.y.begin();
            T *z = (T *)pg.z.begin();
            int ierr =
                Traits::setpts(plan, pg.npts, x, y, z, 0, nullptr, nullptr, nullptr);
            if (ierr != 0) { throw std::runtime_error("Error in cufinufft_setpts"); }
        }

        // Overload for CPU-resident PolarGrid (uploads coordinates to device)
        void set_points(const tomocam::PolarGrid<T> &pg) {
            if (!initialized) {
                throw std::runtime_error(
                    "cuFinfftPlanWrapper::set_points called before make_plan");
            }
            thrust::device_vector<T> d_x(pg.x.begin(), pg.x.end());
            thrust::device_vector<T> d_y(pg.y.begin(), pg.y.end());
            thrust::device_vector<T> d_z(pg.z.begin(), pg.z.end());
            T *x = thrust::raw_pointer_cast(d_x.data());
            T *y = thrust::raw_pointer_cast(d_y.data());
            T *z = thrust::raw_pointer_cast(d_z.data());
            int ierr =
                Traits::setpts(plan, pg.npts, x, y, z, 0, nullptr, nullptr, nullptr);
            if (ierr != 0) { throw std::runtime_error("Error in cufinufft_setpts"); }
        }

        void execute(cuda::std::complex<T> *cz, cuda::std::complex<T> *fz) {
            if (!initialized) {
                throw std::runtime_error(
                    "cuFinfftPlanWrapper::execute: plan not initialized");
            }
            int ierr = Traits::execute(plan, cz, fz);
            if (ierr != 0) { throw std::runtime_error("Error in cufinufft_execute"); }
        }

        ~cuFinfftPlanWrapper() {
            if (initialized) { Traits::destroy(plan); }
        }

        cuFinfftPlanWrapper(const cuFinfftPlanWrapper &) = delete;
        cuFinfftPlanWrapper &operator=(const cuFinfftPlanWrapper &) = delete;

        bool valid() const { return initialized; }

        // move constructor
        cuFinfftPlanWrapper(cuFinfftPlanWrapper &&other) noexcept
            : plan(other.plan), initialized(other.initialized) {
            other.initialized = false;
        }

        // move assignment operator
        cuFinfftPlanWrapper &operator=(cuFinfftPlanWrapper &&other) noexcept {
            if (this != &other) {
                if (initialized) { Traits::destroy(plan); }
                plan = other.plan;
                initialized = other.initialized;
                other.initialized = false;
            }
            return *this;
        }
    };
} // namespace tomocam::gpu::nufft

#endif // CUFINUFFT_PLAN_H
