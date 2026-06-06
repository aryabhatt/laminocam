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

#ifndef CUFFT_PLAN_H
#define CUFFT_PLAN_H

#include <cufft.h>
#include <cuda/std/complex>

namespace tomocam::gpu::fft {

    template <typename T>
    struct cuFFTPlanTraits;

    // float specialization
    template <>
    struct cuFFTPlanTraits<float> {
        using plan_type = cufftHandle;
        using complex_type = cufftComplex;

        static int make_plan(int dim, int *n, int batch, plan_type *plan,
                             cufftType type) {
            return cufftPlanMany(plan, dim, n, nullptr, 1, 0, nullptr, 1, 0, type,
                                 batch);
        }

        static int execute(plan_type plan, complex_type *in, complex_type *out,
                           int direction) {
            return cufftExecC2C(plan, in, out, direction);
        }

        static int execute(plan_type plan, complex_type *in, float *out) {
            return cufftExecC2R(plan, in, out);
        }

        static int execute(plan_type plan, float *in, complex_type *out) {
            return cufftExecR2C(plan, in, out);
        }

        static void destroy(plan_type plan) { cufftDestroy(plan); }
    };

    // double specialization
    template <>
    struct cuFFTPlanTraits<double> {
        using plan_type = cufftHandle;
        using complex_type = cufftDoubleComplex;

        static int make_plan(int dim, int *n, int batch, plan_type *plan,
                             cufftType type) {
            return cufftPlanMany(plan, dim, n, nullptr, 1, 0, nullptr, 1, 0, type,
                                 batch);
        }

        static int execute(plan_type plan, complex_type *in, complex_type *out,
                           int direction) {
            return cufftExecZ2Z(plan, in, out, direction);
        }

        static int execute(plan_type plan, complex_type *in, double *out) {
            return cufftExecZ2D(plan, in, out);
        }

        static int execute(plan_type plan, double *in, complex_type *out) {
            return cufftExecD2Z(plan, in, out);
        }

        static void destroy(plan_type plan) { cufftDestroy(plan); }
    };

    template <typename T>
    class cuFFTPlanWrapper {
      private:
        using Traits = cuFFTPlanTraits<T>;
        typename Traits::plan_type plan;
        cufftType fft_type;
        int device_id;

      public:
        cuFFTPlanWrapper(int ndim, int *n, cufftType type, int gpu_id)
            : device_id(gpu_id), fft_type(type) {
            SAFE_CALL(cudaSetDevice(device_id));
            int batch = n[0];
            int dims[] = {n[1], n[2]};
            int ierr = Traits::make_plan(ndim, dims, batch, &plan, fft_type);
            if (ierr != 0) { throw std::runtime_error("Error in cufftPlanMany"); }
        }

        void execute(typename Traits::complex_type *in,
                     typename Traits::complex_type *out, int direction) {
            SAFE_CALL(cudaSetDevice(device_id));
            int ierr = Traits::execute(plan, in, out, direction);
            if (ierr != 0) { throw std::runtime_error("Error in cufftExec"); }
        }

        // Overloads for cuda::std::complex (binary-compatible with cufft complex types)
        void execute(cuda::std::complex<T> *in, cuda::std::complex<T> *out,
                     int direction) {
            execute(reinterpret_cast<typename Traits::complex_type *>(in),
                    reinterpret_cast<typename Traits::complex_type *>(out), direction);
        }

        void execute(T *in, typename Traits::complex_type *out) {
            SAFE_CALL(cudaSetDevice(device_id));
            int ierr = Traits::execute(plan, in, out);
            if (ierr != 0) { throw std::runtime_error("Error in cufftExec"); }
        }

        void execute(T *in, cuda::std::complex<T> *out) {
            execute(in, reinterpret_cast<typename Traits::complex_type *>(out));
        }

        void execute(typename Traits::complex_type *in, T *out) {
            SAFE_CALL(cudaSetDevice(device_id));
            int ierr = Traits::execute(plan, in, out);
            if (ierr != 0) { throw std::runtime_error("Error in cufftExec"); }
        }

        void execute(cuda::std::complex<T> *in, T *out) {
            execute(reinterpret_cast<typename Traits::complex_type *>(in), out);
        }

        ~cuFFTPlanWrapper() {
            if (cudaSetDevice(device_id) == cudaSuccess) {
                if (plan != 0) { Traits::destroy(plan); }
            }
        }

        cuFFTPlanWrapper(const cuFFTPlanWrapper &) = delete;
        cuFFTPlanWrapper &operator=(const cuFFTPlanWrapper &) = delete;

        // move constructor
        cuFFTPlanWrapper(cuFFTPlanWrapper &&other) noexcept
            : plan(other.plan), fft_type(other.fft_type),
              device_id(other.device_id) {
            other.plan = 0; // invalidate the moved-from object
        }
        // move assignment operator
        cuFFTPlanWrapper &operator=(cuFFTPlanWrapper &&other) noexcept {
            if (this != &other) {
                SAFE_CALL(cudaSetDevice(device_id));
                if (plan != 0) { Traits::destroy(plan); } // destroy current plan
                plan = other.plan;
                fft_type = other.fft_type;
                device_id = other.device_id;
                other.plan = 0; // invalidate the moved-from object
            }
            return *this;
        }
    };

} // namespace tomocam::gpu::fft
#endif // CUFFT_PLAN_H
