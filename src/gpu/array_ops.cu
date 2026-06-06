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

#include <cuda/std/complex>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"

namespace tomocam::gpu {

    template <typename T>
    concept Real_t = std::is_same_v<T, float> || std::is_same_v<T, double>;

    // -------------------------------------------------------------------------
    // Internal functors
    // -------------------------------------------------------------------------

    namespace array {
        namespace detail {

            template <typename T>
            struct ToComplexFn {
                __host__ __device__ cuda::std::complex<T> operator()(T x) const {
                    return {x, T(0)};
                }
            };

            template <typename T>
            struct ToRealFn {
                __host__ __device__ T operator()(cuda::std::complex<T> x) const {
                    return x.real();
                }
            };

            template <typename T>
            struct AbsFn {
                __host__ __device__ T operator()(T x) const {
                    return cuda::std::abs(x);
                }
            };

            // Magnitude of complex -> real
            template <typename T>
            struct ComplexAbsFn {
                __host__ __device__ T operator()(cuda::std::complex<T> x) const {
                    return cuda::std::abs(x);
                }
            };

            template <typename T>
            struct SqFn {
                __host__ __device__ T operator()(T x) const { return x * x; }
            };

            // |x|^2 for complex, avoids sqrt inside transform_reduce
            template <typename T>
            struct NormSqFn {
                __host__ __device__ T operator()(cuda::std::complex<T> x) const {
                    return x.real() * x.real() + x.imag() * x.imag();
                }
            };

            // x = x + alpha * y  =>  f(y, x) = x + alpha * y
            template <typename T>
            struct AxpyFn {
                T alpha;
                explicit AxpyFn(T a) : alpha(a) {}
                __host__ __device__ T operator()(T y, T x) const {
                    return x * alpha + y;
                }
            };

            // x = x + y * alpha  =>  f(y, x) = x + y * beta
            template <typename T>
            struct XpayFn {
                T beta;
                explicit XpayFn(T b) : beta(b) {}
                __host__ __device__ T operator()(T y, T x) const {
                    return x + beta * y;
                }
            };

            template <typename T>
            struct SqrtFn {
                __host__ __device__ T operator()(T x) const {
                    return cuda::std::sqrt(x);
                }
            };

            // clamp near-zero values: if |v| < threshold, return replacement
            template <typename T>
            struct ClampSmallFn {
                T threshold, replacement;
                explicit ClampSmallFn(T t, T r) : threshold(t), replacement(r) {}
                __host__ __device__ T operator()(T v) const {
                    return (cuda::std::abs(v) < threshold) ? replacement : v;
                }
            };

            // element-wise reciprocal: 1 / v
            template <typename T>
            struct ReciprocalFn {
                __host__ __device__ T operator()(T v) const { return T(1) / v; }
            };

        } // namespace detail

        // x = x * alpha + y
        template <typename T>
        void axpy(DeviceArray<T> &x, T alpha, const DeviceArray<T> &y) {
            thrust::transform(y.begin(), y.end(), x.begin(), x.begin(),
                              detail::AxpyFn<T>(alpha));
        }

        // x += y * beta
        template <typename T>
        void xpay(DeviceArray<T> &x, const DeviceArray<T> &y, T beta) {
            thrust::transform(y.begin(), y.end(), x.begin(), x.begin(),
                              detail::XpayFn<T>(beta));
        }

        // to_complex: DeviceArray<T> -> DeviceArray<complex<T>>
        template <Real_t T>
        DeviceArray<cuda::std::complex<T>> to_complex(const DeviceArray<T> &a) {
            DeviceArray<cuda::std::complex<T>> b(a.dims());
            thrust::transform(a.begin(), a.end(), b.begin(),
                              detail::ToComplexFn<T>{});
            return b;
        }

        // to_real: DeviceArray<complex<T>> -> DeviceArray<T> (take real part)
        template <Real_t T>
        DeviceArray<T> to_real(const DeviceArray<cuda::std::complex<T>> &a) {
            DeviceArray<T> b(a.dims());
            thrust::transform(a.begin(), a.end(), b.begin(), detail::ToRealFn<T>{});
            return b;
        }

        // element-wise absolute value: out[i] = abs(a[i])
        template <Real_t T>
        DeviceArray<T> abs(const DeviceArray<T> &a) {
            DeviceArray<T> b(a.dims());
            thrust::transform(a.begin(), a.end(), b.begin(), detail::AbsFn<T>{});
            return b;
        }

        // max / min: real arrays
        template <Real_t T>
        T max(const DeviceArray<T> &a) {
            return *thrust::max_element(a.begin(), a.end());
        }
        template <Real_t T>
        T min(const DeviceArray<T> &a) {
            return *thrust::min_element(a.begin(), a.end());
        }

        // L2 norm: sqrt(sum of squares)
        template <Real_t T>
        T norm2(const DeviceArray<T> &a) {
            T sq = thrust::transform_reduce(a.begin(), a.end(), detail::SqFn<T>{},
                                            T(0), thrust::plus<T>{});
            return std::sqrt(sq);
        }

        // norm1: L1 norm  (sum of absolute values)
        template <Real_t T>
        T norm1(const DeviceArray<T> &a) {
            return thrust::transform_reduce(a.begin(), a.end(), detail::AbsFn<T>{},
                                            T(0), thrust::plus<T>{});
        }

        // dot: inner product of two real arrays
        template <Real_t T>
        T dot(const DeviceArray<T> &a, const DeviceArray<T> &b) {
            return thrust::inner_product(a.begin(), a.end(), b.begin(), T(0));
        }

        // sqrt of each element: out = sqrt(a)
        template <typename T>
        DeviceArray<T> sqrt(const DeviceArray<T> &a) {
            DeviceArray<T> out(a.dims());
            thrust::transform(a.begin(), a.end(), out.begin(), detail::SqrtFn<T>{});
            return out;
        }

    } // namespace array
} // namespace tomocam::gpu

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

namespace tomocam::gpu::array {

    using cf = cuda::std::complex<float>;
    using cd = cuda::std::complex<double>;

    // Generic T functions — instantiate for float, double, and complex variants
#define INST_GENERIC(T)                                                             \
    template void axpy(DeviceArray<T> &, T, const DeviceArray<T> &);                \
    template void xpay(DeviceArray<T> &, const DeviceArray<T> &, T);                \
    template DeviceArray<T> sqrt(const DeviceArray<T> &);

    INST_GENERIC(float)
    INST_GENERIC(double)
    INST_GENERIC(cf)
    INST_GENERIC(cd)
#undef INST_GENERIC

    // Real_t-constrained functions — float and double only
#define INST_REAL(T)                                                                \
    template DeviceArray<cuda::std::complex<T>> to_complex(const DeviceArray<T> &); \
    template DeviceArray<T> to_real(const DeviceArray<cuda::std::complex<T>> &);    \
    template DeviceArray<T> abs(const DeviceArray<T> &);                            \
    template T max(const DeviceArray<T> &);                                         \
    template T min(const DeviceArray<T> &);                                         \
    template T norm2(const DeviceArray<T> &);                                       \
    template T norm1(const DeviceArray<T> &);                                       \
    template T dot(const DeviceArray<T> &, const DeviceArray<T> &);

    INST_REAL(float)
    INST_REAL(double)
#undef INST_REAL

} // namespace tomocam::gpu::array
