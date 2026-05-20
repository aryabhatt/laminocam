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

#ifndef GPU_DEVICE_ARRAY_OPS_H
#define GPU_DEVICE_ARRAY_OPS_H

#include <cmath>
#include <type_traits>

#include <cuda/std/complex>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include "gpu/device_array.h"

namespace tomocam::gpu::array {

    template <typename T>
    concept Real_t = std::is_same_v<T, float> || std::is_same_v<T, double>;

    // -------------------------------------------------------------------------
    // Internal functors
    // -------------------------------------------------------------------------

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
            __host__ __device__ T operator()(T x) const { return cuda::std::abs(x); }
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

        // x += alpha * y  =>  f(yi, xi) = xi + alpha * yi
        template <typename T>
        struct AxpyFn {
            T alpha;
            explicit AxpyFn(T a) : alpha(a) {}
            __host__ __device__ T operator()(T yi, T xi) const {
                return xi + alpha * yi;
            }
        };

        // p = z + beta * p  =>  f(zi, pi) = zi + beta * pi
        template <typename T>
        struct XpayFn {
            T beta;
            explicit XpayFn(T b) : beta(b) {}
            __host__ __device__ T operator()(T zi, T pi) const {
                return zi + beta * pi;
            }
        };

        // x *= s
        template <typename T>
        struct ScaleFn {
            T s;
            explicit ScaleFn(T scale) : s(scale) {}
            __host__ __device__ T operator()(T xi) const { return xi * s; }
        };

        template <typename T>
        struct SqrtFn {
            __host__ __device__ T operator()(T x) const {
                return cuda::std::sqrt(x);
            }
        };

    } // namespace detail

    // scale  x *= a
    template <typename T>
    void scale_inplace(DeviceArray<T> &x, T a) {
        thrust::device_ptr<T> px(x.begin());
        thrust::transform(px, px + x.size(), px, detail::ScaleFn<T>(a));
    }

    // scale and return new array: out = a * x
    template <typename T>
    DeviceArray<T> scale(const DeviceArray<T> &x, T a) {
        DeviceArray<T> out(x.dims());
        thrust::device_ptr<const T> px(x.begin());
        thrust::device_ptr<T> pout(out.begin());
        thrust::transform(px, px + x.size(), pout, detail::ScaleFn<T>(a));
        return out;
    }

    // x += alpha * y  (in-place)
    template <typename T>
    void axpy(DeviceArray<T> &x, T alpha, const DeviceArray<T> &y) {
        thrust::device_ptr<T> px(x.begin());
        thrust::device_ptr<const T> py(y.begin());
        thrust::transform(py, py + y.size(), px, px, detail::AxpyFn<T>(alpha));
    }

    // x = y + beta * x  (in-place)
    template <typename T>
    void xpay(DeviceArray<T> &x, const DeviceArray<T> &y, T beta) {
        thrust::device_ptr<T> px(x.begin());
        thrust::device_ptr<const T> py(y.begin());
        thrust::transform(py, py + x.size(), px, px, detail::XpayFn<T>(beta));
    }

    // out = a - b
    template <typename T>
    DeviceArray<T> subtract(const DeviceArray<T> &a, const DeviceArray<T> &b) {
        DeviceArray<T> out(a.dims());
        thrust::device_ptr<const T> pa(a.begin());
        thrust::device_ptr<const T> pb(b.begin());
        thrust::device_ptr<T> pout(out.begin());
        thrust::transform(pa, pa + a.size(), pb, pout, thrust::minus<T>{});
        return out;
    }

    // a -= b  (in-place)
    template <typename T>
    void subtract_inplace(DeviceArray<T> &a, const DeviceArray<T> &b) {
        thrust::device_ptr<T> pa(a.begin());
        thrust::device_ptr<const T> pb(b.begin());
        thrust::transform(pa, pa + a.size(), pb, pa, thrust::minus<T>{});
    }

    // out = a + b
    template <typename T>
    DeviceArray<T> add(const DeviceArray<T> &a, const DeviceArray<T> &b) {
        DeviceArray<T> out(a.dims());
        thrust::device_ptr<const T> pa(a.begin());
        thrust::device_ptr<const T> pb(b.begin());
        thrust::device_ptr<T> pout(out.begin());
        thrust::transform(pa, pa + a.size(), pb, pout, thrust::plus<T>{});
        return out;
    }

    // a += b  (in-place)
    template <typename T>
    void add_inplace(DeviceArray<T> &a, const DeviceArray<T> &b) {
        thrust::device_ptr<T> pa(a.begin());
        thrust::device_ptr<const T> pb(b.begin());
        thrust::transform(pa, pa + a.size(), pb, pa, thrust::plus<T>{});
    }

    // multiply two DeviceArrays element-wise: out = a * b
    template <typename T>
    DeviceArray<T> multiply(const DeviceArray<T> &a, const DeviceArray<T> &b) {
        DeviceArray<T> out(a.dims());
        thrust::device_ptr<const T> pa(a.begin());
        thrust::device_ptr<const T> pb(b.begin());
        thrust::device_ptr<T> pout(out.begin());
        thrust::transform(pa, pa + a.size(), pb, pout, thrust::multiplies<T>{});
        return out;
    }

    // a *= b  (in-place element-wise multiplication)
    template <typename T>
    void multiply_inplace(DeviceArray<T> &a, const DeviceArray<T> &b) {
        thrust::device_ptr<T> pa(a.begin());
        thrust::device_ptr<const T> pb(b.begin());
        thrust::transform(pa, pa + a.size(), pb, pa, thrust::multiplies<T>{});
    }

    // to_complex: DeviceArray<T> -> DeviceArray<complex<T>>
    template <Real_t T>
    DeviceArray<cuda::std::complex<T>> to_complex(const DeviceArray<T> &a) {
        DeviceArray<cuda::std::complex<T>> b(a.dims());
        thrust::device_ptr<const T> in(a.begin());
        thrust::device_ptr<cuda::std::complex<T>> out(b.begin());
        thrust::transform(in, in + a.size(), out, detail::ToComplexFn<T>{});
        return b;
    }

    // to_real: DeviceArray<complex<T>> -> DeviceArray<T> (take real part)
    template <Real_t T>
    DeviceArray<T> to_real(const DeviceArray<cuda::std::complex<T>> &a) {
        DeviceArray<T> b(a.dims());
        thrust::device_ptr<const cuda::std::complex<T>> in(a.begin());
        thrust::device_ptr<T> out(b.begin());
        thrust::transform(in, in + a.size(), out, detail::ToRealFn<T>{});
        return b;
    }

    // element-wise absolute value: out[i] = abs(a[i])
    template <Real_t T>
    DeviceArray<T> abs(const DeviceArray<T> &a) {
        DeviceArray<T> b(a.dims());
        thrust::device_ptr<const T> in(a.begin());
        thrust::device_ptr<T> out(b.begin());
        thrust::transform(in, in + a.size(), out, detail::AbsFn<T>{});
        return b;
    }

    // max / min: real arrays
    template <Real_t T>
    T max(const DeviceArray<T> &a) {
        thrust::device_ptr<const T> first(a.begin());
        return *thrust::max_element(first, first + a.size());
    }
    template <Real_t T>
    T min(const DeviceArray<T> &a) {
        thrust::device_ptr<const T> first(a.begin());
        return *thrust::min_element(first, first + a.size());
    }

    // L2 norm: sqrt(sum of squares)
    template <Real_t T>
    T norm2(const DeviceArray<T> &a) {
        thrust::device_ptr<const T> first(a.begin());
        T sq = thrust::transform_reduce(first, first + a.size(), detail::SqFn<T>{},
                                        T(0), thrust::plus<T>{});
        return std::sqrt(sq);
    }

    // norm1: L1 norm  (sum of absolute values)
    template <Real_t T>
    T norm1(const DeviceArray<T> &a) {
        thrust::device_ptr<const T> first(a.begin());
        return thrust::transform_reduce(first, first + a.size(), detail::AbsFn<T>{},
                                        T(0), thrust::plus<T>{});
    }

    // dot: inner product of two real arrays
    template <Real_t T>
    T dot(const DeviceArray<T> &a, const DeviceArray<T> &b) {
        thrust::device_ptr<const T> dev_a(a.begin());
        thrust::device_ptr<const T> dev_b(b.begin());
        return thrust::inner_product(dev_a, dev_a + a.size(), dev_b, T(0));
    }

    // sqrt of each element: out = sqrt(a)
    template <typename T>
    DeviceArray<T> sqrt(const DeviceArray<T> &a) {
        DeviceArray<T> out(a.dims());
        thrust::device_ptr<const T> pa(a.begin());
        thrust::device_ptr<T> pout(out.begin());
        thrust::transform(pa, pa + a.size(), pout, detail::SqrtFn<T>{});
        return out;
    }

} // namespace tomocam::gpu::array

#endif // GPU_DEVICE_ARRAY_OPS_H
