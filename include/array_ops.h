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

#include <algorithm>
#include <complex>
#include <execution>
#include <numeric>
#include <type_traits>

#include "array.h"

#ifndef ARRAY_OPS__H
#define ARRAY_OPS__H

namespace tomocam::array {

    template <typename T>
    concept Real_t = std::is_same_v<T, float> | std::is_same_v<T, double>;

    template <typename Real_t>
    Array<std::complex<Real_t>> to_complex(const Array<Real_t> &a) {
        Array<std::complex<Real_t>> b(a.dims());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(),
                       [](Real_t x) { return std::complex<Real_t>(x); });
        return b;
    }

    template <typename Real_t>
    Array<Real_t> to_real(const Array<std::complex<Real_t>> &a) {
        Array<Real_t> b(a.dims());
        std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(),
                       [](std::complex<Real_t> x) { return x.real(); });
        return b;
    }

    template <typename T>
    T max(const Array<T> &a) {
        return *std::max_element(std::execution::par_unseq, a.begin(), a.end());
    }

    template <typename T>
    T min(const Array<T> &a) {
        return *std::min_element(std::execution::par_unseq, a.begin(), a.end());
    }

    template <typename T>
    T norm2(const Array<T> &a) {
        return std::sqrt(std::transform_reduce(std::execution::par_unseq, a.begin(),
                                               a.end(), T(0), std::plus<T>(),
                                               [](T x) { return x * x; }));
    }

    template <typename T>
    T norm1(const Array<T> &a) {
        return std::transform_reduce(std::execution::par_unseq, a.begin(), a.end(),
                                     T(0), std::plus<T>(),
                                     [](T x) { return std::abs(x); });
    }

    template <typename T>
    T dot(const Array<T> &a, const Array<T> &b) {
        return std::transform_reduce(std::execution::par_unseq, a.begin(), a.end(),
                                     b.begin(), T(0), std::plus<T>(),
                                     [](T x, T y) { return x * y; });
    }
    // Transpose the array a according to the given axes.
    template <typename T>
    Array<T> transpose(const Array<T> &a, std::array<size_t, 3> axes) {

        // if axes is {0, 1, 2}, return a copy
        if (axes == std::array<size_t, 3>{0, 1, 2}) { return a.clone(); }

        dims_t dims = a.dims();
        std::array<size_t, 3> dims_arr = {dims.n1, dims.n2, dims.n3};
        dims_t new_dims = {dims_arr[axes[0]], dims_arr[axes[1]], dims_arr[axes[2]]};
        Array<T> b(new_dims);
#pragma omp parallel for collapse(3)
        for (size_t i = 0; i < dims.n1; ++i) {
            for (size_t j = 0; j < dims.n2; ++j) {
                for (size_t k = 0; k < dims.n3; ++k) {
                    size_t idx1[3] = {i, j, k};
                    dims_t idx2 = {idx1[axes[0]], idx1[axes[1]], idx1[axes[2]]};
                    b[idx2] = a[{i, j, k}];
                }
            }
        }
        return b;
    }

} // namespace tomocam::array
#endif // ARRAY_OPS__H
