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

#ifndef ARRAY3__H
#define ARRAY3__H

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <execution>
#include <memory>
#include <random>
#include <tuple>
#include <type_traits>

#include "dtypes.h"

namespace tomocam {

    // a non-owning view into a 3D array
    template <typename T>
    struct Slice {
        std::array<size_t, 2> dims; // nrows, ncols
        T *ptr;

        /// methods
        size_t size() const { return dims[0] * dims[1]; }
        // iterators
        T *begin() { return ptr; }
        const T *begin() const { return ptr; }
        T *end() { return ptr + (dims[0] * dims[1]); }
        const T *end() const { return ptr + (dims[0] * dims[1]); }
        // indexing
        T &operator[](std::array<size_t, 2> idx) {
            return ptr[idx[0] * dims[1] + idx[1]];
        }
        const T &operator[](std::array<size_t, 2> idx) const {
            return ptr[idx[0] * dims[1] + idx[1]];
        }
    };

    template <typename T>
    class Array {
      private:
        dims_t dims_;
        size_t size_;
        std::unique_ptr<T[]> ptr_;

        // get flat index
        size_t flatIdx(size_t i, size_t j, size_t k) const {
            return dims_.flat_idx(i, j, k);
        }

      public:
        Array() : dims_(0, 0, 0), size_(0), ptr_(nullptr) {}

        Array(size_t x, size_t y, size_t z)
            : dims_(x, y, z), size_(dims_.size()),
              ptr_(std::make_unique<T[]>(size_)) {}

        Array(dims_t d)
            : dims_(d), size_(dims_.size()), ptr_(std::make_unique<T[]>(size_)) {}

        Array(const Array<T> &) = delete;
        Array<T> &operator=(const Array<T> &) = delete;
        Array(Array<T> &&) noexcept = default;
        Array<T> &operator=(Array<T> &&) noexcept = default;

        [[nodiscard]] Array<T> clone() const {
            Array<T> rv(dims_);
            std::copy(this->begin(), this->end(), rv.begin());
            return std::move(rv);
        }

        T *begin() { return ptr_.get(); }
        const T *begin() const { return ptr_.get(); }
        T *end() { return ptr_.get() + size_; }
        const T *end() const { return ptr_.get() + size_; }

        [[nodiscard]] dims_t dims() const { return dims_; }
        [[nodiscard]] size_t size() const { return size_; }
        [[nodiscard]] size_t nslices() const { return dims_.x(); }
        [[nodiscard]] size_t nrows() const { return dims_.y(); }
        [[nodiscard]] size_t ncols() const { return dims_.z(); }

        // indexing
        T &operator[](size_t i) { return ptr_[i]; }
        const T &operator[](size_t i) const { return ptr_[i]; }

        // indexing for total-variation regularization (with OOB handling)
        T at(size_t i, size_t j, size_t k) const {
            if (i < 0 || i >= dims_.x() || j < 0 || j >= dims_.y() || k < 0 ||
                k >= dims_.z()) {
                return T(0); // for out-of-bounds, return zero
            }
            return ptr_[flatIdx(i, j, k)];
        }

#if (__cplusplus == 202302L)
        T &operator[](size_t i, size_t j, size_t k) {
            return ptr_[flatIdx(i, j, k)];
        }
        T operator[](size_t i, size_t j, size_t k) const {
            return ptr_[flatIdx(i, j, k)];
        }
#else
        T &operator[](dims_t i) { return ptr_[flatIdx(i.x(), i.y(), i.z())]; }
        const T &operator[](dims_t i) const {
            return ptr_[flatIdx(i.x(), i.y(), i.z())];
        }
#endif

        // get contiguous view to part or whole array
        Slice<T> slice(size_t i) {
            size_t nrows = dims_.n2;
            size_t ncols = dims_.n3;
            T *ptr = ptr_.get() + (i * dims_.n2 * dims_.n3);
            return Slice<T>{{nrows, ncols}, ptr};
        }
        const Slice<T> slice(size_t i) const {
            size_t nrows = dims_.n2;
            size_t ncols = dims_.n3;
            T *ptr = ptr_.get() + (i * dims_.n2 * dims_.n3);
            return Slice<T>{{nrows, ncols}, ptr};
        }

        // multiplication operators
        Array<T> &operator*=(T v) {
            std::transform(std::execution::par_unseq, this->begin(), this->end(),
                           this->begin(), [v](T x) { return x * v; });
            return *this;
        }
        Array<T> operator*(T v) {
            auto tmp = this->clone();
            tmp *= v;
            return tmp;
        }

        // division
        Array<T> &operator/=(T v) {
            if (std::abs(v) == 0) {
                throw std::runtime_error("Division by zero in Array<T>::operator/=");
            }
            std::transform(std::execution::par_unseq, this->begin(), this->end(),
                           ptr_.get(), [v](T x) { return x / v; });
            return *this;
        }
        Array<T> &operator/=(const Array<T> &v) {
            std::transform(std::execution::par_unseq, this->begin(), this->end(),
                           v.ptr_.get(), ptr_.get(), [](T x, T y) {
                               if (y == 0) {
                                   throw std::runtime_error(
                                       "Division by zero in Array<T>::operator/=");
                               }
                               return x / y;
                           });
            return *this;
        }
        Array<T> operator/(T scalar) const {
            auto rv = this->clone();
            rv /= scalar;
            return rv;
        }
        Array<T> operator/(const Array<T> &rhs) const {
            auto rv = this->clone();
            rv /= rhs;
            return rv;
        }

        // addition
        Array<T> &operator+=(T v) {
            std::transform(std::execution::par_unseq, this->begin(), this->end(),
                           ptr_.get(), [v](T x) { return x + v; });
            return *this;
        }

        Array<T> &operator+=(const Array<T> &v) {
            std::transform(std::execution::par_unseq, this->begin(), this->end(),
                           v.ptr_.get(), ptr_.get(), std::plus<T>());
            return *this;
        }
        Array<T> operator+(const Array<T> &rhs) const {
            auto tmp = this->clone();
            tmp += rhs;
            return tmp;
        }

        Array<T> &operator+=(const Array<T> &rhs) const {
            auto tmp = this->clone();
            tmp += rhs;
            return tmp;
        }

        // subtraction
        Array<T> &operator-=(T v) {
            std::transform(std::execution::par_unseq, this->begin(), this->end(),
                           ptr_.get(), [v](T x) { return x - v; });
            return *this;
        }
        Array<T> &operator-=(const Array<T> &v) {
            std::transform(std::execution::par_unseq, this->begin(), this->end(),
                           v.ptr_.get(), ptr_.get(), std::minus<T>());
            return *this;
        }

        Array<T> operator-(const Array<T> &rhs) const {
            auto rv = this->clone();
            rv -= rhs;
            return rv;
        }

        // factory methods
        static Array<T> zeros(const dims_t &ref_dims) {
            Array<T> rv(ref_dims);
            std::fill(std::execution::par_unseq, rv.begin(), rv.end(), T(0));
            return rv;
        }
        //  new array filled with value v
        static Array<T> ones(const dims_t &ref_dims) {
            Array<T> rv(ref_dims);
            std::fill(std::execution::par_unseq, rv.begin(), rv.end(), (T(1)));
            return rv;
        }

        // new array with random values
        static Array<T> random(const dims_t &ref_dims) {
            Array<T> rv(ref_dims);
            std::random_device rd;
            std::mt19937 gen(rd());
            if constexpr (std::is_floating_point<T>::value) {
                std::uniform_real_distribution<T> dis(0.0, 1.0);
                std::generate(std::execution::par_unseq, rv.begin(), rv.end(),
                              [&]() { return dis(gen); });
            } else {
                static_assert(std::is_floating_point<T>::value,
                              "Unsupported type for random_like");
            }
            return rv;
        }
    };
} // namespace tomocam
#endif // ARRAY3__H
