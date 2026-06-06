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
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>

#include "gpu/device_array.h"

namespace tomocam::gpu {
    /// Add
    template <typename T>
    DeviceArray<T> DeviceArray<T>::operator+(const DeviceArray<T> &other) const {
        DeviceArray<T> result(dims_);
        thrust::transform(this->begin(), this->end(), other.begin(), result.begin(),
                          thrust::plus<T>());
        return result;
    }
    /// In-place add
    template <typename T>
    DeviceArray<T> &DeviceArray<T>::operator+=(const DeviceArray<T> &other) {
        thrust::transform(this->begin(), this->end(), other.begin(), this->begin(),
                          thrust::plus<T>());
        return *this;
    }
    /// Subtract
    template <typename T>
    DeviceArray<T> DeviceArray<T>::operator-(const DeviceArray<T> &other) const {
        DeviceArray<T> result(dims_);
        thrust::transform(this->begin(), this->end(), other.begin(), result.begin(),
                          thrust::minus<T>());
        return result;
    }
    /// In-place subtract
    template <typename T>
    DeviceArray<T> &DeviceArray<T>::operator-=(const DeviceArray<T> &other) {
        thrust::transform(this->begin(), this->end(), other.begin(), this->begin(),
                          thrust::minus<T>());
        return *this;
    }

    /// scalar multiplication
    template <typename T>
    DeviceArray<T> &DeviceArray<T>::operator*=(T scalar) {
        thrust::transform(this->begin(), this->end(), this->begin(),
                          [scalar] __device__(const T &x) { return x * scalar; });
        return *this;
    }

    template <typename T>
    DeviceArray<T> DeviceArray<T>::operator*(const T &scalar) const {
        auto result = this->clone();
        result *= scalar;
        return result;
    }

    // multiply two arrays element-wise
    template <typename T>
    DeviceArray<T> DeviceArray<T>::operator*(const DeviceArray<T> &other) const {
        DeviceArray<T> result(dims_);
        thrust::transform(this->begin(), this->end(), other.begin(), result.begin(),
                          thrust::multiplies<T>());
        return result;
    }

    // In-place multiply two arrays element-wise
    template <typename T>
    DeviceArray<T> &DeviceArray<T>::operator*=(const DeviceArray<T> &other) {
        thrust::transform(this->begin(), this->end(), other.begin(), this->begin(),
                          thrust::multiplies<T>());
        return *this;
    }

    // scalar division
    template <typename T>
    DeviceArray<T> &DeviceArray<T>::operator/=(T scalar) {
        if (scalar == (T)0) { throw std::runtime_error("Division by zero"); }
        thrust::transform(this->begin(), this->end(), this->begin(),
                          [scalar] __device__(const T &x) { return x / scalar; });
        return *this;
    }

    template <typename T>
    DeviceArray<T> DeviceArray<T>::operator/(const T &scalar) const {
        if (scalar == (T)0) { throw std::runtime_error("Division by zero"); }
        auto result = this->clone();
        result /= scalar;
        return result;
    }

    // element-wise division
    template <typename T>
    DeviceArray<T> DeviceArray<T>::operator/(const DeviceArray<T> &other) const {
        DeviceArray<T> result(dims_);
        thrust::transform(this->begin(), this->end(), other.begin(), result.begin(),
                          thrust::divides<T>());
        return result;
    }
    // In-place element-wise division
    template <typename T>
    DeviceArray<T> &DeviceArray<T>::operator/=(const DeviceArray<T> &other) {
        thrust::transform(this->begin(), this->end(), other.begin(), this->begin(),
                          thrust::divides<T>());
        return *this;
    }
    // Explicit instantiations
    template class DeviceArray<float>;
    template class DeviceArray<double>;
    template class DeviceArray<cuda::std::complex<float>>;
    template class DeviceArray<cuda::std::complex<double>>;

} // namespace tomocam::gpu
