
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
