/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 *IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 *the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#ifndef TOMOCAM_DEV_ARRAY_H
#define TOMOCAM_DEV_ARRAY_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/device_ptr.h>

#include "array.h"
#include "gpu/device_ptr.h"
#include "gpu/memory.h"
#include "gpu/utils.h"

namespace tomocam::gpu {

    /// Container wrapping a CUDA device pointer as a 3D array.
    /// Manages device memory with RAII semantics using cunique_ptr. Provides
    /// 3D indexing through implicit conversion to DevicePtr for use in CUDA kernels.
    ///
    /// @tparam T Element type of the device array
    ///
    /// Usage:
    /// @code
    /// DeviceArray<float> dev_array({10, 20, 30});  // Allocate 10x20x30
    ///
    /// Slice<float> host_data(...);
    /// DeviceArray<float> dev_from_host(host_data);  // Copy from host
    ///
    /// __global__ void my_kernel(DevicePtr<float> data) {
    ///     float val = data(i, j, k);
    /// }
    /// my_kernel<<<blocks, threads>>>(dev_array);  // Implicit conversion to
    /// DevicePtr
    /// @endcode
    template <typename T>
    class DeviceArray {
      public:
        using value_type = T;

      protected:
        dims_t dims_;
        size_t size_;
        memory::cunique_ptr<T> dev_ptr_;

      public:
        /// Default constructor creates empty array.
        DeviceArray() : dims_(), size_(0), dev_ptr_(nullptr) {}

        /// Allocates device memory for a 3D array of given dimensions.
        /// Memory is zero-initialized.
        DeviceArray(dims_t d) : dims_(d) {
            size_ = d.n1 * d.n2 * d.n3;
            dev_ptr_ = memory::make_cunique_ptr<T>(size_);
            SAFE_CALL(cudaMemset(dev_ptr_.get(), 0, bytes()));
        }

        /// Constructs device array from Array<T>
        DeviceArray(const Array<T> &rhs)
            : dims_(rhs.dims()), size_(rhs.size()),
              dev_ptr_(memory::make_cunique_ptr<T>(rhs.size())) {
            copyH2D(dev_ptr_.get(), rhs.begin(), bytes());
        }

        /// Copy constructor is deleted to prevent unintended device memory
        /// duplication.
        DeviceArray(const DeviceArray<T> &rhs) = delete;
        /// Copy assignment is deleted to prevent unintended device memory
        /// duplication.
        DeviceArray<T> &operator=(const DeviceArray<T> &rhs) = delete;

        /// Destructor automatically managed by cunique_ptr.
        ~DeviceArray() = default;

        /// Creates explicit clone of the device array on GPU.
        /// Allocates without zero-initialization since the full contents are
        /// immediately overwritten by the device-to-device copy.
        [[nodiscard]] DeviceArray<T> clone() const {
            DeviceArray<T> copy;
            copy.dims_ = dims_;
            copy.size_ = size_;
            copy.dev_ptr_ = memory::make_cunique_ptr<T>(size_);
            copyD2D(copy.dev_ptr_.get(), dev_ptr_.get(), bytes());
            return copy;
        }

        /// copy data back to host as Array
        [[nodiscard]] Array<T> to_host() const {
            Array<T> host_array(dims_);
            copyD2H(host_array.begin(), dev_ptr_.get(), bytes());
            return host_array;
        }

        /// Move constructor for efficient transfer of ownership.
        DeviceArray(DeviceArray<T> &&rhs) = default;

        /// Move assignment operator for efficient transfer of ownership.
        DeviceArray<T> &operator=(DeviceArray<T> &&rhs) = default;

        /// Implicit conversion operator to DevicePtr for use in CUDA kernels.
        operator DevicePtr<T>() { return DevicePtr<T>(dev_ptr_.get(), dims_); }
        operator DevicePtr<const T>() const {
            return DevicePtr<const T>(dev_ptr_.get(), dims_);
        }

        /// Not dereferenceable on the host.
        T *data() { return dev_ptr_.get(); }
        const T *data() const { return dev_ptr_.get(); }

        /// Iterators for thrust compatibility.
        thrust::device_ptr<T> begin() {
            return thrust::device_ptr<T>(dev_ptr_.get());
        }
        thrust::device_ptr<const T> begin() const {
            return thrust::device_ptr<const T>(dev_ptr_.get());
        }

        thrust::device_ptr<T> end() {
            return thrust::device_ptr<T>(dev_ptr_.get() + size_);
        }
        thrust::device_ptr<const T> end() const {
            return thrust::device_ptr<const T>(dev_ptr_.get() + size_);
        }

        /// Returns total number of elements in the array.
        size_t size() const { return size_; }

        /// Returns total size in bytes of the array.
        size_t bytes() const { return sizeof(T) * size_; }

        /// Returns dimensions (n1, n2, n3) of the 3D array.
        dims_t dims() const { return dims_; }

        /// Returns number of slices (first dimension).
        size_t nslices() const { return dims_.n1; }

        /// Returns number of rows (second dimension).
        size_t nrows() const { return dims_.n2; }

        /// Returns number of columns (third dimension).
        size_t ncols() const { return dims_.n3; }

        /// arithmatic operators
        DeviceArray<T> operator+(const DeviceArray<T> &rhs) const;
        DeviceArray<T> &operator+=(const DeviceArray<T> &rhs);
        DeviceArray<T> operator-(const DeviceArray<T> &rhs) const;
        DeviceArray<T> &operator-=(const DeviceArray<T> &rhs);
        DeviceArray<T> &operator*=(T scalar);
        DeviceArray<T> operator*(const T &scalar) const;
        DeviceArray<T> operator*(const DeviceArray<T> &rhs) const;
        DeviceArray<T> &operator*=(const DeviceArray<T> &rhs);
        DeviceArray<T> &operator/=(T scalar);
        DeviceArray<T> operator/(const T &scalar) const;
        DeviceArray<T> operator/(const DeviceArray<T> &rhs) const;
        DeviceArray<T> &operator/=(const DeviceArray<T> &rhs);
    };

} // namespace tomocam::gpu

#endif // TOMOCAM_DEV_ARRAY__H
