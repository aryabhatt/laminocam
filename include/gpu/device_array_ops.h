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

#include <type_traits>

#include <cuda/std/complex>

#include "gpu/device_array.h"

namespace tomocam::gpu::array {

    template <typename T>
    concept Real_t = std::is_same_v<T, float> || std::is_same_v<T, double>;

    /// multiply complex array by real array: a = a * b
    template <Real_t T>
    void cmul(DeviceArray<cuda::std::complex<T>> &a, const DeviceArray<T> &b);

    /// axpy: x = alpha * x + y
    template <typename T>
    void axpy(DeviceArray<T> &x, T alpha, const DeviceArray<T> &y);

    /// xpay: x = x + y *
    template <typename T>
    void xpay(DeviceArray<T> &x, const DeviceArray<T> &y, T beta);

    template <typename T>
    DeviceArray<T> sqrt(const DeviceArray<T> &a);

    template <Real_t T>
    DeviceArray<cuda::std::complex<T>> to_complex(const DeviceArray<T> &a);
    template <Real_t T>
    DeviceArray<T> to_real(const DeviceArray<cuda::std::complex<T>> &a);

    template <Real_t T>
    DeviceArray<T> abs(const DeviceArray<T> &a);
    template <Real_t T>
    T max(const DeviceArray<T> &a);
    template <Real_t T>
    T min(const DeviceArray<T> &a);
    template <Real_t T>
    T norm2(const DeviceArray<T> &a);
    template <Real_t T>
    T norm1(const DeviceArray<T> &a);
    template <Real_t T>
    T dot(const DeviceArray<T> &a, const DeviceArray<T> &b);

} // namespace tomocam::gpu::array

#endif // GPU_DEVICE_ARRAY_OPS_H
