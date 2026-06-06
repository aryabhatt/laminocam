
#ifndef FINITEDIFF_H
#define FINITEDIFF_H

#include <cuda_runtime.h>

#include "gpu/device_array.h"
#include "gpu/device_ptr.h"
#include "gpu/utils.h"

namespace tomocam::gpu::opt {

    // gradient
    template <typename T>
    std::array<DeviceArray<T>, 3> grad(const DeviceArray<T> &u);

    // divergence
    template <typename T>
    DeviceArray<T> divergence(const std::array<DeviceArray<T>, 3> &u);

    // negative divergence
    template <typename T>
    DeviceArray<T> neg_divergence(const std::array<DeviceArray<T>, 3> &u) {
        auto div = divergence(u);
        return (div * (-1));
    }

    // laplacian
    template <typename T>
    DeviceArray<T> laplacian(const DeviceArray<T> &u);

    // negative laplacian
    template <typename T>
    DeviceArray<T> neg_laplacian(const DeviceArray<T> &u) {
        auto lap = laplacian(u);
        return (lap * (-1));
    }

} // namespace tomocam::gpu::opt

#endif // FINITEDIFF_H
