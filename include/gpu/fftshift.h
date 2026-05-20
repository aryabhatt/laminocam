
#ifndef TOMOCAM_GPU_FFTSHIFT_H
#define TOMOCAM_GPU_FFTSHIFT_H

#include <cuda_runtime.h>

#include "gpu/device_array.h"

namespace tomocam::gpu {

    template <typename T>
    DeviceArray<T> roll(const DeviceArray<T> &input, int3 delta);

    template <typename T>
    DeviceArray<T> fftshift2(const DeviceArray<T> &input) {
        size_t nrows = input.nrows();
        size_t ncols = input.ncols();
        int3 delta = {0, 0, 0};
        delta.y = nrows / 2;
        delta.z = ncols / 2;

        return roll(input, delta);
    }

    template <typename T>
    DeviceArray<T> ifftshift2(const DeviceArray<T> &input) {
        size_t nrows = input.nrows();
        size_t ncols = input.ncols();
        int3 delta = {0, 0, 0};
        delta.y = nrows / 2;
        if (nrows % 2 == 1) { delta.y += 1; }
        delta.z = ncols / 2;
        if (ncols % 2 == 1) { delta.z += 1; }
        return roll(input, delta);
    }
} // namespace tomocam::gpu

#endif // TOMOCAM_GPU_FFTSHIFT_H
