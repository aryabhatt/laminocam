#ifndef GPU_PADDING_H
#define GPU_PADDING_H

#include "gpu/device_array.h"
#include "padding.h"

namespace tomocam::gpu {
    /**
     * @tparam T
     * @param input
     * @param padding
     * @param type
     * @return A new DeviceArray with the specified padding applied to the input
     * array. The type of padding is determined by the PadType enum.
     *
     */
    template <typename T>
    DeviceArray<T> pad2d(const DeviceArray<T> &input, float factor, PadType type);

    /**
     * @tparam T
     * @param input
     * @param output dimensions
     * @param type
     * @return A new DeviceArray with the specified padding applied to the input
     * array. The type of padding is determined by the PadType enum.
     *
     */
    template <typename T>
    DeviceArray<T> crop3d(const DeviceArray<T> &input, dims_t out_dims,
                          PadType type);

} // namespace tomocam::gpu

#endif // GPU_PADDING_H
