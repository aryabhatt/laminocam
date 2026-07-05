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

    template <typename T>
    DeviceArray<T> pad3d(const DeviceArray<T> &input, dims_t out_dims, PadType type);

} // namespace tomocam::gpu

#endif // GPU_PADDING_H
