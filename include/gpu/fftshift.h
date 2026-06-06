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
