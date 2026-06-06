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
