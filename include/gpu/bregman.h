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

#ifndef TOMOCAM_GPU_BREGMAN_H
#define TOMOCAM_GPU_BREGMAN_H

#include <array>

#include "gpu/device_array.h"

namespace tomocam::gpu::opt {

    /**
     * @brief Compute isotropic norm sk[i] = sqrt( sum_j (dx[j][i] + b[j][i])^2 )
     *        used in the split-Bregman TV shrinkage step.
     */
    template <typename T>
    DeviceArray<T> compute_sk(const std::array<DeviceArray<T>, 3> &dx,
                              const std::array<DeviceArray<T>, 3> &b);

    /**
     * @brief Isotropic TV shrinkage:
     *        d[i] = max(0, sk[i] - lambda/mu) * (dx[i] + b[i]) / (sk[i] + eps)
     */
    template <typename T>
    DeviceArray<T> shrink(const DeviceArray<T> &dx, const DeviceArray<T> &b,
                          const DeviceArray<T> &sk, T lambda, T mu);

} // namespace tomocam::gpu::opt

#endif // TOMOCAM_GPU_BREGMAN_H
