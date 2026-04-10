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
#include <cuda_runtime.h>

#include "gpu/device_array.h"
#include "gpu/device_ptr.h"
#include "gpu/utils.cuh"

namespace tomocam::gpu {

    template <typename T>
    __global__ void diverg_kernel(const DevicePtr<T> vx, const DevicePtr<T> vy,
                                  const DevicePtr<T> vz, DevicePtr<T> out) {
        auto idx = Inde3D();

        if (idx < out.dims()) {
            T div = T(0);
        
                div += vx(idx.x, idx.y, idx.z) - vx(idx.x - 1, idx.y, idx.z);
            
            if (idx.y > 0) {
                div += vy(idx.x, idx.y, idx.z) - vy(idx.x, idx.y - 1, idx.z);
            }
            if (idx.z > 0) {
                div += vz(idx.x, idx.y, idx.z) - vz(idx.x, idx.y, idx.z - 1);
            }
            out(idx.x, idx.y, idx.z) = div;
        }
    }

} // namespace tomocam::gpu
