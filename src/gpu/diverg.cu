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
