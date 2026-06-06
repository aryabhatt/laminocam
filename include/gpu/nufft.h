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

#ifndef CUNUFFT_H
#define CUNUFFT_H

#include "dtypes.h"
#include "gpu/cufinufft_plan_cache.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/polar_grid.h"
#include "gpu/utils.h"

namespace tomocam::gpu::nufft {

    template <typename T>
    using complex_t = cuda::std::complex<T>;

    // 3D Type-1 gpu::NUFFT (Non-uniform -> Uniform)
    template <typename T>
    void nufft3d1(DeviceArray<complex_t<T>> &cz, DeviceArray<complex_t<T>> &fz,
                  const gpu::PolarGrid<T> &pg) {

        // get current GPU ID
        int gpu_id;
        cudaGetDevice(&gpu_id);

        // get plan from cache
        std::array<int64_t, 3> n_modes = {(int64_t)fz.ncols(), (int64_t)fz.nrows(),
                                          (int64_t)fz.nslices()};
        // get plan from cache (type, dims, n_modes, sign, gpu_id)
        auto &plan = plans::cache<T>.get_plan(1, 3, n_modes, 1, gpu_id);
        plan.set_points(pg);

        // execute the plan
        plan.execute(cz.data(), fz.data());
    }

    // 3D Type-2 NUFFT (GPU arrays, GPU PolarGrid): uniform -> nonuniform
    template <typename T>
    void nufft3d2(DeviceArray<complex_t<T>> &cz, DeviceArray<complex_t<T>> &fz,
                  const gpu::PolarGrid<T> &pg) {

        int gpu_id;
        cudaGetDevice(&gpu_id);

        // get plan from cache
        std::array<int64_t, 3> n_modes = {(int64_t)fz.ncols(), (int64_t)fz.nrows(),
                                          (int64_t)fz.nslices()};
        // get plan from cache (type, dims, n_modes, sign, gpu_id)
        auto &plan = plans::cache<T>.get_plan(2, 3, n_modes, -1, gpu_id);
        plan.set_points(pg);

        // execute the plan — order is (NU Data, Uniform Data)
        plan.execute(cz.data(), fz.data());
    }

} // namespace tomocam::gpu::nufft

#endif // CUNUFFT_H
