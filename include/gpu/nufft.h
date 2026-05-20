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

#include <cmath>
#include <complex>
#include <vector>

#include "array.h"
#include "dtypes.h"
#include "gpu/cufinufft_plan_cache.h"
#include "gpu/device_array.h"
#include "gpu/polar_grid.h"
#include "gpu/utils.h"
#include "polar_grid.h"

namespace tomocam::gpu::nufft {

    template <typename T>
    using complex = cuda::std::complex<T>;

    // 3D Type-1 NUFFT (CPU arrays, CPU PolarGrid): nonuniform -> uniform
    template <typename T>
    void nufft3d1(const Array<std::complex<T>> &cz, Array<std::complex<T>> &fz,
                  const PolarGrid<T> &pg) {

        int gpu_id;
        cudaGetDevice(&gpu_id);

        // move data to device
        auto d_cz = gpu::DeviceArray<complex<T>>(cz.dims());
        gpu::copy_to_device(cz.begin(), d_cz.begin(), d_cz.bytes());

        // allocate output array on device
        auto d_fz = gpu::DeviceArray<complex<T>>(fz.dims());

        // get plan from cache
        std::array<int64_t, 3> n_modes = {(int64_t)fz.ncols(), (int64_t)fz.nrows(),
                                          (int64_t)fz.nslices()};
        auto &plan = nufft::plans::cache<T>.get_plan(1, 3, n_modes, 1, gpu_id);
        plan.set_points(pg);

        // execute the plan
        plan.execute(d_cz.begin(), d_fz.begin());

        // copy result back to host
        gpu::copy_to_host(d_fz.begin(), fz.begin(), d_fz.bytes());
    }

    // 3D Type-2 NUFFT (CPU arrays, CPU PolarGrid): uniform -> nonuniform
    template <typename T>
    void nufft3d2(Array<std::complex<T>> &cz, const Array<std::complex<T>> &fz,
                  const PolarGrid<T> &pg) {

        int gpu_id;
        cudaGetDevice(&gpu_id);

        // move data to device
        auto d_fz = gpu::DeviceArray<complex<T>>(fz.dims());
        gpu::copy_to_device(fz.begin(), d_fz.begin(), d_fz.bytes());

        // allocate output array on device
        auto d_cz = gpu::DeviceArray<complex<T>>(cz.dims());

        // get plan from cache
        std::array<int64_t, 3> n_modes = {(int64_t)fz.ncols(), (int64_t)fz.nrows(),
                                          (int64_t)fz.nslices()};
        auto &plan = nufft::plans::cache<T>.get_plan(2, 3, n_modes, -1, gpu_id);
        plan.set_points(pg);

        // execute the plan — order is (NU Data, Uniform Data)
        plan.execute(d_cz.begin(), d_fz.begin());

        // copy result back to host
        gpu::copy_to_host(d_cz.begin(), cz.begin(), d_cz.bytes());
    }

    // 3D Type-1 NUFFT (DeviceArrays, GPU PolarGrid): nonuniform -> uniform
    template <typename T>
    void nufft3d1(const DeviceArray<complex<T>> &cz, DeviceArray<complex<T>> &fz,
                  const gpu::PolarGrid<T> &pg) {

        int gpu_id;
        cudaGetDevice(&gpu_id);

        std::array<int64_t, 3> n_modes = {(int64_t)fz.ncols(), (int64_t)fz.nrows(),
                                          (int64_t)fz.nslices()};
        auto &plan = nufft::plans::cache<T>.get_plan(1, 3, n_modes, 1, gpu_id);
        plan.set_points(pg);
        plan.execute(const_cast<complex<T> *>(cz.begin()), fz.begin());
    }

    // 3D Type-2 NUFFT (DeviceArrays, GPU PolarGrid): uniform -> nonuniform
    template <typename T>
    void nufft3d2(DeviceArray<complex<T>> &cz, const DeviceArray<complex<T>> &fz,
                  const gpu::PolarGrid<T> &pg) {

        int gpu_id;
        cudaGetDevice(&gpu_id);

        std::array<int64_t, 3> n_modes = {(int64_t)fz.ncols(), (int64_t)fz.nrows(),
                                          (int64_t)fz.nslices()};
        auto &plan = nufft::plans::cache<T>.get_plan(2, 3, n_modes, -1, gpu_id);
        plan.set_points(pg);
        // execute — order is (NU Data, Uniform Data)
        plan.execute(cz.begin(), const_cast<complex<T> *>(fz.begin()));
    }

} // namespace tomocam::gpu::nufft

#endif // CUNUFFT_H
