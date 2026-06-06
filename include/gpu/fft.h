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


#ifndef GPU_FFT_H
#define GPU_FFT_H

#include <array>
#include <cuda/std/complex>

#include "gpu/cufft_plan_cache.h"
#include "gpu/device_array.h"

namespace tomocam::gpu::fft {

    template <typename T>
    using complex = cuda::std::complex<T>;

    template <typename T>
    DeviceArray<complex<T>> fft2d(DeviceArray<complex<T>> &data) {

        // get dimensions
        int batch = static_cast<int>(data.nslices());
        int n1 = static_cast<int>(data.nrows());
        int n2 = static_cast<int>(data.ncols());

        // allocate output array
        DeviceArray<complex<T>> output(data.dims());

        // get plan from cache
        int dim = 2;
        std::array<int, 3> n_modes = {batch, n1, n2};
        int device_id = -1;
        SAFE_CALL(cudaGetDevice(&device_id));
        auto &plan = cache::plans<T>.get_plan(dim, n_modes, CUFFT_C2C, device_id);
        plan.execute(data.data(), output.data(), CUFFT_FORWARD);
        return output;
    }

    template <typename T>
    DeviceArray<complex<T>> ifft2d(DeviceArray<complex<T>> &data) {

        // get dimensions
        int batch = static_cast<int>(data.nslices());
        int n1 = static_cast<int>(data.nrows());
        int n2 = static_cast<int>(data.ncols());

        // allocate output array
        DeviceArray<complex<T>> output(data.dims());

        // get plan from cache
        int dim = 2;
        std::array<int, 3> n_modes = {batch, n1, n2};
        int device_id = -1;
        SAFE_CALL(cudaGetDevice(&device_id));
        auto &plan = cache::plans<T>.get_plan(dim, n_modes, CUFFT_C2C, device_id);
        plan.execute(data.data(), output.data(), CUFFT_INVERSE);
        return output;
    }

    template <typename T>
    DeviceArray<complex<T>> rfft2d(DeviceArray<T> &data) {

        // get dimensions
        int batch = static_cast<int>(data.nslices());
        int n1 = static_cast<int>(data.nrows());
        int n2 = static_cast<int>(data.ncols());

        // allocate output array
        DeviceArray<complex<T>> output({batch, n1, n2 / 2 + 1});

        // get plan from cache
        int dim = 2;
        std::array<int, 3> n_modes = {batch, n1, n2};
        int device_id = -1;
        SAFE_CALL(cudaGetDevice(&device_id));
        auto &plan = cache::plans<T>.get_plan(dim, n_modes, CUFFT_R2C, device_id);
        plan.execute(data.data(), output.data());
        return output;
    }

    template <typename T>
    DeviceArray<T> irfft2d(DeviceArray<complex<T>> &data, dims_t output_dims) {

        // get dimensions
        int batch = static_cast<int>(output_dims.n1);
        int n1 = static_cast<int>(output_dims.n2);
        int n2 = static_cast<int>(output_dims.n3);

        // allocate output array
        DeviceArray<T> output(output_dims);

        // get plan from cache
        int dim = 2;
        std::array<int, 3> n_modes = {batch, n1, n2};

        int device_id = -1;
        SAFE_CALL(cudaGetDevice(&device_id));
        auto &plan = cache::plans<T>.get_plan(dim, n_modes, CUFFT_C2R, device_id);
        plan.execute(data.data(), output.data());
        return output;
    }

} // namespace tomocam::gpu::fft

#endif // GPU_FFT_H
