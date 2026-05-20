

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
        plan.execute(data.begin(), output.begin(), CUFFT_FORWARD);
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
        plan.execute(data.begin(), output.begin(), CUFFT_INVERSE);
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
        plan.execute(data.begin(), output.begin());
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
        plan.execute(data.begin(), output.begin());
        return output;
    }

} // namespace tomocam::gpu::fft

#endif // GPU_FFT_H
