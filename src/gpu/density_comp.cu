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

#include <thrust/fill.h>
#include <thrust/transform.h>

#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/nufft.h"
#include "gpu/polar_grid.h"
#include "gpu/precond.h"

namespace tomocam::gpu::opt {

    // -------------------------------------------------------------------------
    // Thrust functors (must be struct-based to work without --extended-lambda)
    // -------------------------------------------------------------------------

    template <typename T>
    using complex_t = cuda::std::complex<T>;

    // w[j] /= max(|v[j]|, eps)
    template <typename T>
    struct DivAbsFn {
        T eps;
        explicit DivAbsFn(T e = T(1e-12)) : eps(e) {}
        __host__ __device__ complex_t<T> operator()(complex_t<T> wj,
                                                    complex_t<T> vj) const {
            T av = cuda::std::abs(vj);
            if (av < eps) av = eps;
            return wj / av;
        }
    };

    // clamp: max(x, floor_val)
    template <typename T>
    struct ClampFloorFn {
        T floor_val;
        explicit ClampFloorFn(T f) : floor_val(f) {}
        __host__ __device__ T operator()(T x) const {
            return x < floor_val ? floor_val : x;
        }
    };

    // elementwise real division: z = r / w
    template <typename T>
    struct DivFn {
        __host__ __device__ T operator()(T a, T b) const { return a / b; }
    };

    // -------------------------------------------------------------------------
    // DensityComp constructor: Pipe-Menon Jacobi iterations on GPU
    // -------------------------------------------------------------------------
    template <typename T>
    DensityComp<T>::DensityComp(const gpu::PolarGrid<T> &pg, dims_t recon_dims,
                                 size_t n_iters) {
        dims_t pg_dims = pg.dims();

        // Initialize nonuniform complex weights w to (1 + 0i)
        DeviceArray<T> ones_real(pg_dims);
        thrust::fill(ones_real.begin(), ones_real.end(), T(1));
        DeviceArray<complex_t<T>> w = gpu::array::to_complex(ones_real);

        // Uniform and nonuniform complex work arrays
        DeviceArray<complex_t<T>> u(recon_dims);
        DeviceArray<complex_t<T>> v(pg_dims);

        for (size_t iter = 0; iter < n_iters; iter++) {
            // u = type-1(w): nonuniform → uniform
            gpu::nufft::nufft3d1(w, u, pg);
            // v = type-2(u): uniform → nonuniform
            gpu::nufft::nufft3d2(v, u, pg);
            // w[j] /= max(|v[j]|, 1e-12)
            thrust::transform(w.begin(), w.end(), v.begin(), w.begin(),
                              DivAbsFn<T>());
        }

        // Final density map: u = type-1(w), take real part, floor-clamp
        gpu::nufft::nufft3d1(w, u, pg);
        DeviceArray<T> d = gpu::array::to_real(u);
        T floor_val = T(1e-3) * gpu::array::max(d);
        w_ = DeviceArray<T>(recon_dims);
        thrust::transform(d.begin(), d.end(), w_.begin(), ClampFloorFn<T>(floor_val));
    }

    // -------------------------------------------------------------------------
    // DensityComp::apply: z[i] = r[i] / w_[i]
    // -------------------------------------------------------------------------
    template <typename T>
    void DensityComp<T>::apply(const DeviceArray<T> &r, DeviceArray<T> &z) const {
        z = DeviceArray<T>(r.dims());
        thrust::transform(r.begin(), r.end(), w_.begin(), z.begin(), DivFn<T>());
    }

    // Explicit instantiations
    template class DensityComp<float>;
    template class DensityComp<double>;

} // namespace tomocam::gpu::opt
