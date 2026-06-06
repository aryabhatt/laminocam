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

#include <array>
#include <cstdio>
#include <format>
#include <iostream>
#include <limits>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/gpu_opt.h"
#include "gpu/mem_check.h"
#include "gpu/utils.h"

namespace tomocam::gpu::opt {

    // -------------------------------------------------------------------------
    // GPU Conjugate Gradient Solver
    // Same preconditioned CG algorithm as src/conjgrad.cpp, with GPU-native
    // DeviceArray types and Thrust/cuFFT inner operations.
    // -------------------------------------------------------------------------
    template <typename T>
    DeviceArray<T> cgsolver(const gpuFunction<T> &A, const DeviceArray<T> &y,
                            const DeviceArray<T> &x0, size_t max_iter, T tol,
                            T xtol) {

        // precondtioners
        DeviceArray<T> ones(x0.dims());
        thrust::fill(ones.begin(), ones.begin() + ones.size(), T(1));
        auto pre = A(ones);
        thrust::transform(
            pre.begin(), pre.begin() + pre.size(), pre.begin(),
            [] __device__(T val) { return abs(val) < T(1.2e-7) ? T(1e-6) : val; });

        auto precond_apply = [&pre](const DeviceArray<T> &r) {
            return r / pre;
        };
        // auto precond_apply = [](const DeviceArray<T> &r) { return r.clone(); };

        // Initialize solution and residual arrays
        DeviceArray<T> x = x0.clone();
        // r = y - A(x)
        auto r = y - A(x);

        // z = M^{-1} r,  p = z,  rs_old = z^T r
        DeviceArray<T> z = precond_apply(r);
        DeviceArray<T> p = z.clone();
        T rs_old = gpu::array::dot(z, r);
        // Checkpoint: ones + pre + x + r + z + p = 6 N (relative to caller baseline)
        MEM_CHECK("cg: persistent init (ones, pre, x, r, z, p = 6N relative)",
                  6 * x0.size() * sizeof(T));

        for (size_t iter = 0; iter < max_iter; iter++) {

            // Ap = A(p),  pAp = p^T Ap
            DeviceArray<T> Ap = A(p);
            // Checkpoint: 6N CG persistent + Ap (1N) + neg_laplacian peak (~2N) = ~9N relative
            // (neg_laplacian inside the Ap lambda allocates two arrays briefly)
            MEM_CHECK("cg: iter – after Ap=A(p) (6N CG + Ap + neg_lap peak = ~9N relative)",
                      9 * x0.size() * sizeof(T));
            T pAp = gpu::array::dot(p, Ap);
            T pAp_thresh = T(100) * std::numeric_limits<T>::epsilon() *
                           gpu::array::dot(p, p) * gpu::array::dot(Ap, Ap);
            if (std::abs(pAp) < std::sqrt(pAp_thresh)) {
                std::cerr << std::format("CG: p^T A p is too small ({:.5e}), stopping\n", pAp);
                break;
            }

            T alpha = rs_old / pAp;
            gpu::array::xpay(x, p, alpha);   // x += alpha * p
            gpu::array::xpay(r, Ap, -alpha); // r -= alpha * Ap

            // dx: relative step size (computed before p is updated)
            T dx = gpu::array::norm2(p * alpha) /
                   (gpu::array::norm2(x) + T(1e-8));

            // Apply preconditioner and compute new residual norm
            T rs_new = 0;
            z = precond_apply(r);
            rs_new = gpu::array::dot(z, r);

            // Update search direction: p = z + beta * p
            T beta = rs_new / rs_old;
            gpu::array::axpy(p, beta, z); // p = z + beta * p
            rs_old = rs_new;

            T res = std::sqrt(gpu::array::dot(r, r));
            std::cout << std::format(
                "\t CG iter {:3d}: residual = {:.6e}, dx = {:.6e}\n", iter + 1, res,
                dx);
            if (res < tol) break;
            if (dx < xtol) {
                std::cout << "CG converged based on solution change\n";
                break;
            }
        }
        return x;
    }

    // Template instantiations
    template DeviceArray<float> cgsolver(const gpuFunction<float> &A,
                                         const DeviceArray<float> &y,
                                         const DeviceArray<float> &x0,
                                         size_t max_iter, float tol, float xtol);
    template DeviceArray<double> cgsolver(const gpuFunction<double> &A,
                                          const DeviceArray<double> &y,
                                          const DeviceArray<double> &x0,
                                          size_t max_iter, double tol, double xtol);

} // namespace tomocam::gpu::opt
