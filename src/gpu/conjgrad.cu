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

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/gpu_opt.h"
#include "gpu/utils.h"

#include "gpu/sysmat.h"
#include "gpu/precond.h"

namespace tomocam::gpu::opt {

    // -------------------------------------------------------------------------
    // GPU Conjugate Gradient Solver
    // Same preconditioned CG algorithm as src/conjgrad.cpp, with GPU-native
    // DeviceArray types and Thrust/cuFFT inner operations.
    // -------------------------------------------------------------------------
    template <typename T>
    DeviceArray<T> cgsolver(const gpuFunction<T> &A, const DeviceArray<T> &y,
                            const DeviceArray<T> &x0, size_t max_iter, T tol) {

        // Initialize solution and residual arrays
        DeviceArray<T> x = x0.clone();
        DeviceArray<T> r(x0.dims());

        // Create preconditioner instance 
        auto precond = gpuRampPreconditioner<T>(x0.dims());

        // r = y - Ad(x)
        DeviceArray<T> tmp = A(x);
        array::subtract_into(r, y, tmp);

        // z = M^{-1} r,  p = z,  rs_old = z^T r
        DeviceArray<T> z = precond.apply(r);
        DeviceArray<T> p = z.clone();
        T rs_old = array::dot(z, r);

        for (size_t iter = 0; iter < max_iter; iter++) {

            // Ap = A(p),  pAp = p^T Ap
            DeviceArray<T> Ap = A(p);
            T pAp = array::dot(p, Ap);
            if (std::abs(pAp) < T(1.e-10)) {
                std::cerr << "pAp is close to zero\n";
                break;
            }

            T alpha = rs_old / pAp;
            array::axpy(x, alpha, p);   // x += alpha * p
            array::axpy(r, -alpha, Ap); // r -= alpha * Ap

            // Apply preconditioner and compute new residual norm
            T rs_new = 0;
            z = precond.apply(r);
            rs_new = array::dot(z, r);
            std::cout << std::format("\tCG iter: {}, residual: {}\n", iter,
                                     std::sqrt(rs_new));
            if (std::sqrt(rs_new) < tol) break;

            // Update search direction: p = z + beta * p
            T beta = rs_new / rs_old;
            array::xpay(p, z, beta);
            rs_old = rs_new;
        }
        return x;
    }

    // Template instantiations
    template DeviceArray<float> cgsolver(const gpuFunction<float> &A,
                                         const DeviceArray<float> &y,
                                         const DeviceArray<float> &x0,
                                         size_t max_iter, float tol);
    template DeviceArray<double> cgsolver(const gpuFunction<double> &A,
                                          const DeviceArray<double> &y,
                                          const DeviceArray<double> &x0,
                                          size_t max_iter, double tol);

} // namespace tomocam::gpu::opt
