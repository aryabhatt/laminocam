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

#include <concepts>
#include <format>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include "gpu/bregman.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/finitediff.h"
#include "gpu/gpu_opt.h"
#include "gpu/mem_check.h"

namespace tomocam::gpu::opt {

    constexpr double EPSILON = 1e-8;

    // isotropic TV shrinkage functor
    template <typename T>
    struct ShrinkFn {
        T lambda;
        T mu;
        ShrinkFn(T lambda, T mu) : lambda(lambda), mu(mu) {}

        template <typename Tuple>
        __host__ __device__ T operator()(const Tuple &t) const {
            using namespace cuda::std;
            T dx = thrust::get<0>(t);
            T b = thrust::get<1>(t);
            T sk = thrust::get<2>(t);
            T val = (dx + b) / (sk + EPSILON);
            return max(T(0), sk - lambda / mu) * val;
        }
    };

    template <typename T>
    DeviceArray<T> shrink(const DeviceArray<T> &dx, const DeviceArray<T> &b,
                          const DeviceArray<T> &sk, T lambda, T mu) {
        DeviceArray<T> d(b.dims());
        auto zip = thrust::zip_iterator(
            thrust::make_tuple(dx.begin(), b.begin(), sk.begin()));
        thrust::transform(zip, zip + dx.size(), d.begin(), ShrinkFn(lambda, mu));
        return d;
    }

    /// compute sk = sqrt(∑(dx[i] + b[i])^2) for i=1,2,3
    template <typename T>
    struct AddSquareFn {
        __host__ __device__ T operator()(T dx, T b) const {
            return (dx + b) * (dx + b);
        }
    };
    template <typename T>
    DeviceArray<T> compute_sk(const std::array<DeviceArray<T>, 3> &dx,
                              const std::array<DeviceArray<T>, 3> &b) {
        DeviceArray<T> sk(dx[0].dims());
        for (size_t i = 0; i < 3; ++i) {
            DeviceArray<T> temp(dx[i].dims());
            thrust::transform(dx[i].begin(), dx[i].begin() + dx[i].size(),
                              b[i].begin(), temp.begin(), AddSquareFn<T>());
            sk += temp;
        }

        return array::sqrt(sk);
    }

    /// Split Bregman method for TV-regularized least squares problems
    template <typename T>
    DeviceArray<T> split_bregman(const gpuFunction<T> &A, const DeviceArray<T> &yT,
                                 const DeviceArray<T> &x0, T lambda, T mu,
                                 size_t outer_max, size_t inner_max, T tol, T xtol) {

        DeviceArray<T> x = x0.clone();
        DeviceArray<T> x_old = x0.clone();

        // bregman variables
        std::array<DeviceArray<T>, 3> b;
        std::array<DeviceArray<T>, 3> d;
        for (size_t i = 0; i < 3; ++i) {
            b[i] = DeviceArray<T>(x.dims());
            d[i] = DeviceArray<T>(x.dims());
        }
        // Checkpoint: x + x_old + b[0..2] + d[0..2] = 8 N
        MEM_CHECK("sb: persistent (x, x_old, b[3], d[3] = 8N)",
                  8 * x.size() * sizeof(T));

        // update A^TA to add laplacian of x
        // Ap  = (A^TA - ∇^T∇) u
        gpuFunction<T> Ap = [&A, &mu](const DeviceArray<T> &u) {
            auto Au = A(u);
            auto lap_u = neg_laplacian(u);
            array::xpay(Au, lap_u, mu);
            return Au;
        };

        for (int iter = 0; iter < outer_max; ++iter) {

            // x-update: solve (A^TA + μ∇^T∇)x = A^T y + μ∇^T(d - b)
            std::array<DeviceArray<T>, 3> d_b;
            for (size_t i = 0; i < 3; ++i) { d_b[i] = d[i] - b[i]; }

            // compute right-hand side: A^T y + μ∇^T(d - b)
            auto rhs = yT.clone();
            array::xpay(rhs, neg_divergence(d_b), mu);
            // Checkpoint: 8N persistent + d_b[3] (3N) + rhs (1N) = 12N
            // (neg_divergence transient +2N has already resolved)
            MEM_CHECK("sb: outer workspace resolved (8N + d_b[3] + rhs = 12N)",
                      12 * x.size() * sizeof(T));

            // use conjugate gradient to solve the linear system
            // Checkpoint: entering cgsolver, expect +6N inside (CG persistent)
            MEM_CHECK("sb: entering cgsolver (12N + ~6N CG persistent expected)",
                      18 * x.size() * sizeof(T));
            x = cgsolver(Ap, rhs, x, inner_max, tol, xtol);

            // isotropic TV shrinkage
            auto dx = grad(x);

            // sk = sqrt(∑(dx[i] + b[i])^2)
            auto sk = compute_sk(dx, b);
            // Checkpoint: after CG, grad, sk. Live: 8N persistent + dx[3] (3N) + sk (1N) = 12N
            MEM_CHECK("sb: post-cgsolver, after grad+sk (8N + dx[3] + sk = 12N)",
                      12 * x.size() * sizeof(T));

            // update d inplace
            // d[i] = max(0, sk - lambda / mu) * (dx[i] + b[i]) / (sk + EPSILON)
            for (size_t i = 0; i < 3; ++i) {
                d[i] = shrink(dx[i], b[i], sk, lambda, mu);
            }

            // Bregman update
            // b[i] = b[i] + (dx[i] - d[i])
            for (size_t i = 0; i < 3; ++i) { b[i] += dx[i] - d[i]; }

            // Check convergence
            // norm_diff = ‖xᵏ⁺¹ − xᵏ‖₂ / ‖xᵏ‖₂
            T norm_diff = array::norm2(x - x_old) / (array::norm2(x_old) + EPSILON);
            std::cout << std::format(
                "Outer iter: {}, ‖xᵏ⁺¹ − xᵏ‖₂ / ‖xᵏ‖₂: {:.6e}\n", iter, norm_diff);
            x_old = x.clone();
            if (norm_diff < xtol) { break; }
        }
        return x;
    }

    // explicit template instantiations for float and double
    template DeviceArray<float>
    split_bregman(const gpuFunction<float> &A, const DeviceArray<float> &yT,
                  const DeviceArray<float> &x0, float lambda, float mu,
                  size_t outer_max, size_t inner_max, float tol, float xtol);
    template DeviceArray<double>
    split_bregman(const gpuFunction<double> &A, const DeviceArray<double> &yT,
                  const DeviceArray<double> &x0, double lambda, double mu,
                  size_t outer_max, size_t inner_max, double tol, double xtol);

    template DeviceArray<float>
    compute_sk(const std::array<DeviceArray<float>, 3> &,
               const std::array<DeviceArray<float>, 3> &);
    template DeviceArray<double>
    compute_sk(const std::array<DeviceArray<double>, 3> &,
               const std::array<DeviceArray<double>, 3> &);

    template DeviceArray<float> shrink(const DeviceArray<float> &,
                                       const DeviceArray<float> &,
                                       const DeviceArray<float> &, float, float);
    template DeviceArray<double> shrink(const DeviceArray<double> &,
                                        const DeviceArray<double> &,
                                        const DeviceArray<double> &, double, double);
} // namespace tomocam::gpu::opt
