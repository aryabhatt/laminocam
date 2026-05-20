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

#include <format>

#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/optimize.h"

namespace tomocam::gpu::opt {

    constexpr double EPSILON = 1e-8;

    template <typename T>
    DeviceArray<T> split_bregman(gpuFunction<T> A, const DeviceArray<T> &yT,
                                 const DeviceArray<T> &x0, T lambda, T mu,
                                 size_t outer_max, size_t inner_max, T tol, T xtol) {

        DeviceArray<T> x = x0.clone();
        DeviceArray<T> x_old = x0.clone();

        // update A^TA to add laplacian of x
        // Ap  = (A^TA  +   ∇^T∇) u
        gpuFunction<T> Ap = [&A, &mu](const Array<T> &u) {
            auto Au = A(u);
            auto lap_u = laplacian(u);
            array::xpay(Au, mu, lap_u);
            return Au;
        };

        for (int iter = 0; iter < outer_max; ++iter) {

            // x-update: solve (A^TA + μ∇^T∇)x = A^T y + μ∇^T(d - b)
            std::array<DeviceArray<T>, 3> d_b;
            for (size_t i = 0; i < 3; ++i) {
                array::subtract_into(d_b[i], d[i], b[i]);
            }
            auto rhs = array::subtract(yT, array::scale(divergence(d_b), mu));

            // use conjugate gradient to solve the linear system
            x = cgsolver(Ap, rhs, x, inner_max, tol);

            // isotropic TV shrinkage
            auto dx = grad_u(x);
            auto sk = DeviceArray<T>(x.dims());
            for (size_t i = 0; i < 3; ++i) {
                array::add_into(dx[i], dx[i], b[i]);
            }
            auto sk = array::sqrt(sk);

            // update d inplace
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < dx[i].size(); ++j) {
                    T val = (dx[i][j] + b[i][j]) / (sk[j] + EPSILON);
                    d[i][j] = std::max(T(0), sk[j] - lambda / mu) * val;
                }
            }
            // Bregman update
            for (size_t i = 0; i < 3; ++i) { b[i] += dx[i] - d[i]; }

            // Check convergence
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
    split_bregman(gpuFunction<float> A, const DeviceArray<float> &yT,
                  const DeviceArray<float> &x0, float lambda, float mu,
                  size_t outer_max, size_t inner_max, float tol, float xtol);
    template DeviceArray<double>
    split_bregman(gpuFunction<double> A, const DeviceArray<double> &yT,
                  const DeviceArray<double> &x0, double lambda, double mu,
                  size_t outer_max, size_t inner_max, double tol, double xtol);
} // namespace tomocam::gpu::opt
