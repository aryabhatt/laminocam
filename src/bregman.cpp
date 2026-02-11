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

#include "bregman.h"
#include "array.h"
#include "array_ops.h"
#include "optimize.h"

namespace tomocam::opt {

    constexpr double EPSILON = 1e-8;

    /* Solve the optimization problem:
     *   min_x 0.5 ||A x - y||^2 + lambda TV(x)
     * using split Bregman method
     */
    template <typename T>
    Array<T> split_bregman(const std::function<Array<T>(const Array<T> &)> &A,
                           const Array<T> &yT, const Array<T> &x0, T lambda, T mu,
                           size_t outer_max, size_t inner_max, T tol, T xtol) {

        Array<T> x = x0.clone();
        Array<T> x_old = x0.clone();
        std::array<Array<T>, 3> d;
        std::array<Array<T>, 3> b;
        for (int i = 0; i < 3; ++i) {
            d[i] = Array<T>::zeros_like(x);
            b[i] = Array<T>::zeros_like(x);
        }

        // update A^TA to add laplacian of x
        // Ap  = (A^TA  +   ∇^T∇) u
        Function<T> Ap = [&](const Array<T> &u) { return A(u) - laplacian(u) * mu; };

        for (int iter = 0; iter < outer_max; ++iter) {

            // x-update: solve (A^TA + μ∇^T∇)x = A^T y + μ∇^T(d - b)
            std::array<Array<T>, 3> d_b;
            for (size_t i = 0; i < 3; ++i) { d_b[i] = d[i] - b[i]; }
            auto rhs = yT - divergence(d_b) * mu;

            // use conjugate gradient to solve the linear system
            x = cgsolver(Ap, rhs, x, inner_max, tol);

            // isotropic TV shrinkage
            auto dx = grad_u(x);
            auto sk = Array<T>::zeros_like(x);
            for (size_t i = 0; i < x.size(); ++i) {
                T sum_sq = 0;
                for (size_t j = 0; j < 3; ++j) {
                    T val = dx[j][i] + b[j][i];
                    sum_sq += val * val;
                }
                sk[i] = std::sqrt(sum_sq);
            }

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
            x_old = x.clone();
            if (norm_diff < xtol) { break; }
        }
        return x;
    }

    // explicit template instantiations for float and double
    template Array<float>
    split_bregman<float>(const std::function<Array<float>(const Array<float> &)> &,
                         const Array<float> &, const Array<float> &, float, float,
                         size_t, size_t, float, float);
    template Array<double> split_bregman<double>(
        const std::function<Array<double>(const Array<double> &)> &,
        const Array<double> &, const Array<double> &, double, double, size_t, size_t,
        double, double);
} // namespace tomocam::opt
