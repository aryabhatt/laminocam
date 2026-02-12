/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 *National Laboratory (subject to receipt of any required approvals from the
 *U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 *IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 *the U.S. Government has been granted for itself and others acting on its
 *behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 *to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#ifndef TOMOCAM_OPTIMIZE_H
#define TOMOCAM_OPTIMIZE_H

#include <cmath>
#include <iostream>

#include "array.h"

namespace tomocam::opt {

    template <typename T>
    using Function = std::function<Array<T>(const Array<T> &)>;

    template <typename T>
    using Residual = std::function<T(const Array<T> &)>;

    enum class Optimizer { SPLIT_BREGMAN, CG_SOLVER, NAG_OPT };

    template <typename T>
    struct OptimizerConfig {
        Optimizer method;
        // split bregman parameters
        T lambda;
        T mu;
        // NAG + QGMRF parameters
        T lipschitz;
        T sigma;
        T p;

        // convergence parameters
        T tol;
        T xtol;
        size_t outer_max;
        size_t inner_max;

        OptimizerConfig()
            : method(Optimizer::SPLIT_BREGMAN), lambda(0.1), mu(5.0), outer_max(50),
              inner_max(4), tol(1e-5), xtol(1e-5), lipschitz(1.0), sigma(1.e+3),
              p(1.2) {}
        OptimizerConfig(Optimizer meth, T lam, T m, size_t out_max, size_t in_max,
                        T tol1, T tol2, T L, T sig, T param)
            : method(meth), lambda(lam), mu(m), outer_max(out_max),
              inner_max(in_max), tol(tol1), xtol(tol2), lipschitz(L), sigma(sig),
              p(param) {}
    };

    /** Split Bregman method for L1-regularized optimization problems
     * @brief Split Bregman method for L1-regularized optimization problems
     * @param A Function representing the system matrix
     * @param y Right-hand side vector
     * @param x0 Initial guess for the solution
     * @param lambda Regularization parameter
     * @param mu Bregman parameter
     * @param outer_max Maximum number of outer iterations
     * @param inner_max Maximum number of inner iterations
     * @param tol Tolerance for convergence based on residual norm
     * @param xtol Tolerance for convergence based on solution change
     * @return Reconstructed solution vector
     */
    template <typename T>
    Array<T> split_bregman(const Function<T> &A, const Array<T> &y,
                           const Array<T> &x0, T lambda, T mu, size_t outer_max,
                           size_t inner_max, T tol, T xtol);

    /**
     * @brief Conjugate Gradient Solver for linear systems
     * @param A Function representing the system matrix
     * @param y Right-hand side vector
     * @param x Initial guess for the solution
     * @param max_iter Maximum number of iterations
     * @param tol Tolerance for convergence
     * @return Approximate solution vector
     */
    template <typename T>
    Array<T> cgsolver(const Function<T> &A, const Array<T> &y, const Array<T> &x,
                      size_t max_iter, T tol);

    /**
     * @brief Nesterov's Optimal Gradient Method with Boyd's momentum term
     * @param grad Gradient function
     * @param loss Loss function
     * @param x Initial solution
     * @param max_iters Maximum number of iterations
     * @param lipschitz Lipschitz constant of the gradient
     * @param tol Tolerance for convergence based on loss change
     * @param xtol Tolerance for convergence based on solution change
     * @param max_inner_iters Maximum number of inner iterations for line search
     * @return Optimized solution
     */
    template <typename T>
    Array<T> nagopt(const Function<T> &grad, const Residual<T> &loss, Array<T> &x,
                    size_t max_iters, T lipschitz, T tol, T xtol,
                    size_t max_inner_iters = 20);

    /**
     * @brief Estimate the Lipschitz constant of a gradient function using the power
     * iteration method
     * @param function representing the system matrix
     * @param x0 reference for the size of the input
     * @param max_iters Maximum number of iterations (default: 20)
     * @param tol Tolerance for convergence (default: 1e-5)
     * @return Estimated Lipschitz constant
     */
    template <typename T>
    T lipschitz(const Function<T> &grad, const Array<T> &x0, size_t max_iters = 20,
                T tol = 1e-5);

    /**
     * @brief Compute the gradient of the q-generalized Gaussian Markov Random Field
     * (qGGMRF) prior
     * @param x Input array
     * @param g Output gradient array (updated in place)
     * @param sigma Scale parameter
     * @param p Exponent parameter
     */
    template <typename T>
    void qggmrf(const Array<T> &x, Array<T> &g, T sigma, T p);

} // namespace tomocam::opt

#endif // TOMOCAM_OPTIMIZE__H
