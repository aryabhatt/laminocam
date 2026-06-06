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

    enum class Optimizer { SPLIT_BREGMAN, CG_SOLVER };

    template <typename T>
    struct OptimizerConfig {
        Optimizer method;
        // split bregman parameters
        T lambda;
        T mu;

        // convergence parameters
        T tol;
        T xtol;
        size_t outer_max;
        size_t inner_max;

        OptimizerConfig()
            : method(Optimizer::SPLIT_BREGMAN), lambda(0.1), mu(5.0), outer_max(50),
              inner_max(4), tol(1e-5), xtol(1e-5) {}
        OptimizerConfig(Optimizer meth, T lam, T m, size_t out_max, size_t in_max,
                        T tol1, T tol2)
            : method(meth), lambda(lam), mu(m), outer_max(out_max),
              inner_max(in_max), tol(tol1), xtol(tol2) {}
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
     * @param tol Tolerance for convergence based on residual norm
     * @param xtol Tolerance for convergence based on solution change
     * @return Approximate solution vector
     */
    template <typename T>
    Array<T> cgsolver(const Function<T> &A, const Array<T> &y, const Array<T> &x,
                      size_t max_iter, T tol, T xtol);

} // namespace tomocam::opt

#endif // TOMOCAM_OPTIMIZE__H
