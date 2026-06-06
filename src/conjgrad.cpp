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

#include <execution>
#include <format>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "array.h"
#include "array_ops.h"
#include "optimize.h"
#include "precond.h"

namespace tomocam::opt {
    template <typename T>
    Array<T> cgsolver(const Function<T> &A, const Array<T> &yT, const Array<T> &x0,
                      size_t max_iter, T tol, T xtol) {

        // initialize
        auto x = x0.clone();
        auto r = yT - A(x);

        // diagnoal preconditioner
        auto ones = Array<T>::ones(x.dims());
        auto pre = A(ones);
        for (size_t i = 0; i < pre.size(); i++) {
            if (std::abs(pre[i]) < 1.2E-07) {
                std::cerr << "preconditioner is close to zero\n";
                pre[i] = 1.0E-06;
            }
        }
        auto precond_apply = [&pre](const Array<T> &r) { return r / pre; };
        // auto precond_apply = [](const Array<T> &r) { return r.clone(); };

        auto z = precond_apply(r);
        auto p = z.clone();
        auto rs_old = array::dot(z, r);

        for (size_t iter = 0; iter < max_iter; iter++) {

            auto Ap = A(p);
            auto pAp = array::dot(p, Ap);
            auto pAp_thresh = T(100) * std::numeric_limits<T>::epsilon() *
                              array::dot(p, p) * array::dot(Ap, Ap);
            if (std::abs(pAp) < std::sqrt(pAp_thresh)) {
                std::cerr << std::format("pAp is close to zero: {}, stopping CG\n",
                                         pAp);
                break;
            }
            auto alpha = rs_old / pAp;
            auto step = p * alpha;
            x += step;
            r -= Ap * alpha;

            z = precond_apply(r);
            auto rs_new = array::dot(z, r);
            p = z + (p * (rs_new / rs_old));
            rs_old = rs_new;

            // calculate and print the residual and solution change
            auto res = std::sqrt(array::dot(r, r));
            auto dx = array::norm2(step) / (array::norm2(x) + T(1e-8));
            std::cout << std::format("\t CG Iter: {}, residual: {}, dx: {:.6e}\n",
                                     iter, res, dx);
            if (std::sqrt(res) < tol) { break; }
            if (dx < xtol) { break; }
        }
        return x;
    }

    // template instantiations
    template Array<float> cgsolver<float>(const Function<float> &,
                                          const Array<float> &, const Array<float> &,
                                          size_t, float, float);
    template Array<double> cgsolver<double>(const Function<double> &,
                                            const Array<double> &,
                                            const Array<double> &, size_t, double,
                                            double);

} // namespace tomocam::opt
