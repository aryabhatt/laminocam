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

#include "array.h"
#include "array_ops.h"
#include "optimize.h"

namespace tomocam::opt {
    template <typename T>
    Array<T> cgsolver(const Function<T> &A, const Array<T> &yT, const Array<T> &x0,
                      size_t max_iter, T tol) {

        // initialize
        auto x = x0.clone();
        auto r = yT - A(x);
        auto p = r.clone();
        auto rs_old = array::dot(r, r);

        for (size_t iter = 0; iter < max_iter; iter++) {

            auto Ap = A(p);
            auto pAp = array::dot(p, Ap);
            if (std::abs(pAp) < 1.e-15) {
                std::cerr << "pAp is close to zero\n";
                break;
            }
            auto alpha = rs_old / pAp;
            x += p * alpha;
            r -= Ap * alpha;

            auto rs_new = array::dot(r, r);
            std::cout << std::format("iter: {}, residual: {}\n", iter,
                                     std::sqrt(rs_new));
            if (rs_new < tol * tol) { break; }

            p = r + (p * (rs_new / rs_old));
            rs_old = rs_new;
        }
        return x;
    }

    // template instantiations
    template Array<float> cgsolver<float>(const Function<float> &,
                                          const Array<float> &, const Array<float> &,
                                          size_t, float);
    template Array<double> cgsolver<double>(const Function<double> &,
                                            const Array<double> &,
                                            const Array<double> &, size_t, double);

} // namespace tomocam::opt
