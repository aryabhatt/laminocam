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
#include <iostream>

#include "array.h"
#include "array_ops.h"
#include "optimize.h"

// Power iteration to estimate the Lipschitz constant of A
namespace tomocam::opt {
    template <typename T>
    T lipschitz(const Function<T> &A, const Array<T> &ref, size_t max_iter, T tol) {
        Array<T> x = Array<T>::random(ref.dims());
        x = x / array::norm2(x);
        T L = 0;
        for (int i = 0; i < max_iter; ++i) {
            auto Ax = A(x);
            T norm_Ax = array::norm2(Ax);
            // avoid division by almost zero
            if (std::abs(norm_Ax) < 1e-12) {
                L = 0;
                break;
            }
            x = Ax / norm_Ax;
            T L_new = norm_Ax;
#ifdef DEBUG
            std::cout << std::format("Iteration {}: Lipschitz estimate = {:.6e}\n",
                                     i + 1, L_new);
#endif
            if (std::abs(L_new - L) < tol) { break; }
            L = L_new;
        }
        return L;
    }
    // explicit template instantiation
    template float lipschitz(const Function<float> &, const Array<float> &, size_t,
                             float);
    template double lipschitz(const Function<double> &, const Array<double> &,
                              size_t, double);
} // namespace tomocam::opt
