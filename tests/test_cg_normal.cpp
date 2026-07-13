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

// Test cgsolver via the normal equations A^T A x = A^T c, where A is a
// random sparse matrix and c is a random column vector.  The true solution
// x_true is constructed first so the solver result can be compared directly.

#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "optimize.h"

using namespace tomocam;

template <typename T>
static T rel_err(const Array<T> &a, const Array<T> &b) {
    auto diff = a - b;
    T ref = array::norm2(b);
    return array::norm2(diff) / (ref + T(1e-30));
}

// Build a random sparse matrix (COO), a known solution x_true, RHS b = A^T(A
// x_true), and verify that cgsolver recovers x_true to within tol.
template <typename T>
static bool test_cg_normal(T tol) {
    std::cout << (sizeof(T) == 8 ? "double" : "float ")
              << "  cgsolver via A^T A x = A^T c  ... \n";

    constexpr size_t m = 64; // rows of A
    constexpr size_t n = 32; // cols of A  (overdetermined: m > n)

    // Build COO sparse matrix with ~30 % fill using a fixed seed
    std::mt19937 rng(42);
    std::uniform_real_distribution<T> val_dist(T(-1), T(1));
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    using Triplet = std::tuple<int, int, T>;
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<size_t>(m * n * 0.35));
    for (int r = 0; r < static_cast<int>(m); ++r)
        for (int c = 0; c < static_cast<int>(n); ++c)
            if (prob_dist(rng) < 0.30) triplets.emplace_back(r, c, val_dist(rng));

    // x_true: known solution
    auto x_true = Array<T>::zeros(dims_t{1, 1, n});
    for (size_t i = 0; i < n; ++i) x_true[i] = val_dist(rng);

    // c = A x_true
    auto c = Array<T>::zeros(dims_t{1, 1, m});
    for (auto &[r, col, val] : triplets)
        c[static_cast<size_t>(r)] += val * x_true[static_cast<size_t>(col)];

    // b = A^T c
    auto b = Array<T>::zeros(dims_t{1, 1, n});
    for (auto &[r, col, val] : triplets)
        b[static_cast<size_t>(col)] += val * c[static_cast<size_t>(r)];

    // CG operator: v -> A^T (A v)
    opt::Function<T> F = [&](const Array<T> &v) {
        auto y = Array<T>::zeros(dims_t{1, 1, m});
        for (auto &[r, col, val] : triplets)
            y[static_cast<size_t>(r)] += val * v[static_cast<size_t>(col)];

        auto z = Array<T>::zeros(dims_t{1, 1, n});
        for (auto &[r, col, val] : triplets)
            z[static_cast<size_t>(col)] += val * y[static_cast<size_t>(r)];

        return z;
    };

    auto x0 = Array<T>::zeros(dims_t{1, 1, n});
    auto x_sol = opt::cgsolver(F, b, x0, 2000, tol, tol / T(100));

    T err = rel_err(x_sol, x_true);
    bool ok = err < tol * T(100);
    std::cout << (ok ? "PASSED" : "FAILED") << "  rel_err=" << err << "\n";
    return ok;
}

int main() {
    std::cout << "\n====== CG Normal-Equations Tests ======\n\n";

    int passed = 0, failed = 0;
    auto record = [&](bool ok) { ok ? ++passed : ++failed; };

    record(test_cg_normal<double>(1e-5));
    record(test_cg_normal<float>(1e-3f));

    std::cout << "\nResults: " << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
