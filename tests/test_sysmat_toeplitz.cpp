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

// Compare the two sysmat overloads defined in gradient.cpp:
//   sysmat(x, PolarGrid)           — direct NUFFT path
//   sysmat(x, PointSpreadFunction) — Toeplitz (PSF convolution) path
//
// Both compute A^T A x for the same geometry; they should agree to within
// a tolerance driven by NUFFT approximation error.

#include <cmath>
#include <format>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "polar_grid.h"
#include "toeplitz.h"
#include "tomocam.h"

using namespace tomocam;

// Returns cosine similarity: <a,b> / (||a|| ||b||).  1.0 = identical direction.
static double cosine_sim(const Array<double> &a, const Array<double> &b) {
    double na = array::norm2(a), nb = array::norm2(b);
    return (na > 0 && nb > 0) ? array::dot(a, b) / (na * nb) : 0.0;
}

static bool test_sysmat_agreement(const dims_t &vol_dims,
                                  const std::vector<double> &theta, double gamma,
                                  double tol) {
    std::cout << std::format(
        "vol({},{},{}) nangles={:<3}  sysmat NUFFT vs Toeplitz\n", vol_dims.n1,
        vol_dims.n2, vol_dims.n3, theta.size());

    cpu::PolarGrid<double> pg(theta, std::vector<double>(theta.size(), gamma),
                              std::vector<double>(theta.size(), 0.0), vol_dims.n2,
                              vol_dims.n3);
    cpu::PointSpreadFunction<double> psf(pg, vol_dims);

    auto x = Array<double>::random(vol_dims);

    auto nufft_result = sysmat(x, pg);
    auto toeplitz_result = sysmat(x, psf);

    double n_nufft = array::norm2(nufft_result);
    double n_toeplitz = array::norm2(toeplitz_result);

    double scale_ratio =
        array::dot(nufft_result, toeplitz_result) / (n_toeplitz * n_toeplitz);
    double cos_sim = cosine_sim(nufft_result, toeplitz_result);

    // normalised relative difference: compare nufft with toeplitz / scale_ratio
    auto toeplitz_scaled = toeplitz_result * scale_ratio;
    auto diff = nufft_result - toeplitz_scaled;
    double rel_diff_norm = array::norm2(diff) / n_nufft;

    bool shape_ok = rel_diff_norm < tol;
    bool scale_ok = std::abs(scale_ratio - 1.0) < tol;
    bool ok = shape_ok && scale_ok;

    std::cout << std::format("  ||nufft||={:.4e}  ||toeplitz||={:.4e}  "
                             "scale_ratio={:.4e}  cos_sim={:.6f}\n",
                             n_nufft, n_toeplitz, scale_ratio, cos_sim);
    std::cout << std::format("  normalised rel_diff={:.2e}  shape={} scale={}\n",
                             rel_diff_norm, shape_ok ? "OK" : "FAIL",
                             scale_ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    std::cout << "\n====== sysmat: NUFFT vs Toeplitz agreement tests ======\n\n";

    const double gamma = 0.0;
    const double tol = 1e-4;
    int passed = 0, failed = 0;
    auto record = [&](bool ok) { ok ? ++passed : ++failed; };

    // Small volume, few angles
    {
        const dims_t vol_dims{5, 16, 16};
        const size_t nangles = 7;
        std::vector<double> theta(nangles);
        for (size_t i = 0; i < nangles; ++i)
            theta[i] = (static_cast<double>(i) - nangles / 2.0) * M_PI / nangles;
        record(test_sysmat_agreement(vol_dims, theta, gamma, tol));
    }

    // Larger volume, more angles
    {
        const dims_t vol_dims{8, 32, 32};
        const size_t nangles = 13;
        std::vector<double> theta(nangles);
        for (size_t i = 0; i < nangles; ++i)
            theta[i] = (static_cast<double>(i) - nangles / 2.0) * M_PI / nangles;
        record(test_sysmat_agreement(vol_dims, theta, gamma, tol));
    }

    // Non-zero gamma
    {
        const dims_t vol_dims{5, 16, 16};
        const size_t nangles = 9;
        const double gam = 0.3;
        std::vector<double> theta(nangles);
        for (size_t i = 0; i < nangles; ++i)
            theta[i] = (static_cast<double>(i) - nangles / 2.0) * M_PI / nangles;
        record(test_sysmat_agreement(vol_dims, theta, gam, tol));
    }

    std::cout << "\nResults: " << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
