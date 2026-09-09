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

// Compare the two sysmat2d overloads defined in gradient.cpp for the 2D
// slice-by-slice (vertical rotation axis) tomography path:
//   sysmat2d(x, PolarGrid2D)             - direct NUFFT path
//   sysmat2d(x, PointSpreadFunction2D)   - Toeplitz (broadcast PSF convolution) path
//
// Both compute A^T A x for the same geometry; they should agree to within a
// tolerance driven by NUFFT approximation error. Also checks that the
// broadcast convolution treats every slice independently by comparing a
// slice extracted from a multi-slice run against a single-slice run.

#include <cmath>
#include <format>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "polar_grid2d.h"
#include "toeplitz2d.h"
#include "tomocam.h"

using namespace tomocam;

// Returns cosine similarity: <a,b> / (||a|| ||b||).  1.0 = identical direction.
static double cosine_sim(const Array<double> &a, const Array<double> &b) {
    double na = array::norm2(a), nb = array::norm2(b);
    return (na > 0 && nb > 0) ? array::dot(a, b) / (na * nb) : 0.0;
}

static std::vector<double> make_theta(size_t nangles) {
    std::vector<double> theta(nangles);
    for (size_t i = 0; i < nangles; ++i)
        theta[i] = (static_cast<double>(i) - nangles / 2.0) * M_PI / nangles;
    return theta;
}

static bool test_sysmat2d_agreement(const dims_t &vol_dims, size_t nangles,
                                    double tol) {
    std::cout << std::format(
        "vol({},{},{}) nangles={:<3}  sysmat2d NUFFT vs Toeplitz\n", vol_dims.n1,
        vol_dims.n2, vol_dims.n3, nangles);

    auto theta = make_theta(nangles);
    cpu::PolarGrid2D<double> pg(theta, vol_dims.n3);
    cpu::PointSpreadFunction2D<double> psf(pg, vol_dims.n3);

    auto x = Array<double>::random(vol_dims);

    auto nufft_result = sysmat2d(x, pg);
    auto toeplitz_result = sysmat2d(x, psf);

    double n_nufft = array::norm2(nufft_result);
    double n_toeplitz = array::norm2(toeplitz_result);

    double scale_ratio =
        array::dot(nufft_result, toeplitz_result) / (n_toeplitz * n_toeplitz);
    double cos_sim = cosine_sim(nufft_result, toeplitz_result);

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

// Verify the broadcast convolution treats each slice independently: extract
// slice i from a multi-slice Toeplitz run and compare to a fresh, single-
// slice (nslices=1) run built from the same input slice.
static bool test_broadcast_consistency(const dims_t &vol_dims, size_t nangles,
                                       double tol) {
    std::cout << std::format(
        "vol({},{},{}) nangles={:<3}  broadcast-across-slices consistency\n",
        vol_dims.n1, vol_dims.n2, vol_dims.n3, nangles);

    auto theta = make_theta(nangles);
    cpu::PolarGrid2D<double> pg(theta, vol_dims.n3);
    cpu::PointSpreadFunction2D<double> psf(pg, vol_dims.n3);

    auto x = Array<double>::random(vol_dims);
    auto batched = sysmat2d(x, psf);

    bool ok = true;
    for (size_t i = 0; i < vol_dims.n1; i += std::max<size_t>(1, vol_dims.n1 / 2)) {
        // build a single-slice input equal to slice i of x
        dims_t single_dims{1, vol_dims.n2, vol_dims.n3};
        Array<double> single(single_dims);
        std::copy(x.begin() + i * vol_dims.n2 * vol_dims.n3,
                  x.begin() + (i + 1) * vol_dims.n2 * vol_dims.n3, single.begin());

        auto single_result = sysmat2d(single, psf);

        double diff = 0.0, norm = 0.0;
        for (size_t j = 0; j < single_dims.n2 * single_dims.n3; ++j) {
            double a = batched[i * vol_dims.n2 * vol_dims.n3 + j];
            double b = single_result[j];
            diff += (a - b) * (a - b);
            norm += a * a;
        }
        double rel = (norm > 0) ? std::sqrt(diff / norm) : 0.0;
        bool slice_ok = rel < tol;
        ok = ok && slice_ok;
        std::cout << std::format("  slice {}: rel_diff={:.2e} {}\n", i, rel,
                                 slice_ok ? "OK" : "FAIL");
    }
    return ok;
}

int main() {
    std::cout << "\n====== sysmat2d: NUFFT vs Toeplitz agreement tests ======\n\n";

    const double tol = 1e-4;
    int passed = 0, failed = 0;
    auto record = [&](bool ok) { ok ? ++passed : ++failed; };

    // Small volume, few angles
    {
        const dims_t vol_dims{5, 16, 16};
        const size_t nangles = 7;
        record(test_sysmat2d_agreement(vol_dims, nangles, tol));
        record(test_broadcast_consistency(vol_dims, nangles, tol));
    }

    // Larger volume, more angles
    {
        const dims_t vol_dims{8, 32, 32};
        const size_t nangles = 13;
        record(test_sysmat2d_agreement(vol_dims, nangles, tol));
        record(test_broadcast_consistency(vol_dims, nangles, tol));
    }

    std::cout << "\nResults: " << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
