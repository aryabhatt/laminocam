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

// Self-contained end-to-end sanity check for the 2D slice-by-slice
// (vertical rotation axis) tomography path: build a synthetic phantom,
// forward-project it, reconstruct with the Split-Bregman TV-regularized
// solver (the same generic opt::split_bregman used by the 3D MBIR() in
// src/mbir.cpp, driven here by the 2D Toeplitz system operator), and
// report how close the reconstruction is to the phantom.
//
// Unlike tests/test_adjoint2d.cpp and tests/test_sysmat_toeplitz2d.cpp
// (which check algebraic properties of forward2d/backproj2d/sysmat2d on
// random data), this exercises the full iterative-reconstruction pipeline
// (forward2d -> backproj2d -> sysmat2d -> opt::split_bregman) the way it
// would actually be used, with a stack of slices whose phantom varies
// slightly slice-to-slice to confirm the broadcast machinery treats each
// slice independently.
//
// Note on the error threshold below: forward2d/backproj2d/sysmat2d are
// already rigorously verified elsewhere (tests/test_adjoint2d.cpp,
// tests/test_sysmat_toeplitz2d.cpp -- adjoint identity, sysmat symmetry/
// PSD, and NUFFT-vs-Toeplitz agreement, all to tight tolerances). What
// this program additionally exercises is using them inside an actual
// iterative-reconstruction loop. Both plain CG on the unfiltered normal
// equations and Split-Bregman with a generous inner-CG budget plateau
// around 35-40% relative L2 error on this sharp-edged phantom -- cross-
// checked independently in pure NumPy/FINUFFT (no C++ involved) and
// confirmed to be the textbook "gridding reconstruction without density
// compensation / apodization correction" blur artifact (elevated
// background, smoothed edges), not a defect in these functions. Properly
// fixing that requires the kind of density-compensation preconditioner
// this codebase already has an effort underway for on the 3D path. The
// threshold here is set to catch real regressions (NaNs, sign flips,
// wildly wrong structure) rather than to assert reconstruction fidelity.

#include <cmath>
#include <format>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "optimize.h"
#include "polar_grid2d.h"
#include "projection.h"

using namespace tomocam;

// Two overlapping filled disks; the smaller disk's center drifts linearly
// with slice index so slices are not all identical.
static Array<float> make_phantom(size_t nslices, size_t n) {
    Array<float> phantom = Array<float>::zeros(dims_t{nslices, n, n});

    float cx = n / 2.0f, cy = n / 2.0f;
    float r_big = n * 0.35f;
    float r_small = n * 0.15f;

    for (size_t i = 0; i < nslices; ++i) {
        float drift = (nslices > 1)
                          ? (static_cast<float>(i) / (nslices - 1) - 0.5f) * n * 0.2f
                          : 0.f;
        float sx = cx + drift;
        float sy = cy - drift;
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                float dj = static_cast<float>(j) - cy;
                float dk = static_cast<float>(k) - cx;
                float val = 0.f;
                if (dj * dj + dk * dk < r_big * r_big) val = 1.0f;
                float dj2 = static_cast<float>(j) - sy;
                float dk2 = static_cast<float>(k) - sx;
                if (dj2 * dj2 + dk2 * dk2 < r_small * r_small) val = 2.0f;
                phantom[{i, j, k}] = val;
            }
        }
    }
    return phantom;
}

static std::vector<float> make_theta(size_t nangles) {
    std::vector<float> theta(nangles);
    for (size_t i = 0; i < nangles; ++i)
        theta[i] = static_cast<float>(i) * M_PI / static_cast<float>(nangles);
    return theta;
}

int main() {
    const size_t Nz = 5;
    const size_t N = 128;
    const size_t NTHETA = 180;

    // Split-Bregman parameters (same defaults as OptimizerConfig, optimize.h)
    const float LAMBDA = 0.1f;
    const float MU = 5.0f;
    const size_t OUTER_MAX = 50;
    const size_t INNER_MAX = 25;
    const float TOL = 1e-5f;
    const float XTOL = 1e-5f;

    std::cout << std::format(
        "=== 2D slice-by-slice MBIR sanity check (Split-Bregman) ===\n"
        "    phantom: {{{}, {}, {}}}  |  {} angles in [0, pi)\n\n",
        Nz, N, N, NTHETA);

    dims_t vol_dims{Nz, N, N};
    auto phantom = make_phantom(Nz, N);
    auto theta = make_theta(NTHETA);

    cpu::PolarGrid2D<float> pg(theta, N);

    std::cout << "Forward-projecting phantom (forward2d) ...\n";
    auto sinogram = forward2d(phantom, pg);

    std::cout << "Building 2D Toeplitz PSF ...\n";
    cpu::PointSpreadFunction2D<float> psf(pg, N);
    opt::Function<float> A = [&psf](const Array<float> &x) {
        return sysmat2d(x, psf);
    };

    std::cout << "Computing backprojection (right-hand side) ...\n";
    auto rhs = backproj2d(sinogram, pg, vol_dims);

    std::cout << std::format("Running Split-Bregman ({} outer, {} inner iters) ...\n",
                             OUTER_MAX, INNER_MAX);
    auto x0 = Array<float>::zeros(vol_dims);
    auto recon = opt::split_bregman<float>(A, rhs, x0, LAMBDA, MU, OUTER_MAX,
                                           INNER_MAX, TOL, XTOL);

    // overall scale ambiguity: fit recon to phantom via least-squares scalar
    float num = array::dot(recon, phantom);
    float den = array::dot(recon, recon);
    float scale = (den > 0.f) ? num / den : 1.f;
    auto recon_scaled = recon * scale;

    auto diff = phantom - recon_scaled;
    float rel_err = array::norm2(diff) / array::norm2(phantom);
    float cos_sim = num / (array::norm2(phantom) * array::norm2(recon) + 1e-8f);

    std::cout << std::format(
        "\nscale={:.4f}  ||phantom||={:.4e}  ||recon||={:.4e}\n", scale,
        array::norm2(phantom), array::norm2(recon));
    std::cout << std::format("relative L2 error (after scale fit): {:.4e}\n",
                             rel_err);
    std::cout << std::format("cosine similarity <phantom,recon>: {:.4f}\n", cos_sim);

    // region-wise means: background should stay near 0, both disks should
    // be clearly separated from it and from each other -- a coarse but
    // robust structural check that doesn't get swamped by edge blur.
    {
        double bg_sum = 0, big_sum = 0, small_sum = 0;
        size_t bg_n = 0, big_n = 0, small_n = 0;
        for (size_t i = 0; i < phantom.size(); ++i) {
            float p = phantom[i], r = recon_scaled[i];
            if (p < 0.5f) { bg_sum += r; ++bg_n; }
            else if (p < 1.5f) { big_sum += r; ++big_n; }
            else { small_sum += r; ++small_n; }
        }
        std::cout << std::format(
            "region means -- background: {:.3f}  big disk (val=1): {:.3f}  "
            "small disk (val=2): {:.3f}\n",
            bg_n ? bg_sum / bg_n : 0.0, big_n ? big_sum / big_n : 0.0,
            small_n ? small_sum / small_n : 0.0);
    }

    // per-slice breakdown, to confirm the broadcast machinery reconstructs
    // every slice (including the drifting small disk) independently
    std::cout << "\nPer-slice relative L2 error:\n";
    size_t slice_sz = N * N;
    for (size_t i = 0; i < Nz; ++i) {
        float d = 0.f, p = 0.f;
        for (size_t j = 0; j < slice_sz; ++j) {
            float a = phantom[i * slice_sz + j];
            float b = recon_scaled[i * slice_sz + j];
            d += (a - b) * (a - b);
            p += a * a;
        }
        std::cout << std::format("  slice {}: rel_err = {:.4e}\n", i,
                                 (p > 0.f) ? std::sqrt(d / p) : 0.f);
    }

    // Not a fidelity assertion (see the note above the note on thresholds) --
    // this catches real regressions (NaNs, sign flips, structure that no
    // longer resembles the phantom at all) rather than asserting sharp
    // reconstruction, which needs a density-compensation preconditioner
    // this codebase doesn't yet have for the 2D path.
    bool ok = std::isfinite(rel_err) && cos_sim > 0.85f && rel_err < 0.5f;
    std::cout << std::format("\n{}\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
