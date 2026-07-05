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

// Compare the two GPU sysmat overloads in src/gpu/gradient.cu:
//   sysmat(x, gpu::PolarGrid)              — cufinufft direct path
//   sysmat(x, gpu::PointSpreadFunction)    — Toeplitz (GPU PSF convolution)
//
// Both compute A^T A x for the same geometry; results must agree to within
// a tolerance set by cufinufft approximation error (~1.2e-6 for float).

#include <cmath>
#include <format>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "gpu/cufft_plan_cache.h"
#include "gpu/cufinufft_plan_cache.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/polar_grid.h"
#include "gpu/projection.h"
#include "gpu/toeplitz.h"

using namespace tomocam;
using tomocam::gpu::DeviceArray;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static float cosine_sim(const Array<float> &a, const Array<float> &b) {
    float na = array::norm2(a), nb = array::norm2(b);
    if (na == 0.f || nb == 0.f) return 0.f;
    return array::dot(a, b) / (na * nb);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test case
// ─────────────────────────────────────────────────────────────────────────────

static bool test_gpu_sysmat(const dims_t &vol_dims, const std::vector<float> &theta,
                            float gamma, float tol) {
    std::cout << std::format("vol({},{},{}) nangles={:<3} gamma={:.2f}  "
                             "GPU NUFFT vs GPU Toeplitz\n\n",
                             vol_dims.n1, vol_dims.n2, vol_dims.n3, theta.size(),
                             gamma);

    std::vector<float> gamma_vec(theta.size(), gamma);
    gpu::PolarGrid<float> pg(theta, gamma_vec, vol_dims.n2, vol_dims.n3);
    gpu::PointSpreadFunction<float> psf(pg, vol_dims);

    // Random volume on host, then upload
    auto h_x = Array<float>::random(vol_dims);
    DeviceArray<float> d_x(h_x);

    // Both sysmat paths on GPU
    auto d_nufft = gpu::sysmat(d_x, pg);
    auto d_toeplitz = gpu::sysmat(d_x, psf);

    // Download for comparison
    auto h_nufft = d_nufft.to_host();
    auto h_toeplitz = d_toeplitz.to_host();

    float n_nufft = array::norm2(h_nufft);
    float n_toeplitz = array::norm2(h_toeplitz);
    float scale_ratio = (n_nufft > 0.f) ? n_toeplitz / n_nufft : 0.f;
    float cos_sim = cosine_sim(h_nufft, h_toeplitz);

    // Normalise Toeplitz result to match NUFFT magnitude, then compute rel diff
    auto h_toeplitz_scaled = h_toeplitz * (1.f / scale_ratio);
    auto diff = h_nufft - h_toeplitz_scaled;
    float rel_diff = array::norm2(diff) / n_nufft;

    bool shape_ok = rel_diff < tol;
    bool scale_ok = std::abs(scale_ratio - 1.f) < tol;
    bool ok = shape_ok && scale_ok;

    std::cout << std::format("  ||nufft||={:.4e}  ||toeplitz||={:.4e}  "
                             "scale_ratio={:.4e}  cos_sim={:.6f}\n",
                             n_nufft, n_toeplitz, scale_ratio, cos_sim);
    std::cout << std::format("  rel_diff={:.2e}  shape={}  scale={}\n", rel_diff,
                             shape_ok ? "OK" : "FAIL", scale_ok ? "OK" : "FAIL");
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n====== GPU sysmat: NUFFT vs Toeplitz agreement tests ======\n\n";

    const float tol = 5e-4f;
    int passed = 0, failed = 0;
    auto record = [&](bool ok) { ok ? ++passed : ++failed; };

    auto make_theta = [](size_t n, float lo, float hi) {
        std::vector<float> t(n);
        for (size_t i = 0; i < n; ++i)
            t[i] = lo + i * (hi - lo) / static_cast<float>(n - 1);
        return t;
    };

    // Small volume, few angles, gamma = 0
    {
        dims_t vol{5, 15, 15};
        auto theta = make_theta(7, -1.1f, 1.1f);
        record(test_gpu_sysmat(vol, theta, 0.f, tol));
    }

    // Larger volume, more angles, gamma = 0
    {
        dims_t vol{21, 311, 311};
        auto theta = make_theta(71, -1.2f, 1.2f);
        record(test_gpu_sysmat(vol, theta, 0.f, tol));
    }

    // Larger volume, more angles, gamma = 0.3
    {
        dims_t vol{21, 311, 311};
        auto theta = make_theta(71, -1.1f, 1.1f);
        record(test_gpu_sysmat(vol, theta, 0.3f, tol));
    }

    std::cout << "\nResults: " << passed << " passed, " << failed << " failed\n";

    // Flush caches before CUDA context tears down
    gpu::fft::cache::plans<float>.clear();
    gpu::nufft::plans::cache<float>.clear();
    cudaDeviceReset();

    return (failed == 0) ? 0 : 1;
}
