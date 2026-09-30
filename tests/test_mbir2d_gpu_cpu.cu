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

// GPU 2D reconstruction checks:
//   1. PointSpreadFunction2D::convolve: slab-by-slab result equals the
//      single-shot result (slices are independent).
//   2. gpu::MBIR2D agrees with the CPU pipeline (CG and Split-Bregman) on a
//      small phantom, to within the float NUFFT tolerance.

#include <cmath>
#include <format>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "gpu/cufft_plan_cache.h"
#include "gpu/cufinufft_plan_cache.h"
#include "gpu/device_array.h"
#include "gpu/polar_grid2d.h"
#include "gpu/toeplitz2d.h"
#include "gpu/tomocam.h"
#include "optimize.h"
#include "polar_grid2d.h"
#include "projection.h"
#include "recon_params.h"

using namespace tomocam;
using tomocam::gpu::DeviceArray;

static float rel_err(const Array<float> &a, const Array<float> &b) {
    return array::norm2(a - b) / array::norm2(b);
}

static std::vector<float> make_theta(size_t n) {
    std::vector<float> t(n);
    for (size_t i = 0; i < n; ++i) t[i] = static_cast<float>(i) * M_PI / n;
    return t;
}

// disk whose center drifts with the slice index
static Array<float> make_phantom(size_t nslices, size_t n) {
    Array<float> p = Array<float>::zeros(dims_t{nslices, n, n});
    float c = n / 2.0f, r = n * 0.3f;
    for (size_t i = 0; i < nslices; ++i) {
        float sx = c + (static_cast<float>(i) - nslices / 2.0f) * 2.0f;
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k) {
                float dj = j - c, dk = k - sx;
                if (dj * dj + dk * dk < r * r) p[{i, j, k}] = 1.0f;
            }
    }
    return p;
}

static bool report(const char *name, float err, float tol) {
    bool ok = err < tol;
    std::cout << std::format("  {}  {}  (rel_err = {:.4e}, tol = {:.1e})\n",
                             ok ? "PASS" : "FAIL", name, err, tol);
    return ok;
}

static bool test_chunked_convolve() {
    const dims_t dims{7, 63, 63};
    auto theta = make_theta(40);
    gpu::PolarGrid2D<float> pg(theta, dims.n3);
    gpu::PointSpreadFunction2D<float> psf(pg, dims.n3);

    DeviceArray<float> d_x(Array<float>::random(dims));
    auto whole = psf.convolve(d_x).to_host();
    psf.set_max_partition(2); // 7 slices -> 4 x 2, last zero-padded
    auto slabs = psf.convolve(d_x).to_host();
    return report("convolve: slabs of 2 vs single shot", rel_err(slabs, whole),
                  1e-6f);
}

static bool test_mbir2d(Regularizer reg, const char *name) {
    const size_t Nz = 3, N = 64, NTHETA = 90;
    const dims_t dims{Nz, N, N};
    auto theta = make_theta(NTHETA);
    cpu::PolarGrid2D<float> pg(theta, N);
    auto phantom = make_phantom(Nz, N);
    auto sinogram = forward2d(phantom, pg);

    ReconParams params;
    params.regularizer = reg;
    params.recon_dims = {Nz, N, N};
    params.maxIters = (reg == Regularizer::SPLIT_BREGMAN) ? 5 : 20;
    params.innerIters = 2;
    params.lambda = 0.1f;
    params.mu = 5.0f;
    params.tol = 1e-5f;
    params.xtol = 1e-5f;

    // CPU reference
    cpu::PointSpreadFunction2D<float> psf(pg, N);
    opt::Function<float> A = [&psf](const Array<float> &x) {
        return sysmat2d(x, psf);
    };
    auto rhs = backproj2d(sinogram, pg, dims);
    auto x0 = Array<float>::zeros(dims);
    Array<float> ref =
        (reg == Regularizer::SPLIT_BREGMAN)
            ? opt::split_bregman<float>(A, rhs, x0, params.lambda, params.mu,
                                        params.maxIters, params.innerIters,
                                        params.tol, params.xtol)
            : opt::cgsolver<float>(A, rhs, x0, params.maxIters, params.tol,
                                   params.xtol);

    auto out = gpu::MBIR2D<float>(sinogram, theta, params);
    // Split-Bregman runs several nonlinear outer iterations on top of the
    // ~4e-4 GPU/CPU NUFFT difference, so allow more drift than plain CG.
    float tol = (reg == Regularizer::SPLIT_BREGMAN) ? 1e-2f : 5e-3f;
    bool ok = report(name, rel_err(out, ref), tol);

    // Both must recover the phantom equally well.
    float e_gpu = rel_err(out, phantom), e_cpu = rel_err(ref, phantom);
    std::cout << std::format("        error vs phantom: GPU {:.4f}, CPU {:.4f}\n",
                             e_gpu, e_cpu);
    bool same_quality = std::abs(e_gpu - e_cpu) < 0.01f * e_cpu + 1e-3f;
    if (!same_quality) std::cout << "  FAIL  reconstruction quality differs\n";
    return ok && same_quality;
}

int main() {
    std::cout << "=== GPU 2D reconstruction tests ===\n";
    int failed = 0;
    failed += !test_chunked_convolve();
    failed += !test_mbir2d(Regularizer::UNCONSTRAINED, "MBIR2D CG: GPU vs CPU");
    failed +=
        !test_mbir2d(Regularizer::SPLIT_BREGMAN, "MBIR2D Split-Bregman: GPU vs CPU");
    std::cout << std::format("\nResults: {} passed, {} failed\n", 3 - failed,
                             failed);

    gpu::nufft::plans::cache<float>.clear();
    gpu::fft::cache::plans<float>.clear();
    return failed == 0 ? 0 : 1;
}
