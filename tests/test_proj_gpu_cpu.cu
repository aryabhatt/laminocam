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

// Compare CPU (finufft) and GPU (cufinufft) implementations of:
//   - forward projection  (src/projection.cpp  vs  src/gpu/projection.cu)
//   - adjoint projection  (src/projection.cpp  vs  src/gpu/projection.cu)
//   - sysmat              (src/gradient.cpp    vs  include/gpu/sysmat.h)
//
// Setup: 141 angles uniformly spaced in [-1.222, 1.222] radians,
//        random float volume of shape {1, 64, 64}.

#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// CPU headers
#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "polar_grid.h"

// CPU projection/sysmat declarations (avoid tomocam.h which pulls in toml++)
namespace tomocam {
    template <typename T>
    Array<T> forward(const Array<T> &volume, const PolarGrid<T> &grid,
                     T gamma = T(0));

    template <typename T>
    Array<T> backward(const Array<T> &projections, const PolarGrid<T> &grid,
                      const dims_t &dims, T gamma = T(0), bool filter = false,
                      const std::string &filter_type = "");

    template <typename T>
    Array<T> sysmat(const Array<T> &x, const PolarGrid<T> &grid);
} // namespace tomocam

// GPU headers
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/polar_grid.h"
#include "gpu/projection.h"

using namespace tomocam;
using tomocam::gpu::DeviceArray;

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const std::string &name, float rel_err) {
    std::string tag = cond ? "  PASS" : "  FAIL";
    std::cout << std::format("{}  {}  (rel_err = {:.4e})\n", tag, name, rel_err);
    if (cond)
        ++g_pass;
    else
        ++g_fail;
}

static Array<float> random_array(dims_t d) {
    Array<float> a(d);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.f, 1.f);
    for (size_t i = 0; i < a.size(); ++i) a[i] = dis(gen);
    return a;
}

// Compute relative L2 error between two host arrays (same size).
static float rel_error(const Array<float> &cpu, const Array<float> &gpu) {
    float diff_sq = 0.f, ref_sq = 0.f;
    for (size_t i = 0; i < cpu.size(); ++i) {
        float d = cpu[i] - gpu[i];
        diff_sq += d * d;
        ref_sq += cpu[i] * cpu[i];
    }
    return (ref_sq > 0.f) ? std::sqrt(diff_sq / ref_sq) : std::sqrt(diff_sq);
}

// -------------------------------------------------------------------------
// Geometry
// -------------------------------------------------------------------------

static constexpr size_t NTHETA = 141;
static constexpr size_t Nz = 63;
static constexpr size_t N = 255;
static constexpr float THETA_MIN = -1.222f;
static constexpr float THETA_MAX = 1.222f;
static constexpr float REL_TOL = 5e-5f;

static std::vector<float> make_theta() {
    std::vector<float> theta(NTHETA);
    for (size_t i = 0; i < NTHETA; ++i)
        theta[i] =
            THETA_MIN + i * (THETA_MAX - THETA_MIN) / static_cast<float>(NTHETA - 1);
    return theta;
}

// -------------------------------------------------------------------------
// Test: forward  CPU vs GPU
// -------------------------------------------------------------------------

void test_forward(const Array<float> &vol, const PolarGrid<float> &cpu_pg,
                  const tomocam::gpu::PolarGrid<float> &gpu_pg) {

    // CPU
    auto cpu_proj = tomocam::forward(vol, cpu_pg);

    // GPU: upload volume, run, download
    DeviceArray<float> d_vol(vol);
    auto d_proj = tomocam::gpu::forward(d_vol, gpu_pg);
    auto gpu_proj = d_proj.to_host();

    float err = rel_error(cpu_proj, gpu_proj);
    check(err < REL_TOL, "forward: CPU vs GPU", err);
}

// -------------------------------------------------------------------------
// Test: backward (adjoint)  CPU vs GPU
// -------------------------------------------------------------------------

void test_backward(const Array<float> &proj, const PolarGrid<float> &cpu_pg,
                   const tomocam::gpu::PolarGrid<float> &gpu_pg, dims_t vol_dims) {

    // CPU (unfiltered backprojection)
    auto cpu_vol = tomocam::backward(proj, cpu_pg, vol_dims);

    // GPU: upload projections, run, download
    DeviceArray<float> d_proj(proj);
    auto d_vol = tomocam::gpu::backward(d_proj, gpu_pg, vol_dims);
    auto gpu_vol = d_vol.to_host();

    float err = rel_error(cpu_vol, gpu_vol);
    check(err < REL_TOL, "backward (adjoint): CPU vs GPU", err);
}

// -------------------------------------------------------------------------
// Test: sysmat  CPU vs GPU
// -------------------------------------------------------------------------

void test_sysmat(const Array<float> &vol, const PolarGrid<float> &cpu_pg,
                 const tomocam::gpu::PolarGrid<float> &gpu_pg) {

    // CPU
    auto cpu_sm = tomocam::sysmat(vol, cpu_pg);

    // GPU
    DeviceArray<float> d_vol(vol);
    auto d_sm = tomocam::gpu::sysmat(d_vol, gpu_pg);
    auto gpu_sm = d_sm.to_host();

    float err = rel_error(cpu_sm, gpu_sm);
    check(err < REL_TOL, "sysmat (A^T A): CPU vs GPU", err);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

int main() {
    std::cout << std::format("=== CPU vs GPU projection tests ===\n");
    std::cout << std::format(
        "    volume: {{{}, {}, {}}}  |  {} angles in [{:.3f}, {:.3f}] rad\n\n", Nz,
        N, N, NTHETA, THETA_MIN, THETA_MAX);

    dims_t vol_dims = {Nz, N, N};
    auto theta = make_theta();

    // Build both grids from the same angle set
    std::vector<float> gamma(NTHETA, 0.f);
    PolarGrid<float> cpu_pg(theta, gamma, N, N);
    tomocam::gpu::PolarGrid<float> gpu_pg(theta, gamma, N, N);

    // Random volume and random projections (same seed → reproducible)
    auto vol = Array<float>::random(vol_dims);
    auto proj = random_array(cpu_pg.dims()); // shape: {NTHETA, N, N}

    test_forward(vol, cpu_pg, gpu_pg);
    test_backward(proj, cpu_pg, gpu_pg, vol_dims);
    test_sysmat(vol, cpu_pg, gpu_pg);

    std::cout << std::format("\nResults: {} passed, {} failed\n", g_pass, g_fail);
    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
