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

// Compare CPU (finufft) and GPU (cufinufft) implementations of the 2D
// slice-by-slice (vertical rotation axis) tomography path:
//   - forward2d   (src/projection.cpp  vs  src/gpu/projection.cu)
//   - backproj2d  (src/projection.cpp  vs  src/gpu/projection.cu)
//   - sysmat2d    (src/gradient.cpp    vs  src/gpu/gradient.cu)
//
// Setup: 141 angles uniformly spaced in [-1.222, 1.222] radians,
//        random float volume of shape {5, 255, 255}.

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
#include "polar_grid2d.h"

// CPU projection/sysmat declarations (avoid tomocam.h which pulls in toml++)
namespace tomocam {
    template <typename T>
    Array<T> forward2d(const Array<T> &volume, const cpu::PolarGrid2D<T> &grid);

    template <typename T>
    Array<T> backproj2d(const Array<T> &sinogram, const cpu::PolarGrid2D<T> &grid,
                        const dims_t &dims);

    template <typename T>
    Array<T> sysmat2d(const Array<T> &x, const cpu::PolarGrid2D<T> &grid);
} // namespace tomocam

// GPU headers
#include "gpu/cufft_plan_cache.h"
#include "gpu/cufinufft_plan_cache.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/polar_grid2d.h"
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
static constexpr size_t Nz = 5;
static constexpr size_t N = 255;
static constexpr float THETA_MIN = -1.222f;
static constexpr float THETA_MAX = 1.222f;
static constexpr float REL_TOL = 5e-4f;

static std::vector<float> make_theta() {
    std::vector<float> theta(NTHETA);
    for (size_t i = 0; i < NTHETA; ++i)
        theta[i] =
            THETA_MIN + i * (THETA_MAX - THETA_MIN) / static_cast<float>(NTHETA - 1);
    return theta;
}

// -------------------------------------------------------------------------
// Test: forward2d  CPU vs GPU
// -------------------------------------------------------------------------

void test_forward2d(const Array<float> &vol,
                    const tomocam::cpu::PolarGrid2D<float> &cpu_pg,
                    const tomocam::gpu::PolarGrid2D<float> &gpu_pg) {

    auto cpu_proj = tomocam::forward2d(vol, cpu_pg);

    DeviceArray<float> d_vol(vol);
    auto d_proj = tomocam::gpu::forward2d(d_vol, gpu_pg);
    auto gpu_proj = d_proj.to_host();

    float err = rel_error(cpu_proj, gpu_proj);
    check(err < REL_TOL, "forward2d: CPU vs GPU", err);
}

// -------------------------------------------------------------------------
// Test: backproj2d  CPU vs GPU
// -------------------------------------------------------------------------

void test_backproj2d(const Array<float> &proj,
                     const tomocam::cpu::PolarGrid2D<float> &cpu_pg,
                     const tomocam::gpu::PolarGrid2D<float> &gpu_pg,
                     dims_t vol_dims) {

    auto cpu_vol = tomocam::backproj2d(proj, cpu_pg, vol_dims);

    DeviceArray<float> d_proj(proj);
    auto d_vol = tomocam::gpu::backproj2d(d_proj, gpu_pg, vol_dims);
    auto gpu_vol = d_vol.to_host();

    float err = rel_error(cpu_vol, gpu_vol);
    check(err < REL_TOL, "backproj2d: CPU vs GPU", err);
}

// -------------------------------------------------------------------------
// Test: sysmat2d  CPU vs GPU
// -------------------------------------------------------------------------

void test_sysmat2d(const Array<float> &vol,
                   const tomocam::cpu::PolarGrid2D<float> &cpu_pg,
                   const tomocam::gpu::PolarGrid2D<float> &gpu_pg) {

    auto cpu_sm = tomocam::sysmat2d(vol, cpu_pg);

    DeviceArray<float> d_vol(vol);
    auto d_sm = tomocam::gpu::sysmat2d(d_vol, gpu_pg);
    auto gpu_sm = d_sm.to_host();

    float err = rel_error(cpu_sm, gpu_sm);
    check(err < REL_TOL, "sysmat2d (A^T A): CPU vs GPU", err);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

int main() {
    std::cout << std::format("=== CPU vs GPU 2D projection tests ===\n");
    std::cout << std::format(
        "    volume: {{{}, {}, {}}}  |  {} angles in [{:.3f}, {:.3f}] rad\n\n", Nz,
        N, N, NTHETA, THETA_MIN, THETA_MAX);

    dims_t vol_dims = {Nz, N, N};
    auto theta = make_theta();

    tomocam::cpu::PolarGrid2D<float> cpu_pg(theta, N);
    tomocam::gpu::PolarGrid2D<float> gpu_pg(theta, N);

    auto vol = Array<float>::random(vol_dims);
    auto proj = random_array(dims_t{Nz, NTHETA, N}); // shape: {Nz, NTHETA, N}

    test_forward2d(vol, cpu_pg, gpu_pg);
    test_backproj2d(proj, cpu_pg, gpu_pg, vol_dims);
    test_sysmat2d(vol, cpu_pg, gpu_pg);

    std::cout << std::format("\nResults: {} passed, {} failed\n", g_pass, g_fail);

    // Flush caches before CUDA context tears down (avoids a cudaSetDevice
    // failure during static destruction as the driver shuts down).
    gpu::fft::cache::plans<float>.clear();
    gpu::nufft::plans::cache<float>.clear();
    cudaDeviceReset();

    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
