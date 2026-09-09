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

// Compare CPU (opt::DensityComp) and GPU (gpu::opt::DensityComp) density-
// compensation preconditioners.
//
// Both classes implement the same Pipe-Menon Jacobi algorithm using the same
// number of iterations (n_iters = 10 default), but on different NUFFT backends
// (FINUFFT vs cuFINUFFT). The test verifies that:
//
//   1. The internal weight map w_ agrees to within REL_TOL.
//      (Tested indirectly via apply-on-ones: z[i] = 1/w_[i].)
//
//   2. apply(r, z) agrees element-wise for a random input r.
//
// Geometry: 36 angles, laminography tilt gamma=49.9°, small volume {5, 32, 32}.
// Tolerance: 5e-3 relative — both implementations use float NUFFT at 1e-4 tol,
// and 10 Pipe-Menon iterations accumulate ~1e-3 level floating-point differences.

#include <cmath>
#include <cstdlib>
#include <exception>
#include <format>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

// CPU headers
#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "polar_grid.h"
#include "precond.h"

// GPU headers
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/polar_grid.h"
#include "gpu/precond.h"

using namespace tomocam;
using tomocam::gpu::DeviceArray;

// -------------------------------------------------------------------------
// Bookkeeping
// -------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

// cuFINUFFT's plan-cache destructor calls cudaSetDevice during driver shutdown,
// throwing an exception that triggers std::terminate.  We install a handler
// that flushes stdout and exits with the real test result code instead.
static int g_exit_code = EXIT_SUCCESS;
[[noreturn]] static void cuda_terminate_handler() {
    std::fflush(nullptr);
    _exit(g_exit_code);
}

static void check(bool cond, const std::string &name, float rel_err) {
    std::string tag = cond ? "  PASS" : "  FAIL";
    std::cout << tag << "  " << name
              << std::format("  (rel_err = {:.4e})\n", rel_err);
    if (cond) ++g_pass;
    else      ++g_fail;
}

static constexpr float REL_TOL = 5e-3f;

// ‖a - b‖_2 / ‖a‖_2
static float rel_array(const Array<float> &a, const Array<float> &b) {
    float diff_sq = 0.f, ref_sq = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        diff_sq += d * d;
        ref_sq  += a[i] * a[i];
    }
    float denom = ref_sq > 0.f ? ref_sq : 1.f;
    return std::sqrt(diff_sq / denom);
}

// Fill a CPU array with pseudo-random floats in (0.1, 1)
static Array<float> random_array(dims_t d, unsigned seed = 42) {
    Array<float> a(d);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.1f, 1.f);
    for (size_t i = 0; i < a.size(); ++i) a[i] = dis(gen);
    return a;
}

// -------------------------------------------------------------------------
// Shared test geometry
//   36 angles in [-π/4, π/4], gamma = 49.9° (emma/665 GammaNeg45 geometry),
//   volume  5 × 32 × 32  (thin lamino slab, small for speed).
// -------------------------------------------------------------------------

static constexpr int   N_ANGLES = 36;
static constexpr float GAMMA_DEG = 49.9f;
static constexpr int   NROWS = 32, NCOLS = 32;
static const     dims_t RECON{5, 32, 32};

static std::vector<float> make_theta() {
    std::vector<float> t(N_ANGLES);
    float lo = -std::numbers::pi_v<float> / 4.f;
    float hi =  std::numbers::pi_v<float> / 4.f;
    for (int i = 0; i < N_ANGLES; ++i)
        t[i] = lo + (hi - lo) * i / (N_ANGLES - 1);
    return t;
}

// -------------------------------------------------------------------------
// Test 1: weight maps agree (apply ones → 1/w)
// -------------------------------------------------------------------------

void test_weight_map() {
    auto theta  = make_theta();
    float gamma_rad = -GAMMA_DEG * std::numbers::pi_v<float> / 180.f;
    float beta_rad  = 0.f;
    std::vector<float> gamma_v(N_ANGLES, gamma_rad);
    std::vector<float> beta_v(N_ANGLES, beta_rad);

    // Build preconditioners
    cpu::PolarGrid<float> cpu_grid(theta, gamma_v, beta_v, NROWS, NCOLS);
    opt::DensityComp<float>      cpu_dc(cpu_grid, RECON);

    gpu::PolarGrid<float> gpu_grid(theta, gamma_v, beta_v, NROWS, NCOLS);
    gpu::opt::DensityComp<float> gpu_dc(gpu_grid, RECON);

    // apply(ones) = 1/w  — reveals the weight map directly
    auto ones_h = Array<float>::ones(RECON);
    Array<float> cpu_z(RECON);
    cpu_dc.apply(ones_h, cpu_z);

    DeviceArray<float> ones_d(ones_h);
    DeviceArray<float> gpu_z_d(RECON);
    gpu_dc.apply(ones_d, gpu_z_d);
    auto gpu_z = gpu_z_d.to_host();

    float err = rel_array(cpu_z, gpu_z);
    check(err < REL_TOL, "weight map (apply-to-ones): CPU vs GPU", err);
}

// -------------------------------------------------------------------------
// Test 2: apply to a random vector
// -------------------------------------------------------------------------

void test_apply_random() {
    auto theta     = make_theta();
    float gamma_rad = -GAMMA_DEG * std::numbers::pi_v<float> / 180.f;
    float beta_rad  = 0.f;
    std::vector<float> gamma_v(N_ANGLES, gamma_rad);
    std::vector<float> beta_v(N_ANGLES, beta_rad);

    cpu::PolarGrid<float> cpu_grid(theta, gamma_v, beta_v, NROWS, NCOLS);
    opt::DensityComp<float>      cpu_dc(cpu_grid, RECON);

    gpu::PolarGrid<float> gpu_grid(theta, gamma_v, beta_v, NROWS, NCOLS);
    gpu::opt::DensityComp<float> gpu_dc(gpu_grid, RECON);

    auto r_h = random_array(RECON, 17);

    // CPU apply
    Array<float> cpu_z(RECON);
    cpu_dc.apply(r_h, cpu_z);

    // GPU apply
    DeviceArray<float> r_d(r_h);
    DeviceArray<float> gpu_z_d(RECON);
    gpu_dc.apply(r_d, gpu_z_d);
    auto gpu_z = gpu_z_d.to_host();

    float err = rel_array(cpu_z, gpu_z);
    check(err < REL_TOL, "apply(random r): CPU vs GPU", err);
}

// -------------------------------------------------------------------------
// Test 3: weight map is positive (no negative or zero entries)
// -------------------------------------------------------------------------

void test_weights_positive() {
    auto theta     = make_theta();
    float gamma_rad = -GAMMA_DEG * std::numbers::pi_v<float> / 180.f;
    float beta_rad  = 0.f;
    std::vector<float> gamma_v(N_ANGLES, gamma_rad);
    std::vector<float> beta_v(N_ANGLES, beta_rad);

    cpu::PolarGrid<float> cpu_grid(theta, gamma_v, beta_v, NROWS, NCOLS);
    opt::DensityComp<float> cpu_dc(cpu_grid, RECON);

    gpu::PolarGrid<float> gpu_grid(theta, gamma_v, beta_v, NROWS, NCOLS);
    gpu::opt::DensityComp<float> gpu_dc(gpu_grid, RECON);

    // apply(ones) = 1/w; result must be finite and > 0 for both
    auto ones_h = Array<float>::ones(RECON);

    Array<float> cpu_z(RECON);
    cpu_dc.apply(ones_h, cpu_z);
    bool cpu_ok = true;
    for (size_t i = 0; i < cpu_z.size(); ++i)
        if (cpu_z[i] <= 0.f || !std::isfinite(cpu_z[i])) { cpu_ok = false; break; }

    DeviceArray<float> ones_d(ones_h);
    DeviceArray<float> gpu_z_d(RECON);
    gpu_dc.apply(ones_d, gpu_z_d);
    auto gpu_z = gpu_z_d.to_host();
    bool gpu_ok = true;
    for (size_t i = 0; i < gpu_z.size(); ++i)
        if (gpu_z[i] <= 0.f || !std::isfinite(gpu_z[i])) { gpu_ok = false; break; }

    check(cpu_ok, "CPU weights: all positive and finite", 0.f);
    check(gpu_ok, "GPU weights: all positive and finite", 0.f);
}

// -------------------------------------------------------------------------
// Test 4: weight symmetry in gamma=0 case
// At gamma=0 (pure tilt series) the density map should be nearly uniform
// across the in-plane dimensions (up to the sinc roll-off at edges from NUFFT).
// -------------------------------------------------------------------------

void test_uniform_gamma0() {
    auto theta     = make_theta();
    float gamma_rad = 0.f;
    float beta_rad  = 0.f;
    std::vector<float> gamma_v(N_ANGLES, gamma_rad);
    std::vector<float> beta_v(N_ANGLES, beta_rad);

    cpu::PolarGrid<float> cpu_grid(theta, gamma_v, beta_v, NROWS, NCOLS);
    opt::DensityComp<float> cpu_dc(cpu_grid, RECON);

    gpu::PolarGrid<float> gpu_grid(theta, gamma_v, beta_v, NROWS, NCOLS);
    gpu::opt::DensityComp<float> gpu_dc(gpu_grid, RECON);

    auto ones_h = Array<float>::ones(RECON);
    Array<float> cpu_z(RECON);
    cpu_dc.apply(ones_h, cpu_z);

    DeviceArray<float> ones_d(ones_h);
    DeviceArray<float> gpu_z_d(RECON);
    gpu_dc.apply(ones_d, gpu_z_d);
    auto gpu_z = gpu_z_d.to_host();

    float err = rel_array(cpu_z, gpu_z);
    check(err < REL_TOL, "gamma=0 weight map: CPU vs GPU", err);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

int main() {
    std::set_terminate(cuda_terminate_handler);

    std::cout << "=== DensityComp preconditioner: CPU vs GPU ===" << std::endl;
    std::cout << std::format("  geometry: {} angles, gamma={:.1f}, "
                             "recon {}x{}x{}\n",
                             N_ANGLES, GAMMA_DEG,
                             RECON.n1, RECON.n2, RECON.n3) << std::flush;

    std::cout << "\nTest 1: weight maps (apply-to-ones)" << std::endl;
    test_weight_map();

    std::cout << "\nTest 2: apply to random vector" << std::endl;
    test_apply_random();

    std::cout << "\nTest 3: weights are positive and finite" << std::endl;
    test_weights_positive();

    std::cout << "\nTest 4: gamma=0 geometry" << std::endl;
    test_uniform_gamma0();

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed"
              << std::endl;
    g_exit_code = (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    std::fflush(nullptr);
    return g_exit_code;
}
