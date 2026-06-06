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

// Compare the array operations from include/array_ops.h (CPU) and
// include/gpu/device_array_ops.h (GPU) that are called inside the two
// conjugate-gradient solvers:
//   src/conjgrad.cpp        (tomocam::opt::cgsolver)
//   src/gpu/conjgrad.cu     (tomocam::gpu::opt::cgsolver)
//
// Ops tested:
//   dot    array::dot              vs  gpu::array::dot
//   norm2  array::norm2            vs  gpu::array::norm2
//   axpy   p = z + p*beta (ops)   vs  gpu::array::axpy(p, beta, z)  [x = x*alpha + y]
//   xpay   x += p*alpha (ops)     vs  gpu::array::xpay(x, p, alpha) [x += y*beta]
//   sub    r = a - b (operator-)  vs  DeviceArray operator-

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

#include "array.h"
#include "array_ops.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"

using namespace tomocam;
namespace cpu = tomocam::array;
namespace gpu_arr = tomocam::gpu::array;
using tomocam::gpu::DeviceArray;

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const std::string &name, float rel_err) {
    std::string tag = cond ? "  PASS" : "  FAIL";
    std::cout << tag << "  " << name
              << "  (rel_err = " << rel_err << ")\n";
    if (cond) ++g_pass; else ++g_fail;
}

static constexpr float REL_TOL = 1e-4f;

static Array<float> random_array(dims_t d, unsigned seed = 42) {
    Array<float> a(d);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.1f, 1.f);
    for (size_t i = 0; i < a.size(); ++i) a[i] = dis(gen);
    return a;
}

static float rel_scalar(float cpu_val, float gpu_val) {
    float denom = std::abs(cpu_val) > 0.f ? std::abs(cpu_val) : 1.f;
    return std::abs(gpu_val - cpu_val) / denom;
}

static float rel_array(const Array<float> &cpu_a, const Array<float> &gpu_a) {
    float diff_sq = 0.f, ref_sq = 0.f;
    for (size_t i = 0; i < cpu_a.size(); ++i) {
        float d = cpu_a[i] - gpu_a[i];
        diff_sq += d * d;
        ref_sq  += cpu_a[i] * cpu_a[i];
    }
    return (ref_sq > 0.f) ? std::sqrt(diff_sq / ref_sq) : std::sqrt(diff_sq);
}

// -------------------------------------------------------------------------
// Test: dot  (array::dot vs gpu::array::dot)
// Called in both CG solvers: rs_old = dot(z, r), pAp = dot(p, Ap), etc.
// -------------------------------------------------------------------------

void test_dot_vs_cpu() {
    dims_t d{1, 8, 16};
    auto ha = random_array(d, 1);
    auto hb = random_array(d, 2);

    float cpu_d = cpu::dot(ha, hb);

    DeviceArray<float> da(ha), db(hb);
    float gpu_d = gpu_arr::dot(da, db);

    float err = rel_scalar(cpu_d, gpu_d);
    check(err < REL_TOL, "dot: array::dot vs gpu::array::dot", err);
}

// -------------------------------------------------------------------------
// Test: norm2  (array::norm2 vs gpu::array::norm2)
// Both CG solvers use norm2 to compute the step size:
//   CPU: array::norm2(step), array::norm2(x)
//   GPU: gpu::array::norm2(p * alpha), gpu::array::norm2(x)
// -------------------------------------------------------------------------

void test_norm2_vs_cpu() {
    dims_t d{1, 8, 16};
    auto ha = random_array(d, 3);

    float cpu_n = cpu::norm2(ha);

    DeviceArray<float> da(ha);
    float gpu_n = gpu_arr::norm2(da);  // as used in conjgrad.cu

    float err = rel_scalar(cpu_n, gpu_n);
    check(err < REL_TOL, "norm2: array::norm2 vs gpu::array::norm2", err);
}

// -------------------------------------------------------------------------
// Test: axpy  (p = z + beta * p)
// CPU conjgrad.cpp: p = z + (p * (rs_new / rs_old))  (Array operators)
// GPU conjgrad.cu:  gpu::array::axpy(p, beta, z)  =>  p = p * beta + z
// Both expressions compute z + beta*p.
// -------------------------------------------------------------------------

void test_axpy_vs_cpu() {
    dims_t d{1, 8, 16};
    auto hp = random_array(d, 4);
    auto hz = random_array(d, 5);
    float beta = 2.5f;

    // CPU: p = z + p * beta  (search direction update in conjgrad.cpp)
    auto cpu_p = hz + hp * beta;

    // GPU: axpy(p, beta, z)  =>  p = p * beta + z  (conjgrad.cu line: axpy(p, beta, z))
    DeviceArray<float> dp(hp), dz(hz);
    gpu_arr::axpy(dp, beta, dz);
    auto gpu_p = dp.to_host();

    float err = rel_array(cpu_p, gpu_p);
    check(err < REL_TOL, "axpy: Array ops (z + beta*p) vs gpu::array::axpy(p, beta, z)", err);
}

// -------------------------------------------------------------------------
// Test: xpay  (x += y * beta)
// CPU conjgrad.cpp: x += p * alpha  and  r -= Ap * alpha  (Array operators)
// GPU conjgrad.cu:  gpu::array::xpay(x, p, alpha)  =>  x += p * beta
// -------------------------------------------------------------------------

void test_xpay_vs_cpu() {
    dims_t d{1, 8, 16};
    auto hx = random_array(d, 6);
    auto hy = random_array(d, 7);
    float beta = 0.75f;

    // CPU: x += y * beta  (solution / residual update in conjgrad.cpp)
    auto cpu_p = hx + hy * beta;

    // GPU: xpay(x, y, beta) => x += y * beta
    DeviceArray<float> dx(hx), dy(hy);
    gpu_arr::xpay(dx, dy, beta);
    auto gpu_p = dx.to_host();

    float err = rel_array(cpu_p, gpu_p);
    check(err < REL_TOL, "xpay: Array ops (x + y*beta) vs gpu::array::xpay", err);
}

// -------------------------------------------------------------------------
// Test: operator-  (out = a - b)
// CPU conjgrad.cpp: r = yT - A(x)  (Array operator-)
// GPU conjgrad.cu:  r = A(x) - y   (DeviceArray operator-)
// -------------------------------------------------------------------------

void test_subtract_vs_cpu() {
    dims_t d{1, 8, 16};
    auto ha = random_array(d, 8);
    auto hb = random_array(d, 9);

    // CPU: r = a - b  via Array operator-
    auto cpu_r = ha - hb;

    // GPU: DeviceArray operator-  =>  out = a - b
    DeviceArray<float> da(ha), db(hb);
    auto dout = da - db;
    auto gpu_r = dout.to_host();

    float err = rel_array(cpu_r, gpu_r);
    check(err < REL_TOL, "operator-: Array operator- vs DeviceArray operator-", err);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

int main() {
    std::cout << "=== CG array-ops: CPU vs GPU ===\n\n";

    test_dot_vs_cpu();
    test_norm2_vs_cpu();
    test_axpy_vs_cpu();
    test_xpay_vs_cpu();
    test_subtract_vs_cpu();

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
