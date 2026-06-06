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
// split-Bregman solvers:
//   src/bregman.cpp        (tomocam::opt::split_bregman)
//   src/gpu/bregman.cu     (tomocam::gpu::opt::split_bregman)
//
// Ops tested:
//   operator+=   ha += hb (Array)          vs  DeviceArray::operator+=
//   sqrt         std::sqrt per element     vs  gpu::array::sqrt(a)
//   compute_sk   manual sum-of-sq loop     vs  gpu::opt::compute_sk(dx, b)
//   shrink       manual TV shrink loop     vs  gpu::opt::shrink(dx, b, sk, λ, μ)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

#include "array.h"
#include "array_ops.h"
#include "gpu/bregman.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"

using namespace tomocam;
namespace gpu_arr = tomocam::gpu::array;
namespace gpu_opt = tomocam::gpu::opt;
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
static constexpr float EPSILON = 1e-8f;

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
// Test: operator+=  (in-place element-wise add)
// Both solvers: b[i] += dx[i] - d[i]  and  sk += temp  (GPU compute_sk)
// -------------------------------------------------------------------------

void test_inplace_add_vs_cpu() {
    dims_t d{1, 8, 16};
    auto ha = random_array(d, 1);
    auto hb = random_array(d, 2);

    // CPU: a += b  (Array::operator+=)
    auto cpu_a = ha.clone();
    cpu_a += hb;

    // GPU: da += db  (DeviceArray::operator+=)
    DeviceArray<float> da(ha), db(hb);
    da += db;
    auto gpu_a = da.to_host();

    float err = rel_array(cpu_a, gpu_a);
    check(err < REL_TOL, "operator+=: Array::operator+= vs DeviceArray::operator+=", err);
}

// -------------------------------------------------------------------------
// Test: element-wise sqrt
// GPU bregman.cu: array::sqrt(sk)  in compute_sk
// CPU bregman.cpp: std::sqrt(sum_sq) per element
// -------------------------------------------------------------------------

void test_sqrt_vs_cpu() {
    dims_t d{1, 8, 16};
    auto ha = random_array(d, 3);

    // CPU: element-by-element sqrt
    auto cpu_s = ha.clone();
    for (size_t i = 0; i < cpu_s.size(); ++i) cpu_s[i] = std::sqrt(ha[i]);

    // GPU: gpu::array::sqrt(a)
    DeviceArray<float> da(ha);
    auto ds = gpu_arr::sqrt(da);
    auto gpu_s = ds.to_host();

    float err = rel_array(cpu_s, gpu_s);
    check(err < REL_TOL, "sqrt: element-wise std::sqrt vs gpu::array::sqrt", err);
}

// -------------------------------------------------------------------------
// Test: compute_sk  (sk[i] = sqrt( sum_j (dx[j][i] + b[j][i])^2 ))
// CPU bregman.cpp: manual triple loop
// GPU bregman.cu: gpu::opt::compute_sk(dx, b)
// -------------------------------------------------------------------------

void test_compute_sk_vs_cpu() {
    dims_t d{1, 8, 16};

    std::array<Array<float>, 3> hdx, hb;
    for (int j = 0; j < 3; ++j) {
        hdx[j] = random_array(d, 10 + j);
        hb[j]  = random_array(d, 20 + j);
    }

    // CPU: sk[i] = sqrt( sum_j (dx[j][i] + b[j][i])^2 )
    Array<float> cpu_sk(d);
    for (size_t i = 0; i < cpu_sk.size(); ++i) {
        float sum_sq = 0.f;
        for (int j = 0; j < 3; ++j) {
            float val = hdx[j][i] + hb[j][i];
            sum_sq += val * val;
        }
        cpu_sk[i] = std::sqrt(sum_sq);
    }

    // GPU: compute_sk(dx, b)
    std::array<DeviceArray<float>, 3> ddx = {
        DeviceArray<float>(hdx[0]),
        DeviceArray<float>(hdx[1]),
        DeviceArray<float>(hdx[2])
    };
    std::array<DeviceArray<float>, 3> db = {
        DeviceArray<float>(hb[0]),
        DeviceArray<float>(hb[1]),
        DeviceArray<float>(hb[2])
    };
    auto dsk = gpu_opt::compute_sk(ddx, db);
    auto gpu_sk = dsk.to_host();

    float err = rel_array(cpu_sk, gpu_sk);
    check(err < REL_TOL, "compute_sk: manual loop vs gpu::opt::compute_sk", err);
}

// -------------------------------------------------------------------------
// Test: shrink  (isotropic TV shrinkage)
// CPU bregman.cpp:
//   val = (dx[i] + b[i]) / (sk[i] + eps)
//   d[i] = max(0, sk[i] - lambda/mu) * val
// GPU bregman.cu: gpu::opt::shrink(dx, b, sk, lambda, mu)
// -------------------------------------------------------------------------

void test_shrink_vs_cpu() {
    dims_t d{1, 8, 16};
    auto hdx = random_array(d, 4);
    auto hb  = random_array(d, 5);
    auto hsk = random_array(d, 6);  // positive values only (from random_array)
    float lambda = 0.1f;
    float mu     = 1.0f;

    // CPU: d[i] = max(0, sk[i] - lambda/mu) * (dx[i] + b[i]) / (sk[i] + eps)
    Array<float> cpu_d(d);
    for (size_t i = 0; i < cpu_d.size(); ++i) {
        float val = (hdx[i] + hb[i]) / (hsk[i] + EPSILON);
        cpu_d[i] = std::max(0.f, hsk[i] - lambda / mu) * val;
    }

    // GPU: shrink(dx, b, sk, lambda, mu)
    DeviceArray<float> ddx(hdx), db(hb), dsk(hsk);
    auto dd = gpu_opt::shrink(ddx, db, dsk, lambda, mu);
    auto gpu_d = dd.to_host();

    float err = rel_array(cpu_d, gpu_d);
    check(err < REL_TOL, "shrink: manual loop vs gpu::opt::shrink", err);
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

int main() {
    std::cout << "=== Bregman array-ops: CPU vs GPU ===\n\n";

    test_inplace_add_vs_cpu();
    test_sqrt_vs_cpu();
    test_compute_sk_vs_cpu();
    test_shrink_vs_cpu();

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
