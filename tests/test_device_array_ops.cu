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

#include <cmath>
#include <cstdlib>
#include <iostream>
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

static void check(bool cond, const std::string &name) {
    if (cond) {
        std::cout << "  PASS  " << name << "\n";
        ++g_pass;
    } else {
        std::cout << "  FAIL  " << name << "\n";
        ++g_fail;
    }
}

static constexpr float EPS = 1e-4f;

static Array<float> iota_array(dims_t d, float start = 1.f) {
    Array<float> a(d);
    for (size_t i = 0; i < a.size(); ++i) a[i] = start + static_cast<float>(i);
    return a;
}

static Array<float> const_array(dims_t d, float v) {
    Array<float> a(d);
    for (size_t i = 0; i < a.size(); ++i) a[i] = v;
    return a;
}

// -------------------------------------------------------------------------
// Tests — GPU-only operations (no CPU counterpart)
// -------------------------------------------------------------------------

void test_scale_inplace() {
    dims_t d{1, 2, 4};
    auto ha = const_array(d, 3.f);
    DeviceArray<float> da(ha);
    gpu_arr::scale_inplace(da, 2.f);

    auto result = da.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 6.f) < EPS);
    check(ok, "scale_inplace: 3 * 2 = 6");
}

void test_scale() {
    dims_t d{1, 2, 4};
    auto ha = const_array(d, 5.f);
    DeviceArray<float> da(ha);
    auto out = gpu_arr::scale(da, 3.f);

    auto result = out.to_host();
    auto orig   = da.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 15.f) < EPS);
    for (size_t i = 0; i < orig.size(); ++i)
        ok = ok && (std::abs(orig[i] - 5.f) < EPS);
    check(ok, "scale: 5 * 3 = 15, original unchanged");
}

void test_axpy() {
    // x += alpha * y  =>  1 + 3*2 = 7
    dims_t d{1, 2, 4};
    DeviceArray<float> dx(const_array(d, 1.f));
    DeviceArray<float> dy(const_array(d, 2.f));
    gpu_arr::axpy(dx, 3.f, dy);

    auto result = dx.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 7.f) < EPS);
    check(ok, "axpy: x=1, y=2, alpha=3 => x=7");
}

void test_axpy_zero_alpha() {
    dims_t d{1, 1, 4};
    DeviceArray<float> dx(const_array(d, 7.f));
    DeviceArray<float> dy(const_array(d, 99.f));
    gpu_arr::axpy(dx, 0.f, dy);

    auto result = dx.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 7.f) < EPS);
    check(ok, "axpy: alpha=0 leaves x unchanged");
}

void test_xpay() {
    // x = y + beta * x  =>  1 + 2*4 = 9
    dims_t d{1, 2, 4};
    DeviceArray<float> dx(const_array(d, 4.f));
    DeviceArray<float> dy(const_array(d, 1.f));
    gpu_arr::xpay(dx, dy, 2.f);

    auto result = dx.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 9.f) < EPS);
    check(ok, "xpay: x=4, y=1, beta=2 => x=9");
}

void test_subtract() {
    dims_t d{1, 2, 4};
    DeviceArray<float> da(const_array(d, 7.f));
    DeviceArray<float> db(const_array(d, 3.f));
    auto out = gpu_arr::subtract(da, db);

    auto result = out.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 4.f) < EPS);
    check(ok, "subtract: 7 - 3 = 4");
}

void test_subtract_inplace() {
    dims_t d{1, 2, 4};
    DeviceArray<float> da(const_array(d, 7.f));
    DeviceArray<float> db(const_array(d, 3.f));
    gpu_arr::subtract_inplace(da, db);

    auto result = da.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 4.f) < EPS);
    check(ok, "subtract_inplace: 7 - 3 = 4");
}

void test_subtract_self_is_zero() {
    dims_t d{1, 2, 4};
    DeviceArray<float> da(iota_array(d, 1.f));
    auto db = da.clone();
    auto out = gpu_arr::subtract(da, db);

    auto result = out.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i]) < EPS);
    check(ok, "subtract: x - x = 0");
}

void test_add() {
    dims_t d{1, 2, 4};
    DeviceArray<float> da(const_array(d, 2.f));
    DeviceArray<float> db(const_array(d, 5.f));
    auto out = gpu_arr::add(da, db);

    auto result = out.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 7.f) < EPS);
    check(ok, "add: 2 + 5 = 7");
}

void test_add_inplace() {
    dims_t d{1, 2, 4};
    DeviceArray<float> da(const_array(d, 2.f));
    DeviceArray<float> db(const_array(d, 5.f));
    gpu_arr::add_inplace(da, db);

    auto result = da.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 7.f) < EPS);
    check(ok, "add_inplace: 2 + 5 = 7");
}

void test_multiply() {
    dims_t d{1, 2, 4};
    DeviceArray<float> da(const_array(d, 3.f));
    DeviceArray<float> db(const_array(d, 4.f));
    auto out = gpu_arr::multiply(da, db);

    auto result = out.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 12.f) < EPS);
    check(ok, "multiply: 3 * 4 = 12");
}

void test_multiply_inplace() {
    dims_t d{1, 2, 4};
    DeviceArray<float> da(const_array(d, 3.f));
    DeviceArray<float> db(const_array(d, 4.f));
    gpu_arr::multiply_inplace(da, db);

    auto result = da.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - 12.f) < EPS);
    check(ok, "multiply_inplace: 3 * 4 = 12");
}

void test_abs() {
    dims_t d{1, 2, 4};
    Array<float> ha(d);
    for (size_t i = 0; i < ha.size(); ++i)
        ha[i] = (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1);
    DeviceArray<float> da(ha);
    auto out = gpu_arr::abs(da);

    auto result = out.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - static_cast<float>(i + 1)) < EPS);
    check(ok, "abs: negatives become positive");
}

void test_sqrt() {
    // sqrt([1, 4, 9, 16]) = [1, 2, 3, 4]
    dims_t d{1, 1, 4};
    Array<float> ha(d);
    ha[0] = 1.f; ha[1] = 4.f; ha[2] = 9.f; ha[3] = 16.f;
    DeviceArray<float> da(ha);
    auto out = gpu_arr::sqrt(da);

    auto result = out.to_host();
    bool ok = true;
    for (size_t i = 0; i < result.size(); ++i)
        ok = ok && (std::abs(result[i] - static_cast<float>(i + 1)) < EPS);
    check(ok, "sqrt: sqrt([1,4,9,16]) = [1,2,3,4]");
}

// -------------------------------------------------------------------------
// Tests — GPU vs CPU cross-validation (using array_ops.h as reference)
// -------------------------------------------------------------------------

void test_max_vs_cpu() {
    dims_t d{1, 2, 4};
    auto ha = iota_array(d, 1.f);
    float cpu_max = cpu::max(ha);

    DeviceArray<float> da(ha);
    float gpu_max = gpu_arr::max(da);

    check(std::abs(gpu_max - cpu_max) < EPS,
          "max: GPU matches CPU (" + std::to_string(cpu_max) + ")");
}

void test_min_vs_cpu() {
    dims_t d{1, 2, 4};
    auto ha = iota_array(d, 1.f);
    float cpu_min = cpu::min(ha);

    DeviceArray<float> da(ha);
    float gpu_min = gpu_arr::min(da);

    check(std::abs(gpu_min - cpu_min) < EPS,
          "min: GPU matches CPU (" + std::to_string(cpu_min) + ")");
}

void test_norm2_vs_cpu() {
    dims_t d{1, 2, 4};
    auto ha = iota_array(d, 1.f);
    float cpu_n = cpu::norm2(ha);

    DeviceArray<float> da(ha);
    float gpu_n = gpu_arr::norm2(da);

    check(std::abs(gpu_n - cpu_n) < EPS,
          "norm2: GPU matches CPU (" + std::to_string(cpu_n) + ")");
}

void test_norm1_vs_cpu() {
    dims_t d{1, 2, 4};
    Array<float> ha(d);
    ha[0] = -3.f; ha[1] = 4.f; ha[2] = -1.f; ha[3] = 2.f;
    ha[4] =  5.f; ha[5] = -6.f; ha[6] = 7.f; ha[7] = -8.f;
    float cpu_n = cpu::norm1(ha);

    DeviceArray<float> da(ha);
    float gpu_n = gpu_arr::norm1(da);

    check(std::abs(gpu_n - cpu_n) < EPS,
          "norm1: GPU matches CPU (" + std::to_string(cpu_n) + ")");
}

void test_dot_vs_cpu() {
    dims_t d{1, 2, 4};
    auto ha = iota_array(d, 1.f);
    auto hb = iota_array(d, 2.f);
    float cpu_d = cpu::dot(ha, hb);

    DeviceArray<float> da(ha);
    DeviceArray<float> db(hb);
    float gpu_d = gpu_arr::dot(da, db);

    check(std::abs(gpu_d - cpu_d) < EPS,
          "dot: GPU matches CPU (" + std::to_string(cpu_d) + ")");
}

void test_dot_self_equals_norm2_sq() {
    dims_t d{1, 2, 4};
    auto ha = iota_array(d, 1.f);
    float n = cpu::norm2(ha);

    DeviceArray<float> da(ha);
    auto db = da.clone();
    float d_val = gpu_arr::dot(da, db);

    check(std::abs(d_val - n * n) < 1e-2f, "dot(x,x) == norm2(x)^2");
}

void test_to_complex_vs_cpu() {
    dims_t d{1, 2, 4};
    auto ha = iota_array(d, 1.f);
    auto cpu_c = cpu::to_complex(ha);  // Array<std::complex<float>>

    DeviceArray<float> da(ha);
    auto gpu_dc = gpu_arr::to_complex(da);
    auto gpu_c  = gpu_dc.to_host();   // Array<cuda::std::complex<float>>

    bool ok = true;
    for (size_t i = 0; i < gpu_c.size(); ++i) {
        ok = ok && (std::abs(gpu_c[i].real() - cpu_c[i].real()) < EPS);
        ok = ok && (std::abs(gpu_c[i].imag() - cpu_c[i].imag()) < EPS);
    }
    check(ok, "to_complex: GPU matches CPU");
}

void test_to_real_vs_cpu() {
    dims_t d{1, 2, 4};
    auto ha = iota_array(d, 1.f);

    // CPU path: real -> complex -> real
    auto cpu_c = cpu::to_complex(ha);
    auto cpu_r = cpu::to_real(cpu_c);

    // GPU path: real -> complex -> real
    DeviceArray<float> da(ha);
    auto gpu_dc = gpu_arr::to_complex(da);
    auto gpu_dr = gpu_arr::to_real(gpu_dc);
    auto gpu_r  = gpu_dr.to_host();

    bool ok = true;
    for (size_t i = 0; i < gpu_r.size(); ++i)
        ok = ok && (std::abs(gpu_r[i] - cpu_r[i]) < EPS);
    check(ok, "to_real: GPU matches CPU");
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

int main() {
    std::cout << "=== device_array_ops tests ===\n\n";

    std::cout << "-- GPU-only operations --\n";
    test_scale_inplace();
    test_scale();
    test_axpy();
    test_axpy_zero_alpha();
    test_xpay();
    test_subtract();
    test_subtract_inplace();
    test_subtract_self_is_zero();
    test_add();
    test_add_inplace();
    test_multiply();
    test_multiply_inplace();
    test_abs();
    test_sqrt();

    std::cout << "\n-- GPU vs CPU cross-validation --\n";
    test_max_vs_cpu();
    test_min_vs_cpu();
    test_norm2_vs_cpu();
    test_norm1_vs_cpu();
    test_dot_vs_cpu();
    test_dot_self_equals_norm2_sq();
    test_to_complex_vs_cpu();
    test_to_real_vs_cpu();

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
