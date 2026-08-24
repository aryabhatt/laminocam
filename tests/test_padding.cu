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
#include "padding.h"
#include "gpu/device_array.h"
#include "gpu/padding.h"

using namespace tomocam;
using tomocam::gpu::DeviceArray;
namespace gpu = tomocam::gpu;

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

static constexpr float EPS = 1e-5f;

static Array<float> iota_array(dims_t d, float start = 1.f) {
    Array<float> a(d);
    for (size_t i = 0; i < a.size(); ++i) a[i] = start + static_cast<float>(i);
    return a;
}

static bool arrays_match(const Array<float> &a, const Array<float> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > EPS) return false;
    return true;
}

// -------------------------------------------------------------------------
// pad2d: GPU vs CPU for all three padding types
// -------------------------------------------------------------------------

void test_pad2d(PadType type, const std::string &label) {
    dims_t in_dims{2, 3, 5};
    auto h_in = iota_array(in_dims);
    DeviceArray<float> d_in(h_in);

    auto cpu_out = pad2d(h_in, 2.f, type);
    auto gpu_out = gpu::pad2d(d_in, 2.f, type).to_host();

    check(arrays_match(cpu_out, gpu_out), "pad2d " + label + ": GPU matches CPU");
}

// -------------------------------------------------------------------------
// pad3d: GPU vs CPU for all three padding types
// -------------------------------------------------------------------------

void test_pad3d(PadType type, const std::string &label) {
    dims_t in_dims{2, 3, 5};
    dims_t out_dims{3, 7, 11};
    auto h_in = iota_array(in_dims);
    DeviceArray<float> d_in(h_in);

    auto cpu_out = pad3d(h_in, out_dims, type);
    auto gpu_out = gpu::pad3d(d_in, out_dims, type).to_host();

    check(arrays_match(cpu_out, gpu_out), "pad3d " + label + ": GPU matches CPU");
}

// -------------------------------------------------------------------------
// crop3d (PadType): GPU vs CPU for all three types
// -------------------------------------------------------------------------

void test_crop3d(PadType type, const std::string &label) {
    dims_t in_dims{3, 7, 11};
    dims_t out_dims{2, 3, 5};
    auto h_in = iota_array(in_dims);
    DeviceArray<float> d_in(h_in);

    auto cpu_out = crop3d(h_in, out_dims, type);
    auto gpu_out = gpu::crop3d(d_in, out_dims, type).to_host();

    check(arrays_match(cpu_out, gpu_out), "crop3d " + label + ": GPU matches CPU");
}

// -------------------------------------------------------------------------
// crop3d (explicit offset): GPU vs CPU
// -------------------------------------------------------------------------

void test_crop3d_offset() {
    dims_t in_dims{3, 7, 11};
    dims_t out_dims{2, 3, 5};
    dims_t offset{1, 2, 3};
    auto h_in = iota_array(in_dims);
    DeviceArray<float> d_in(h_in);

    auto cpu_out = crop3d(h_in, out_dims, offset);
    auto gpu_out = gpu::crop3d(d_in, out_dims, offset).to_host();

    check(arrays_match(cpu_out, gpu_out), "crop3d (explicit offset): GPU matches CPU");
}

// -------------------------------------------------------------------------
// Roundtrip: pad3d -> crop3d recovers original
//
// Convention: pad RIGHT places data at start (offset=0), crop LEFT takes from
// start (offset=0); pad LEFT places data at end, crop RIGHT takes from end.
// pad/crop SYMMETRIC are self-inverse.
// -------------------------------------------------------------------------

void test_roundtrip_pad3d(PadType pad_type, PadType crop_type, const std::string &label) {
    dims_t in_dims{2, 3, 5};
    dims_t pad_dims{3, 7, 11};
    auto h_in = iota_array(in_dims);
    DeviceArray<float> d_in(h_in);

    auto d_padded  = gpu::pad3d(d_in, pad_dims, pad_type);
    auto d_cropped = gpu::crop3d(d_padded, in_dims, crop_type);
    auto gpu_out   = d_cropped.to_host();

    bool ok = true;
    for (size_t i = 0; i < h_in.size(); ++i)
        ok = ok && (std::abs(gpu_out[i] - h_in[i]) < EPS);
    check(ok, "roundtrip pad3d(" + label + ")->crop3d recovers original");
}

// Roundtrip for pad2d: pad then crop back to original dims.
// pad2d SYMMETRIC -> crop3d SYMMETRIC; RIGHT -> LEFT; LEFT -> RIGHT.
void test_roundtrip_pad2d(PadType pad_type, PadType crop_type, const std::string &label) {
    dims_t in_dims{2, 3, 5};
    auto h_in = iota_array(in_dims);
    DeviceArray<float> d_in(h_in);

    auto d_padded  = gpu::pad2d(d_in, 2.f, pad_type);
    auto d_cropped = gpu::crop3d(d_padded, in_dims, crop_type);
    auto gpu_out   = d_cropped.to_host();

    bool ok = true;
    for (size_t i = 0; i < h_in.size(); ++i)
        ok = ok && (std::abs(gpu_out[i] - h_in[i]) < EPS);
    check(ok, "roundtrip pad2d(" + label + ")->crop3d recovers original");
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

int main() {
    std::cout << "=== padding tests ===\n\n";

    std::cout << "-- pad2d: GPU vs CPU --\n";
    test_pad2d(PadType::SYMMETRIC, "SYMMETRIC");
    test_pad2d(PadType::LEFT,      "LEFT");
    test_pad2d(PadType::RIGHT,     "RIGHT");

    std::cout << "\n-- pad3d: GPU vs CPU --\n";
    test_pad3d(PadType::SYMMETRIC, "SYMMETRIC");
    test_pad3d(PadType::LEFT,      "LEFT");
    test_pad3d(PadType::RIGHT,     "RIGHT");

    std::cout << "\n-- crop3d (PadType): GPU vs CPU --\n";
    test_crop3d(PadType::SYMMETRIC, "SYMMETRIC");
    test_crop3d(PadType::LEFT,      "LEFT");
    test_crop3d(PadType::RIGHT,     "RIGHT");

    std::cout << "\n-- crop3d (explicit offset): GPU vs CPU --\n";
    test_crop3d_offset();

    std::cout << "\n-- roundtrip pad3d -> crop3d --\n";
    test_roundtrip_pad3d(PadType::SYMMETRIC, PadType::SYMMETRIC, "SYMMETRIC");
    test_roundtrip_pad3d(PadType::RIGHT,     PadType::LEFT,      "RIGHT->LEFT");
    test_roundtrip_pad3d(PadType::LEFT,      PadType::RIGHT,     "LEFT->RIGHT");

    std::cout << "\n-- roundtrip pad2d -> crop3d --\n";
    test_roundtrip_pad2d(PadType::SYMMETRIC, PadType::SYMMETRIC, "SYMMETRIC");
    test_roundtrip_pad2d(PadType::RIGHT,     PadType::LEFT,      "RIGHT->LEFT");
    test_roundtrip_pad2d(PadType::LEFT,      PadType::RIGHT,     "LEFT->RIGHT");

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
