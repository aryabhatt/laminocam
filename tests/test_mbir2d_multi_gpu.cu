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

// Multi-GPU streamed reconstruction checks:
//   1. Halo exactness: grad / neg_divergence (halo 1) and neg_laplacian (halo 2)
//      evaluated on a chunk with halo slices reproduce the interior of the
//      full-volume result.
//   2. ChunkExecutor: every chunk is processed exactly once across the GPUs, and
//      an exception thrown by a worker step is rethrown by run() (this is what
//      lets recon_tomo fall back to the CPU) without hanging the other workers.
//   3. gpu::MBIR2D_streamed (chunks of a few slices, on one GPU and on every GPU)
//      agrees with the single-device solver for CG and Split-Bregman. The math
//      is identical, so only floating-point summation order differs.

#include <atomic>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <thread>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "gpu/cufft_plan_cache.h"
#include "gpu/cufinufft_plan_cache.h"
#include "gpu/device_array.h"
#include "gpu/finitediff.h"
#include "gpu/multigpu.h"
#include "gpu/tomocam.h"
#include "polar_grid2d.h"
#include "projection.h"
#include "recon_params.h"

using namespace tomocam;
using tomocam::gpu::DeviceArray;
using tomocam::gpu::SliceChunk;

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

static bool report(const std::string &name, float err, float tol) {
    bool ok = err < tol;
    std::cout << std::format("  {}  {}  (rel_err = {:.4e}, tol = {:.1e})\n",
                             ok ? "PASS" : "FAIL", name, err, tol);
    return ok;
}

// Apply `op` to every chunk (with halo) of `x` and stitch the interiors back.
template <typename Op>
static Array<float> apply_chunked(const Array<float> &x, size_t chunk_slices,
                                  size_t halo, Op op) {
    Array<float> out(x.dims());
    for (const auto &c : gpu::make_chunks(x.nslices(), chunk_slices, halo)) {
        DeviceArray<float> d_in = gpu::upload_chunk(x, c);
        DeviceArray<float> d_out = op(d_in);
        gpu::download_interior(d_out, out, c);
    }
    return out;
}

static bool test_halo_stencils() {
    const dims_t dims{11, 17, 19};
    auto x = Array<float>::random(dims);
    DeviceArray<float> d_x(x);
    bool ok = true;

    // neg_laplacian: needs halo 2 (central difference of a central difference)
    auto lap_full = gpu::opt::neg_laplacian(d_x).to_host();
    auto neg_lap = [](const DeviceArray<float> &u) {
        return gpu::opt::neg_laplacian(u);
    };
    ok &= report("neg_laplacian, halo 2, chunks of 3",
                 rel_err(apply_chunked(x, 3, 2, neg_lap), lap_full), 1e-6f);
    ok &= report("neg_laplacian, halo 2, chunks of 1",
                 rel_err(apply_chunked(x, 1, 2, neg_lap), lap_full), 1e-6f);
    // with only one halo slice the laplacian is NOT exact: confirms halo 2 is
    // really needed and the test above is sensitive to it
    float short_halo = rel_err(apply_chunked(x, 3, 1, neg_lap), lap_full);
    bool sensitive = short_halo > 1e-3f;
    std::cout << std::format(
        "  {}  neg_laplacian, halo 1 is inexact (err = {:.2e})\n",
        sensitive ? "PASS" : "FAIL", short_halo);
    ok &= sensitive;

    // slice-axis grad component: needs halo 1
    auto gz_full = gpu::opt::grad(d_x)[2].to_host();
    auto gz = [](const DeviceArray<float> &u) {
        return std::move(gpu::opt::grad(u)[2]);
    };
    ok &= report("grad (slice axis), halo 1, chunks of 4",
                 rel_err(apply_chunked(x, 4, 1, gz), gz_full), 1e-6f);

    // neg_divergence of a vector field: needs halo 1
    auto make_field = [](const DeviceArray<float> &u) {
        std::array<DeviceArray<float>, 3> f;
        for (size_t k = 0; k < 3; ++k) f[k] = u.clone();
        return f;
    };
    auto div_full = gpu::opt::neg_divergence(make_field(d_x)).to_host();
    auto neg_div = [&](const DeviceArray<float> &u) {
        return gpu::opt::neg_divergence(make_field(u));
    };
    ok &= report("neg_divergence, halo 1, chunks of 2",
                 rel_err(apply_chunked(x, 2, 1, neg_div), div_full), 1e-6f);
    return ok;
}

static bool test_executor() {
    auto theta = make_theta(20);
    const int ngpus = gpu::resolve_num_gpus(0);
    gpu::ChunkExecutor<float> exec(theta, 33, 33, ngpus);
    const auto chunks = gpu::make_chunks(40, 1, 0); // 40 chunks, dynamic queue
    bool ok = true;

    // every chunk is visited exactly once
    std::vector<std::atomic<int>> visits(chunks.size());
    std::vector<std::atomic<int>> per_device(ngpus);
    exec.run(chunks,
             [&](gpu::DeviceContext<float> &ctx, size_t i, const SliceChunk &) {
                 int cur = -1;
                 SAFE_CALL(cudaGetDevice(&cur));
                 if (cur != ctx.id)
                     throw std::runtime_error("worker on the wrong device");
                 visits[i]++;
                 per_device[ctx.id]++;
                 // give the other workers time to start and pick up chunks
                 std::this_thread::sleep_for(std::chrono::milliseconds(10));
             });
    bool once = true;
    for (auto &v : visits) once &= (v == 1);
    for (int g = 0; g < ngpus; ++g) once &= (per_device[g] > 0); // all GPUs worked
    std::string split;
    for (int g = 0; g < ngpus; ++g)
        split += std::format(" gpu{}={}", g, per_device[g].load());
    std::cout << std::format(
        "  {}  executor visits each chunk once, on its own device ({})\n",
        once ? "PASS" : "FAIL", split);
    ok &= once;

    // an exception in one step reaches the caller
    bool caught = false;
    try {
        exec.run(chunks,
                 [&](gpu::DeviceContext<float> &, size_t i, const SliceChunk &) {
                     if (i == 7) throw std::runtime_error("boom");
                 });
    } catch (const std::runtime_error &e) {
        caught = (std::string(e.what()) == "boom");
    }
    std::cout << std::format("  {}  executor rethrows a worker exception\n",
                             caught ? "PASS" : "FAIL");
    ok &= caught;

    // and the executor is still usable afterwards
    std::atomic<int> count{0};
    exec.run(chunks, [&](gpu::DeviceContext<float> &, size_t, const SliceChunk &) {
        count++;
    });
    bool reusable = (count == static_cast<int>(chunks.size()));
    std::cout << std::format("  {}  executor usable after an error\n",
                             reusable ? "PASS" : "FAIL");
    return ok && reusable;
}

// streamed vs single-device on the same sinogram
static bool test_streamed(Regularizer reg, size_t ngpus, size_t chunk,
                          const std::string &name) {
    const size_t Nz = 7, N = 48, NTHETA = 60;
    auto theta = make_theta(NTHETA);
    cpu::PolarGrid2D<float> pg(theta, N);
    auto sinogram = forward2d(make_phantom(Nz, N), pg);

    ReconParams params;
    params.regularizer = reg;
    params.recon_dims = {Nz, N, N};
    params.maxIters = (reg == Regularizer::SPLIT_BREGMAN) ? 4 : 15;
    params.innerIters = 3;
    params.lambda = 0.1f;
    params.mu = 5.0f;
    params.tol = 1e-7f;
    params.xtol = 1e-7f;

    // reference: single device, no streaming
    params.numGpus = 1;
    auto ref = gpu::MBIR2D<float>(sinogram, theta, params);

    params.numGpus = ngpus;
    params.chunkSlices = chunk;
    params.streamed = true;
    auto out = gpu::MBIR2D<float>(sinogram, theta, params);

    float tol = (reg == Regularizer::SPLIT_BREGMAN) ? 5e-3f : 1e-3f;
    return report(name, rel_err(out, ref), tol);
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::cout << "SKIP: no CUDA device\n";
        return 0;
    }
    std::cout << std::format("=== multi-GPU streamed reconstruction ({} device(s)) "
                             "===\n",
                             ndev);

    int failed = 0;
    failed += !test_halo_stencils();
    failed += !test_executor();

    // chunked on a single GPU: exercises chunking + halos independent of devices
    failed += !test_streamed(Regularizer::UNCONSTRAINED, 1, 2,
                             "CG, 1 GPU, chunks of 2 vs single-device");
    failed += !test_streamed(Regularizer::SPLIT_BREGMAN, 1, 2,
                             "Split-Bregman, 1 GPU, chunks of 2 vs single-device");

    if (ndev >= 2) {
        failed += !test_streamed(
            Regularizer::UNCONSTRAINED, 0, 2,
            std::format("CG, {} GPUs, chunks of 2 vs single-device", ndev));
        failed += !test_streamed(
            Regularizer::SPLIT_BREGMAN, 0, 2,
            std::format("Split-Bregman, {} GPUs, chunks of 2 vs single-device",
                        ndev));
        // auto chunk size: one chunk per GPU
        failed += !test_streamed(
            Regularizer::SPLIT_BREGMAN, 0, 0,
            std::format("Split-Bregman, {} GPUs, auto chunks vs single-device",
                        ndev));
    } else {
        std::cout << "  SKIP  multi-device cases (only one GPU present)\n";
    }

    std::cout << std::format("\nResults: {} failed\n", failed);

    gpu::nufft::plans::cache<float>.clear();
    gpu::fft::cache::plans<float>.clear();
    return failed == 0 ? 0 : 1;
}
