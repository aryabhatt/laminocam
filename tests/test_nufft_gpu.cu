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
#include <complex>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/nufft.h"
#include "gpu/polar_grid.h"
#include "nufft.h"
#include "polar_grid.h"

using namespace tomocam;

// Upload host complex array to device (std::complex and cuda::std::complex
// share the same binary layout — two contiguous real-valued components).
static gpu::DeviceArray<cuda::std::complex<double>>
upload_complex(const Array<std::complex<double>> &h) {
    // Reinterpret via raw bytes: same memory layout, safe for POD-like types.
    Array<cuda::std::complex<double>> tmp(h.dims());
    static_assert(sizeof(std::complex<double>) == sizeof(cuda::std::complex<double>),
                  "complex layout mismatch");
    std::memcpy(tmp.begin(), h.begin(), h.size() * sizeof(std::complex<double>));
    return gpu::DeviceArray<cuda::std::complex<double>>(tmp);
}

// Download device complex array to host std::complex array.
static Array<std::complex<double>>
download_complex(const gpu::DeviceArray<cuda::std::complex<double>> &d) {
    auto tmp = d.to_host(); // Array<cuda::std::complex<double>>
    Array<std::complex<double>> h(tmp.dims());
    std::memcpy(h.begin(), tmp.begin(), tmp.size() * sizeof(std::complex<double>));
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// Type-1: non-uniform → uniform
// ─────────────────────────────────────────────────────────────────────────────
bool test_nufft3d1_cpu_vs_gpu(size_t nangles, size_t nrows, size_t ncols, size_t nz,
                              size_t ny, size_t nx) {
    std::cout << "nufft3d1 CPU vs GPU: grid=(" << nangles << "," << nrows << ","
              << ncols << ") vol=(" << nz << "," << ny << "," << nx << ") ... "
              << std::flush;

    // Equal-spaced angles in [0, π)
    std::vector<double> theta(nangles);
    for (size_t i = 0; i < nangles; ++i)
        theta[i] = M_PI * static_cast<double>(i) / static_cast<double>(nangles);
    std::vector<double> gamma(nangles, 0.0);

    // CPU and GPU polar grids from identical angles
    cpu::PolarGrid<double> cpu_pg(theta, gamma, nrows, ncols);
    gpu::PolarGrid<double> gpu_pg(theta, gamma, nrows, ncols);

    // Random real input → complex source values
    dims_t grid_dims{nangles, nrows, ncols};
    Array<double> h_real(grid_dims);
    for (size_t i = 0; i < h_real.size(); ++i)
        h_real[i] = static_cast<double>(rand()) / RAND_MAX;
    auto h_cz = array::to_complex(h_real);

    // ── CPU ──────────────────────────────────────────────────────────────────
    dims_t vol_dims{nz, ny, nx};
    Array<std::complex<double>> h_fz(vol_dims);
    nufft::nufft3d1(h_cz, h_fz, cpu_pg);

    // ── GPU ──────────────────────────────────────────────────────────────────
    auto d_cz = upload_complex(h_cz);
    gpu::DeviceArray<cuda::std::complex<double>> d_fz(vol_dims);
    gpu::nufft::nufft3d1(d_cz, d_fz, gpu_pg);
    auto h_fz_gpu = download_complex(d_fz);

    // ── Compare ──────────────────────────────────────────────────────────────
    double max_abs_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < h_fz.size(); ++i) {
        double diff = std::abs(h_fz[i] - h_fz_gpu[i]);
        double ref = std::abs(h_fz[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        if (ref > max_ref) max_ref = ref;
    }
    double rel = (max_ref > 0.0) ? max_abs_diff / max_ref : max_abs_diff;

    // FINUFFT and cuFINUFFT both use tol=1e-14; allow generous margin for
    // CPU/GPU floating-point differences (different NUFFT back-ends).
    const double tol = 1e-6;
    if (rel < tol) {
        std::cout << "PASSED (rel=" << rel << ")\n";
        return true;
    }
    std::cout << "FAILED (max_diff=" << max_abs_diff << " rel=" << rel << ")\n";
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Type-2: uniform → non-uniform
// ─────────────────────────────────────────────────────────────────────────────
bool test_nufft3d2_cpu_vs_gpu(size_t nangles, size_t nrows, size_t ncols, size_t nz,
                              size_t ny, size_t nx) {
    std::cout << "nufft3d2 CPU vs GPU: grid=(" << nangles << "," << nrows << ","
              << ncols << ") vol=(" << nz << "," << ny << "," << nx << ") ... "
              << std::flush;

    std::vector<double> theta(nangles);
    for (size_t i = 0; i < nangles; ++i)
        theta[i] = M_PI * static_cast<double>(i) / static_cast<double>(nangles);
    std::vector<double> gamma(nangles, 0.0);

    cpu::PolarGrid<double> cpu_pg(theta, gamma, nrows, ncols);
    gpu::PolarGrid<double> gpu_pg(theta, gamma, nrows, ncols);

    // Random complex uniform-grid input (the Fourier volume)
    dims_t vol_dims{nz, ny, nx};
    Array<double> h_real(vol_dims), h_imag(vol_dims);
    for (size_t i = 0; i < h_real.size(); ++i) {
        h_real[i] = static_cast<double>(rand()) / RAND_MAX;
        h_imag[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    Array<std::complex<double>> h_fz(vol_dims);
    for (size_t i = 0; i < h_fz.size(); ++i) h_fz[i] = {h_real[i], h_imag[i]};

    // ── CPU ──────────────────────────────────────────────────────────────────
    dims_t grid_dims{nangles, nrows, ncols};
    Array<std::complex<double>> h_cz(grid_dims);
    nufft::nufft3d2(h_cz, h_fz, cpu_pg);

    // ── GPU ──────────────────────────────────────────────────────────────────
    auto d_fz = upload_complex(h_fz);
    gpu::DeviceArray<cuda::std::complex<double>> d_cz(grid_dims);
    gpu::nufft::nufft3d2(d_cz, d_fz, gpu_pg);
    auto h_cz_gpu = download_complex(d_cz);

    // ── Compare ──────────────────────────────────────────────────────────────
    double max_abs_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < h_cz.size(); ++i) {
        double diff = std::abs(h_cz[i] - h_cz_gpu[i]);
        double ref = std::abs(h_cz[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        if (ref > max_ref) max_ref = ref;
    }
    double rel = (max_ref > 0.0) ? max_abs_diff / max_ref : max_abs_diff;

    const double tol = 1e-6;
    if (rel < tol) {
        std::cout << "PASSED (rel=" << rel << ")\n";
        return true;
    }
    std::cout << "FAILED (max_diff=" << max_abs_diff << " rel=" << rel << ")\n";
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    srand(42);

    std::cout << "\n====== GPU NUFFT Correctness Tests ======\n";
    std::cout << "Comparing cpu::nufft and gpu::nufft outputs\n\n";

    int passed = 0, failed = 0;
    auto tally = [&](bool ok) { ok ? ++passed : ++failed; };

    // ── Type-1 tests ─────────────────────────────────────────────────────────
    std::cout << "--- Type-1 (non-uniform → uniform) CPU vs GPU ---\n";
    tally(test_nufft3d1_cpu_vs_gpu(8, 16, 16, 16, 16, 16));
    tally(test_nufft3d1_cpu_vs_gpu(12, 20, 18, 20, 18, 16));
    tally(test_nufft3d1_cpu_vs_gpu(6, 12, 14, 14, 12, 10));

    // ── Type-2 tests ─────────────────────────────────────────────────────────
    std::cout << "\n--- Type-2 (uniform → non-uniform) CPU vs GPU ---\n";
    tally(test_nufft3d2_cpu_vs_gpu(8, 16, 16, 16, 16, 16));
    tally(test_nufft3d2_cpu_vs_gpu(12, 20, 18, 20, 18, 16));
    tally(test_nufft3d2_cpu_vs_gpu(6, 12, 14, 14, 12, 10));

    std::cout << "\nResults: " << passed << " passed, " << failed << " failed\n";

    // Flush the plan cache while the CUDA context is still live so that the
    // global-static cache destructor doesn't fire cufinufft_destroy() after
    // CUDA has already unloaded (which would produce cudaErrorCudartUnloading).
    gpu::nufft::plans::cache<double>.clear();
    cudaDeviceReset();

    return (failed == 0) ? 0 : 1;
}
