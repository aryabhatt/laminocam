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

// 2D slice-by-slice (vertical rotation axis) tomography reconstruction,
// driven by the same TOML config format src/recon.cpp uses for the 3D
// laminography path (see include/config.h -- read_toml_file,
// parse_input_datasets, parse_recon_params, parse_output_params are all
// reused unmodified). Input is usually an HDF5 file (tomocam::h5::
// read_images/read_angles, include/hdfread.h, dispatched automatically by
// parse_input_datasets based on the '.h5'/'.hdf5' extension); TIFF +
// angles-file input works too via the same dispatch.
//
// TODO: hard-coded to tomography (vertical-axis) mode until config.h grows
// a laminography/tomography flag -- this executable only ever reconstructs
// with cpu::PolarGrid2D, so every input dataset's gamma/beta must be ~0
// (checked below; recon_lamino remains the entry point for laminography).
// Revisit alongside a 2D density-compensation preconditioner once results
// need to look physically sharp -- see tests/test_mbir2d_sanity.cpp for why
// plain CG/Split-Bregman without one plateaus around 35-40% relative error
// on a sharp phantom.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "config.h"
#include "mbir2d_prep.h"
#include "optimize.h"
#include "polar_grid2d.h"
#include "projection.h"
#include "timer.h"

#ifdef USE_GPU
#include "gpu/cufft_plan_cache.h"
#include "gpu/cufinufft_plan_cache.h"
#include "gpu/tomocam.h"
#endif

using namespace tomocam;

// CPU reconstruction of the prepared sinogram {n_slices, n_angles, n_cols}.
static Array<float> cpu_mbir2d(const Array<float> &sinogram,
                               const std::vector<float> &theta,
                               const ReconParams &params) {
    const dims_t recon_dims = params.recon_dims;
    cpu::PolarGrid2D<float> pg(theta, sinogram.ncols());

    std::cout << "Building 2D Toeplitz PSF ...\n";
    cpu::PointSpreadFunction2D<float> psf(pg, recon_dims.n3);
    opt::Function<float> A = [&psf](const Array<float> &x) {
        return sysmat2d(x, psf);
    };

    std::cout << "Computing backprojection (right-hand side) ...\n";
    auto rhs = backproj2d(sinogram, pg, recon_dims);

    auto x0 = Array<float>::zeros(recon_dims);
    Array<float> recon;
    switch (params.regularizer) {
        case Regularizer::UNCONSTRAINED: {
            std::cout << "Starting unconstrained reconstruction with CG ...\n";
            recon = opt::cgsolver<float>(A, rhs, x0, params.maxIters, params.tol,
                                         params.xtol);
            break;
        }
        case Regularizer::SPLIT_BREGMAN: {
            std::cout << "Starting MBIR with Split-Bregman method ...\n";
            recon = opt::split_bregman<float>(A, rhs, x0, params.lambda, params.mu,
                                              params.maxIters, params.innerIters,
                                              params.tol, params.xtol);
            break;
        }
        default: throw std::invalid_argument("Unsupported optimizer type");
    }
    return recon;
}

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cerr << std::format("Usage: {} <input.toml>\n", argv[0]);
        std::cerr << "Writing an example input file to config_template.toml.\n";
        tomocam::dump_config(tomocam::TOMO_CONFIG_TEMPLATE, "config_template.toml");
        return 1;
    }

    toml::table config = tomocam::read_toml_file(argv[1]);
    auto datasets = tomocam::parse_input_datasets<float>(
        config, std::filesystem::path(argv[1]).parent_path());
    auto params = tomocam::parse_recon_params(config);
    auto output = tomocam::parse_output_params(config);

    if (datasets.empty()) {
        throw std::runtime_error("mbir2d: no [[input]] datasets found");
    }

    dims_t recon_dims = params.recon_dims;
    if (recon_dims.n2 != recon_dims.n3) {
        throw std::runtime_error(std::format("mbir2d: recon_dims must be square "
                                             "in-plane (n2 == n3), got [{}, {}, {}]",
                                             recon_dims.n1, recon_dims.n2,
                                             recon_dims.n3));
    }

    // This executable is hard-coded to the tomography (vertical rotation
    // axis) path: every dataset's gamma/beta must be ~0. Per-projection
    // shifts are read but not yet supported (forward2d/backproj2d have no
    // shifted overload), so warn rather than fail.
    constexpr float TILT_EPS = 1e-6f;
    for (size_t j = 0; j < datasets.size(); ++j) {
        if (std::abs(datasets[j].gamma) > TILT_EPS ||
            std::abs(datasets[j].beta) > TILT_EPS) {
            throw std::runtime_error(std::format(
                "mbir2d: dataset {} has gamma={:.4f}, beta={:.4f} (radians) -- "
                "this executable only supports vertical-axis tomography "
                "(gamma=beta=0). Use recon_lamino for laminography.",
                j, datasets[j].gamma, datasets[j].beta));
        }
        for (auto &s : datasets[j].shifts) {
            if (std::abs(s[0]) > TILT_EPS || std::abs(s[1]) > TILT_EPS) {
                std::cerr << std::format(
                    "Warning: dataset {} has non-zero per-projection shifts; "
                    "mbir2d does not yet support center-of-rotation shifts "
                    "and will ignore them.\n",
                    j);
                break;
            }
        }
    }

    auto [sinogram, theta] = prepare_sinogram(datasets);

    if (recon_dims.n1 != sinogram.nslices()) {
        std::cerr << std::format(
            "Warning: recon_dims[0]={} does not match the input data's row "
            "count ({}); reconstructing {} slices from {}-row projections.\n",
            recon_dims.n1, sinogram.nslices(), recon_dims.n1, sinogram.nslices());
    }

    params.print(std::cout);
    std::cout << std::endl;

    tomocam::Timer t0;
    t0.start();

    Array<float> recon;
#ifdef USE_GPU
    try {
        std::cout << "Running reconstruction on GPU...\n";
        recon = tomocam::gpu::MBIR2D<float>(sinogram, theta, params);
    } catch (const std::exception &e) {
        std::cerr << std::format(
            "GPU reconstruction failed ({}); falling back to CPU\n", e.what());
        cudaGetLastError(); // reset any sticky CUDA error state
        recon = cpu_mbir2d(sinogram, theta, params);
    }
    tomocam::gpu::nufft::plans::cache<float>.clear();
    tomocam::gpu::fft::cache::plans<float>.clear();
    cudaDeviceReset();
#else
    std::cout << "Running reconstruction on CPU...\n";
    recon = cpu_mbir2d(sinogram, theta, params);
#endif
    t0.stop();
    std::cout << std::format("Reconstruction completed in {:.2f} seconds.\n",
                             t0.seconds());

    auto base_dir = std::filesystem::path(output.filepath).parent_path();
    if (!base_dir.empty() && !std::filesystem::exists(base_dir)) {
        std::filesystem::create_directories(base_dir);
    }
    if (output.has_format("tiff")) {
        tomocam::tiff::write(output.filepath, recon);
        std::cout << std::format("Saved reconstruction to {}\n", output.filepath);
    }

    return 0;
}
