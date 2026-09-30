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
#include "optimize.h"
#include "polar_grid2d.h"
#include "projection.h"
#include "timer.h"

using namespace tomocam;

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

    // Stack all datasets' projections and angles (mirrors the stacking
    // block in src/mbir.cpp's MBIR()).
    size_t n_projs = 0;
    for (auto &ds : datasets) n_projs += ds.angles.size();
    dims_t proj_dims = datasets[0].projs.dims();
    proj_dims.n1 = n_projs;

    std::vector<float> theta;
    theta.reserve(n_projs);
    Array<float> stacked_projs(proj_dims);
    {
        size_t offset = 0;
        for (auto &ds : datasets) {
            std::copy(ds.projs.data(), ds.projs.data() + ds.projs.size(),
                      stacked_projs.data() + offset);
            offset += ds.projs.size();
            for (auto &a : ds.angles) theta.push_back(a);
        }
    }
    float max_val = array::max(stacked_projs);
    if (max_val > 0) { stacked_projs /= max_val; }

    if (recon_dims.n1 != stacked_projs.nrows()) {
        std::cerr << std::format(
            "Warning: recon_dims[0]={} does not match the input data's row "
            "count ({}); reconstructing {} slices from {}-row projections.\n",
            recon_dims.n1, stacked_projs.nrows(), recon_dims.n1,
            stacked_projs.nrows());
    }

    // Input data is {n_angles, n_rows, n_cols}; forward2d/backproj2d expect
    // a sinogram shaped {n_slices, n_angles, n_cols} -- swap the first two
    // axes (n_rows becomes the leading n_slices axis).
    auto sinogram = array::transpose(stacked_projs, {1, 0, 2});

    cpu::PolarGrid2D<float> pg(theta, sinogram.ncols());

    std::cout << "Building 2D Toeplitz PSF ...\n";
    cpu::PointSpreadFunction2D<float> psf(pg, recon_dims.n3);
    opt::Function<float> A = [&psf](const Array<float> &x) {
        return sysmat2d(x, psf);
    };

    std::cout << "Computing backprojection (right-hand side) ...\n";
    auto rhs = backproj2d(sinogram, pg, recon_dims);

    params.print(std::cout);
    std::cout << std::endl;

    tomocam::Timer t0;
    t0.start();

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
