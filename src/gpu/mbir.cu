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
#include <cassert>
#include <format>
#include <functional>
#include <iostream>

#include "array.h"
#include "array_ops.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/gpu_opt.h"
#include "gpu/padding.h"
#include "gpu/projection.h"
#include "gpu/toeplitz.h"
#include "padding.h"
#include "projection.h"
#include "recon_params.h"

namespace tomocam::gpu {

    template <typename T>
    Array<T> MBIR(const std::vector<Dataset_t<T>> &datasets,
                  const ReconParams &params) {

        // adjust reconstruction dimensions for padding
        T padfac = static_cast<T>(params.PAD_FACTOR);
        dims_t recon_dims = params.recon_dims;

        recon_dims.n1 = static_cast<size_t>(recon_dims.n1 * padfac);
        if (recon_dims.n1 % 2 == 0) {
            recon_dims.n1 -= 1; // make sure n1 is odd
        }
        recon_dims.n2 = static_cast<size_t>(recon_dims.n2 * padfac);
        if (recon_dims.n2 % 2 == 0) {
            recon_dims.n2 -= 1; // make sure n2 is odd
        }
        recon_dims.n3 = static_cast<size_t>(recon_dims.n3 * padfac);
        if (recon_dims.n3 % 2 == 0) {
            recon_dims.n3 -= 1; // make sure n3 is odd
        }

        //  find the maximum value across all datasets to normalize projections
        T scale = 0;
        size_t n_projs = 0;
        size_t nrows = datasets[0].projs.nrows();
        size_t ncols = datasets[0].projs.ncols();
        for (const auto &[projs, angles, gamma, beta, shifts_ds] : datasets) {
            scale = std::max(scale, tomocam::array::max(projs));
            n_projs += angles.size();
        }
        // compute padded dimensions for stacking
        size_t padded_nrows = static_cast<size_t>(nrows * params.PAD_FACTOR);
        if (padded_nrows % 2 == 0) {
            padded_nrows -= 1; // make sure nrows is odd
        }
        size_t padded_ncols = static_cast<size_t>(ncols * params.PAD_FACTOR);
        if (padded_ncols % 2 == 0) {
            padded_ncols -= 1; // make sure ncols is odd
        }
        dims_t proj_dims = {n_projs, padded_nrows, padded_ncols};

        // combine datasets and build stacked projections on CPU
        std::vector<T> theta;
        std::vector<T> gamma;
        DeviceArray<T> yT;
        PointSpreadFunction<T> psf;
        {
            Array<T> stacked_projs(proj_dims);
            size_t offset = 0;
            for (const auto &[projs, angles, gamma_ref, beta_ref, shifts_ds] : datasets) {
                Array<T> y = projs / scale;
                y = pad2d(y, padfac, PadType::SYMMETRIC);
                if (y.nrows() != padded_nrows || y.ncols() != padded_ncols) {
                    throw std::runtime_error(
                        std::format("Padded projections dimensions {} do not match "
                                    "expected dimensions {}",
                                    y.dims().to_string(), proj_dims.to_string()));
                }
                std::copy(y.data(), y.data() + y.size(),
                          stacked_projs.data() + offset);
                offset += y.size();
                theta.insert(theta.end(), angles.begin(), angles.end());
                for (size_t i = 0; i < angles.size(); ++i) {
                    gamma.push_back(gamma_ref);
                }
            }

            // try GPU NUFFT first; fall back to CPU if device memory is insufficient
            try {
                auto gpu_grid =
                    PolarGrid<T>(theta, gamma, padded_nrows, padded_ncols);
                DeviceArray<T> d_p(stacked_projs);
                yT = backproj(d_p, gpu_grid, recon_dims);
                psf = PointSpreadFunction<T>(gpu_grid, recon_dims);
            } catch (const std::exception &e) {
                std::cerr << std::format(
                    "GPU backprojection/PSF failed ({}); falling back to CPU\n",
                    e.what());
                cudaGetLastError(); // reset any sticky CUDA error state
                auto cpu_grid = tomocam::cpu::PolarGrid<T>(
                    theta, gamma, padded_nrows, padded_ncols);
                yT = DeviceArray<T>(
                    tomocam::backproj(stacked_projs, cpu_grid, recon_dims));
                auto cpu_psf =
                    tomocam::cpu::PointSpreadFunction<T>(cpu_grid, recon_dims);
                psf = PointSpreadFunction<T>(cpu_psf.dims(), cpu_psf.kernel_hat());
            }
        }

        // initialize the reconstruction with zeros
        auto x0 = DeviceArray<T>(recon_dims);

        // define the forward operator
        auto A = [&psf](const DeviceArray<T> &x) { return sysmat(x, psf); };

        // run optimization
        auto recon = DeviceArray<T>(recon_dims);
        switch (params.regularizer) {
            case Regularizer::UNCONSTRAINED: {
                std::cout << "Starting unconstrained reconstruction with CG ...\n";
                recon = opt::cgsolver<T>(A, yT, x0, params.maxIters, params.tol,
                                         params.xtol);
                break;
            }
            case Regularizer::SPLIT_BREGMAN: {
                std::cout << "Starting MBIR with Split-Bregman method ...\n";
                recon = opt::split_bregman<T>(A, yT, x0, params.lambda, params.mu,
                                              params.maxIters, params.innerIters,
                                              params.tol, params.xtol);

                break;
            }
            default: throw std::invalid_argument("Unsupported optimizer type");
        }

        // crop to original dimensions
        recon = crop3d(recon, params.recon_dims, PadType::SYMMETRIC);
        return recon.to_host();
    }

    // explicit template instantiation for float and double
    template Array<float> MBIR(const std::vector<Dataset_t<float>> &datasets,
                               const ReconParams &params);
    template Array<double> MBIR(const std::vector<Dataset_t<double>> &datasets,
                                const ReconParams &params);
} // namespace tomocam::gpu
