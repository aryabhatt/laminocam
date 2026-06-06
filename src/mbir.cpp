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
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "config.h"
#include "density_comp.h"
#include "optimize.h"
#include "padding.h"
#include "polar_grid.h"
#include "tomocam.h"

namespace tomocam {
    template <typename T>
    Array<T> MBIR(const std::vector<Dataset_t<T>> &datasets,
                  const dims_t &recon_dims, const ReconParams &params) {

        // zero-pad reconstruction dimensions by sqrt(2) to avoid aliasing
        float padding = static_cast<T>(params.PAD_FACTOR);
        auto pad = [padding](size_t n) {
            size_t npad = 2 * (static_cast<size_t>(n * (padding - 1)) / 2);
            return n + npad;
        };

        dims_t out_dims = recon_dims;
        out_dims.n1 = pad(recon_dims.n1);
        out_dims.n2 = pad(recon_dims.n2);
        out_dims.n3 = pad(recon_dims.n3);

        std::cout << std::format(
            "Reconstruction dimensions (with padding): {} x {} x {}\n", out_dims.n1,
            out_dims.n2, out_dims.n3);

        // accumulate backprojections across all datasets
        size_t n_datasets = datasets.size();
        auto yT = Array<T>::zeros(out_dims);
        std::vector<PolarGrid<T>> polar_grids(n_datasets);

        for (size_t j = 0; j < n_datasets; ++j) {
            auto &[proj, angles, gamma] = datasets[j];

            // normalize projections
            T proj_max = array::max(proj);
            auto y = proj / proj_max;

            // zero-pad projections by sqrt(2) to avoid aliasing
            y = pad2d(y, padding, PadType::SYMMETRIC);

            // setup polar grid (gamma already encoded in grid)
            size_t nrows = y.nrows();
            size_t ncols = y.ncols();
            polar_grids[j] = PolarGrid<T>(angles, gamma, nrows, ncols);

            // accumulate backprojection
            yT += backproj(y, polar_grids[j], out_dims);
        }

        // build system operator that sums over all datasets
        opt::Function<T> ATA = [&polar_grids](const Array<T> &x) {
            auto Ax = sysmat<T>(x, polar_grids[0]);
            for (size_t j = 1; j < polar_grids.size(); ++j) {
                Ax += sysmat<T>(x, polar_grids[j]);
            }
            return Ax;
        };

        // initial guess: FBP from first dataset
        auto &[proj0, angles0, gamma0] = datasets[0];
        auto y0 = proj0 / array::max(proj0);
        y0 = pad2d(y0, padding, PadType::SYMMETRIC);
        auto x0 = fbp(y0, polar_grids[0], out_dims);

        auto recon = Array<T>(out_dims);

        // run optimization
        switch (params.regularizer) {
            case Regularizer::UNCONSTRAINED: {
                std::cout << "Starting unconstrained iterative reconstruction with "
                             "CG ...\n";
                recon = opt::cgsolver<T>(ATA, yT, x0, params.maxIters, params.tol,
                                         params.xtol);
                break;
            }
            case Regularizer::SPLIT_BREGMAN: {
                std::cout << "Starting MBIR with Split-Bregman method ...\n";
                recon = opt::split_bregman<T>(ATA, yT, x0, params.lambda, params.mu,
                                              params.maxIters, params.innerIters,
                                              params.tol, params.xtol);
                break;
            }
            default: throw std::invalid_argument("Unsupported optimizer type");
        }

        // crop to original dimensions
        return crop3d(recon, recon_dims, PadType::SYMMETRIC);
    }

    // explicit template instantiation
    template Array<float> MBIR<float>(const std::vector<Dataset_t<float>> &datasets,
                                      const dims_t &recon_dims,
                                      const ReconParams &cfg);
    template Array<double>
    MBIR<double>(const std::vector<Dataset_t<double>> &datasets,
                 const dims_t &recon_dims, const ReconParams &cfg);

} // namespace tomocam
