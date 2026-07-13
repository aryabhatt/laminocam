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
#include "optimize.h"
#include "padding.h"
#include "polar_grid.h"
#include "tomocam.h"

namespace tomocam {
    template <typename T>
    Array<T> MBIR(const std::vector<Dataset_t<T>> &datasets,
                  const ReconParams &params) {

        // zero-pad reconstruction dimensions by sqrt(2) to avoid aliasing
        T padfac = static_cast<T>(params.PAD_FACTOR);
        auto pad = [padfac](size_t n) {
            auto n_pad = static_cast<size_t>(n * padfac);
            if (n_pad % 2 == 0) { n_pad -= 1; } // make odd
            return n_pad;
        };

        // pad reconstruction dimensions
        dims_t recon_dims = params.recon_dims;
        dims_t out_dims = recon_dims;
        out_dims.n1 = pad(recon_dims.n1);
        out_dims.n2 = pad(recon_dims.n2);
        out_dims.n3 = pad(recon_dims.n3);

        std::cout << std::format(
            "Reconstruction dimensions (with padding): {} x {} x {}\n", out_dims.n1,
            out_dims.n2, out_dims.n3);

        /*
         * Stack all the projections from all the datasets into a single array,
         * and stack corresponding angles and gamma values into vectors.
         * Build an implicit Topelitz matrix to represent the system operator for the
         * reconstruction.
         */
        size_t n_projs = 0;
        size_t n_datasets = datasets.size();
        dims_t proj_dims = std::get<0>(datasets[0]).dims();
        for (size_t j = 0; j < n_datasets; ++j) {
            n_projs += std::get<1>(datasets[j]).size();
        }
        proj_dims.n1 = n_projs;
        std::vector<T> theta;
        std::vector<T> gamma;
        cpu::PolarGrid<T> pg;
        cpu::PointSpreadFunction<T> psf;
        Array<T> yT;
        {

            size_t offset = 0;
            Array<T> stacked_projs = Array<T>(proj_dims);
            for (size_t j = 0; j < n_datasets; ++j) {
                auto &[proj, angles, gamma_ref] = datasets[j];

                // stack projections into a single array
                std::copy(proj.data(), proj.data() + proj.size(),
                          stacked_projs.data() + offset);
                offset += proj.size();

                // combine angles and gamma values into single vectors
                for (auto &angle : angles) { theta.push_back(angle); }
                for (size_t i = 0; i < angles.size(); ++i) {
                    gamma.push_back(gamma_ref);
                }
            }

            // normalize values to [0, 1] to avoid numerical issues
            T max_val = array::max(stacked_projs);
            if (max_val > 0) { stacked_projs /= max_val; }

            // pad projections to avoid aliasing
            stacked_projs = pad2d(stacked_projs, padfac, PadType::SYMMETRIC);
            size_t nrows = stacked_projs.nrows();
            size_t ncols = stacked_projs.ncols();

            // build polar grid for system matrix
            pg = cpu::PolarGrid<T>(theta, gamma, nrows, ncols);

            // compute backprojection from stacked projections
            yT = backproj(stacked_projs, pg, out_dims);

            // build a point-spread function (PSF) for the system matrix
            psf = cpu::PointSpreadFunction<T>(pg, out_dims);
        }

        // build system operator that sums over all datasets
        opt::Function<T> A = [&psf](const Array<T> &x) { return sysmat(x, psf); };
        // opt::Function<T> A = [&pg](const Array<T> &x) { return sysmat(x, pg); };

        // initialize reconstruction with zeros
        auto x0 = Array<T>::zeros(out_dims);
        Array<T> recon;

        // run optimization
        switch (params.regularizer) {
            case Regularizer::UNCONSTRAINED: {
                std::cout << "Starting unconstrained iterative reconstruction with "
                             "CG ...\n";
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
        return crop3d(recon, recon_dims, PadType::SYMMETRIC);
    }

    // explicit template instantiation
    template Array<float> MBIR<float>(const std::vector<Dataset_t<float>> &datasets,
                                      const ReconParams &cfg);
    template Array<double>
    MBIR<double>(const std::vector<Dataset_t<double>> &datasets,
                 const ReconParams &cfg);

} // namespace tomocam
