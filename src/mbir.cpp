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
#include "optimize.h"
#include "padding.h"
#include "polar_grid.h"
#include "tomocam.h"

namespace tomocam {
    template <typename T>
    Array<T> MBIR(const Array<T> &projs, const std::vector<T> &angles,
                  const dims_t &recon_dims, const opt::OptimizerConfig<T> &optcfg) {

        // normalize projections
        T proj_max = array::max(projs);
        auto y = projs / proj_max;

        // zero-pad projections by sqrt(2) to avoid aliasing
        T padding = static_cast<T>(1.42);
        y = pad2d(y, padding, PadType::SYMMETRIC);

        // adjust reconstruction dimensions
        dims_t out_dims = recon_dims;
        out_dims.n1 = static_cast<size_t>(recon_dims.n1 * padding);
        if (out_dims.n1 % 2 == 0) {
            out_dims.n1 -= 1; // make sure n1 is odd
        }

        out_dims.n2 = static_cast<size_t>(recon_dims.n2 * padding);
        if (out_dims.n2 % 2 == 0) {
            out_dims.n2 -= 1; // make sure n2 is odd
        }
        out_dims.n3 = static_cast<size_t>(recon_dims.n3 * padding);
        if (out_dims.n3 % 2 == 0) {
            out_dims.n3 -= 1; // make sure n3 is odd
        }

        // setup polar grid
        size_t nrows = y.nrows();
        size_t ncols = y.ncols();
        auto polar_grid = PolarGrid<T>(angles, nrows, ncols);

        // precompute backprojection of the projections
        auto yT = backproj(y, polar_grid, out_dims);

        auto recon = Array<T>(out_dims);
        auto x0 = fbp(y, polar_grid, out_dims);

        // run optimization
        switch (optcfg.method) {
            case opt::Optimizer::CG_SOLVER: {
                std::cout << std::format("Starting MBIR with CG optimization...\n");
                opt::Function<T> ATA = [&](const Array<T> &x) {
                    return sysmat(x, polar_grid);
                };
                recon = opt::cgsolver(ATA, yT, x0, optcfg.outer_max, optcfg.tol);
                break;
            }
            case opt::Optimizer::NAG_OPT: {
                std::cout << std::format("Starting MBIR with NAG optimization...\n");
                opt::Function<T> gradient = [&](const Array<T> &x) {
                    auto dx = sysmat(x, polar_grid) - yT;
                    opt::qggmrf(x, dx, optcfg.sigma, optcfg.p);
                    return dx;
                };
                opt::Residual<T> lossfn = [&](const Array<T> &x) {
                    auto tmp = forward(x, polar_grid) - y;
                    return std::sqrt(array::dot(tmp, tmp));
                };
                recon = opt::nagopt(gradient, lossfn, x0, optcfg.outer_max,
                                    optcfg.lipschitz, optcfg.tol, optcfg.xtol,
                                    optcfg.inner_max);
                break;
            }
            case opt::Optimizer::SPLIT_BREGMAN: {
                std::cout << std::format(
                    "Starting MBIR with Split Bregman optimization...\n");
                opt::Function<T> ATA = [&](const Array<T> &x) {
                    return sysmat(x, polar_grid);
                };
                recon = opt::split_bregman(ATA, yT, x0, optcfg.lambda, optcfg.mu,
                                           optcfg.outer_max, optcfg.inner_max,
                                           optcfg.tol, optcfg.xtol);
                break;
            }
            default: throw std::invalid_argument("Unsupported optimizer type");
        }

        // crop to original dimensions
        return crop3d(recon, recon_dims, PadType::SYMMETRIC);
    }

    // explicit template instantiation
    template Array<float> MBIR<float>(const Array<float> &projs,
                                      const std::vector<float> &angles,
                                      const dims_t &recon_dims,
                                      const opt::OptimizerConfig<float> &optcfg);
    template Array<double> MBIR<double>(const Array<double> &projs,
                                        const std::vector<double> &angles,
                                        const dims_t &recon_dims,
                                        const opt::OptimizerConfig<double> &optcfg);

} // namespace tomocam
