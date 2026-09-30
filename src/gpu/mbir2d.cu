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

#include <format>
#include <iostream>

#include "array.h"
#include "gpu/device_array.h"
#include "gpu/gpu_opt.h"
#include "gpu/polar_grid2d.h"
#include "gpu/projection.h"
#include "gpu/toeplitz2d.h"
#include "recon_params.h"

namespace tomocam::gpu {

    template <typename T>
    Array<T> MBIR2D(const Array<T> &sinogram, const std::vector<T> &theta,
                    const ReconParams &params) {

        const dims_t recon_dims = params.recon_dims;

        PolarGrid2D<T> pg(theta, sinogram.ncols());

        // right-hand side: backprojection of the measured sinogram
        std::cout << "Computing backprojection (right-hand side) ...\n";
        DeviceArray<T> rhs;
        {
            DeviceArray<T> d_sino(sinogram);
            rhs = backproj2d(d_sino, pg, recon_dims);
        }

        std::cout << "Building 2D Toeplitz PSF ...\n";
        PointSpreadFunction2D<T> psf(pg, recon_dims.n3);
        auto A = [&psf](const DeviceArray<T> &x) { return sysmat2d(x, psf); };

        auto x0 = DeviceArray<T>(recon_dims);
        DeviceArray<T> recon;
        switch (params.regularizer) {
            case Regularizer::UNCONSTRAINED: {
                std::cout << "Starting unconstrained reconstruction with CG ...\n";
                recon = opt::cgsolver<T>(A, rhs, x0, params.maxIters, params.tol,
                                         params.xtol);
                break;
            }
            case Regularizer::SPLIT_BREGMAN: {
                std::cout << "Starting MBIR with Split-Bregman method ...\n";
                recon = opt::split_bregman<T>(A, rhs, x0, params.lambda, params.mu,
                                              params.maxIters, params.innerIters,
                                              params.tol, params.xtol);
                break;
            }
            default: throw std::invalid_argument("Unsupported optimizer type");
        }
        return recon.to_host();
    }

    template Array<float> MBIR2D(const Array<float> &, const std::vector<float> &,
                                 const ReconParams &);
    template Array<double> MBIR2D(const Array<double> &, const std::vector<double> &,
                                  const ReconParams &);
} // namespace tomocam::gpu
