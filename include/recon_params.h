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
#ifndef TOMOCAM_RECON_PARAMS_H
#define TOMOCAM_RECON_PARAMS_H

#include <array>
#include <cstddef>
#include <format>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "array.h"

namespace tomocam {

    // define Dataset_t (projections, angles, gamma)
    template <typename T>
    using Dataset_t = std::tuple<Array<T>, std::vector<T>, T>;

    enum class Regularizer { SPLIT_BREGMAN, UNCONSTRAINED };

    struct ReconParams {
        Regularizer regularizer = Regularizer::UNCONSTRAINED;
        std::array<size_t, 3> recon_dims = {0, 0, 0};
        size_t maxIters = 100;
        size_t innerIters = 1;
        float lambda = 0.1f;
        float mu = 10.0f;
        float tol = 1e-5f;
        float xtol = 1e-5f;
        float PAD_FACTOR = 1.4142f;

        void print(std::ostream &os) const {
            std::string reg_str = (regularizer == Regularizer::SPLIT_BREGMAN)
                                      ? "Split-Bregman"
                                      : "Unconstrained";
            os << "Reconstruction Parameters:\n";
            os << "  max_outer_iters: " << maxIters << "\n";
            os << std::format("  recon_dims: [{}, {}, {}]\n", recon_dims[0],
                              recon_dims[1], recon_dims[2]);
            os << "  tol: " << tol << "\n";
            os << "  xtol: " << xtol << "\n";
            os << "  regularizer: " << reg_str << "\n";
            if (regularizer == Regularizer::SPLIT_BREGMAN) {
                os << "    inner_iters: " << innerIters << "\n";
                os << "    lambda: " << lambda << "\n";
                os << "    mu: " << mu << "\n";
            }
        }
    };

    struct OutputParams {
        std::string filepath;
        std::vector<std::string> formats;

        // Check if a specific format is in the list of output formats
        bool has_format(const std::string &format) const {
            return std::find(formats.begin(), formats.end(), format) !=
                   formats.end();
        }
    };

} // namespace tomocam

#endif // TOMOCAM_RECON_PARAMS_H
