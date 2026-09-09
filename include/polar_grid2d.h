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
#ifndef CPU_POLAR_GRID2D_H
#define CPU_POLAR_GRID2D_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "array.h"

namespace tomocam::cpu {

    // Non-uniform Fourier grid for 2D (vertical rotation axis) tomography.
    // Points are purely radial: x = r*cos(theta), y = r*sin(theta) with
    // r in [-pi, pi], so every sample already satisfies |q| < pi by
    // construction -- unlike PolarGrid<T>, no mask/weight array is needed.
    template <typename T>
    struct PolarGrid2D {
        size_t npts;
        Array<T> x;
        Array<T> y;

        [[nodiscard]] size_t nangles() const { return x.dims().n2; }
        [[nodiscard]] size_t nradial() const { return x.dims().n3; }
        [[nodiscard]] dims_t dims() const { return x.dims(); }
        [[nodiscard]] size_t size() const { return x.size(); }

        PolarGrid2D() : npts(0) {}

        // theta: one projection angle per sinogram row.
        // ncols:  number of radial samples (== reconstructed image ncols).
        PolarGrid2D(const std::vector<T> &theta, size_t ncols) {
            dims_t dims{1, theta.size(), ncols};
            x = Array<T>(dims);
            y = Array<T>(dims);
            npts = dims.size();

            T dr = (2 * M_PI) / static_cast<T>(ncols);
#pragma omp parallel for collapse(2)
            for (size_t i = 0; i < theta.size(); ++i) {
                for (size_t k = 0; k < ncols; ++k) {
                    T r = (k + 0.5) * dr - M_PI;
                    x[{0, i, k}] = std::cos(theta[i]) * r;
                    y[{0, i, k}] = std::sin(theta[i]) * r;
                }
            }
        }
    };
} // namespace tomocam::cpu

#endif // CPU_POLAR_GRID2D_H
