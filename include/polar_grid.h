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
#ifndef POLAR_GRID__H
#define POLAR_GRID__H

#include "array.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tomocam {

    template <typename T>
    struct PolarGrid {
        size_t npts;
        Array<T> x;
        Array<T> y;
        Array<T> z;

        // array dimensions for non-uniform points
        [[nodiscard]] dims_t dims() const { return x.dims(); }

        // size of the array
        [[nodiscard]] size_t size() const { return x.size(); }

        // constructor
        PolarGrid(const std::vector<T> &theta, T gamma, size_t nrows, size_t ncols) {

            dims_t dims = dims_t{theta.size(), nrows, ncols};
            x = Array<T>(dims);
            y = Array<T>(dims);
            z = Array<T>(dims);
            npts = dims.size();

            // rotation matrix
            T cos_gamma = std::cos(gamma);
            T sin_gamma = std::sin(gamma);

            // compute grid points
            T dh = (2 * M_PI) / static_cast<T>(ncols - 1);
            T dr = (2 * M_PI) / static_cast<T>(nrows - 1);

#pragma omp parallel for collapse(3)
            for (size_t i = 0; i < dims.n1; ++i) {
                for (size_t j = 0; j < dims.n2; ++j) {
                    for (size_t k = 0; k < dims.n3; ++k) {
                        T radius = j * dr - M_PI;

                        T xcrd = radius * std::cos(theta[i]);
                        T ycrd = radius * std::sin(theta[i]);
                        T zcrd = k * dh - M_PI;

                        // apply rotation
                        x[{i, j, k}] = zcrd * cos_gamma - xcrd * sin_gamma;
                        y[{i, j, k}] = ycrd;
                        z[{i, j, k}] = zcrd * sin_gamma + xcrd * cos_gamma;
                    }
                }
            }
        }
    };

} // namespace tomocam

#endif // POLAR_GRID__H
