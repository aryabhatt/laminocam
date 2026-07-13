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
#ifndef CPU_POLAR_GRID_H
#define CPU_POLAR_GRID_H

#include "array.h"
#include "array_ops.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tomocam::cpu {

    template <typename T>
    struct PolarGrid {
        size_t npts;
        Array<T> x;
        Array<T> y;
        Array<T> z;
        Array<T> w;

        // array dimensions for non-uniform points
        [[nodiscard]] dims_t dims() const { return x.dims(); }

        // size of the array
        [[nodiscard]] size_t size() const { return x.size(); }

        // default constructor
        PolarGrid() : npts(0) {}

        // constructor
        PolarGrid(const std::vector<T> &theta, const std::vector<T> &gamma,
                  size_t nrows, size_t ncols) {

            dims_t dims = dims_t{theta.size(), nrows, ncols};
            x = Array<T>(dims);
            y = Array<T>(dims);
            z = Array<T>(dims);
            npts = dims.size();

            // compute grid points
            T dX = (2 * M_PI) / static_cast<T>(ncols);
            T dY = (2 * M_PI) / static_cast<T>(nrows);

#pragma omp parallel for collapse(3)
            for (size_t i = 0; i < dims.n1; ++i) {

                // precompute coses and sines
                T cos_theta = std::cos(theta[i]);
                T sin_theta = std::sin(theta[i]);
                T cos_gamma = std::cos(gamma[i]);
                T sin_gamma = std::sin(gamma[i]);

                for (size_t j = 0; j < dims.n2; ++j) {
                    for (size_t k = 0; k < dims.n3; ++k) {

                        T qX = (k + 0.5) * dX - M_PI;
                        T qY = (j + 0.5) * dY - M_PI;

                        x[{i, j, k}] = qX * cos_gamma - qY * sin_gamma * cos_theta;
                        y[{i, j, k}] = qX * sin_gamma + qY * cos_gamma * cos_theta;
                        z[{i, j, k}] = qY * sin_theta;
                    }
                }
            }
            w = Array<T>::ones(dims);
        }
    };

} // namespace tomocam::cpu

#endif // CPU_POLAR_GRID_H
