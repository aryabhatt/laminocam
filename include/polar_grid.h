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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "rotation.h"

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
                  const std::vector<T> &beta, size_t nrows, size_t ncols) {

            dims_t dims = dims_t{theta.size(), nrows, ncols};
            x = Array<T>(dims);
            y = Array<T>(dims);
            z = Array<T>(dims);
            w = Array<T>(dims);
            npts = dims.size();

            // compute grid points
            T dX = (2 * M_PI) / static_cast<T>(ncols);
            T dY = (2 * M_PI) / static_cast<T>(nrows);

#pragma omp parallel for collapse(3)
            for (size_t i = 0; i < dims.n1; ++i) {

                // compute rotation matrix for each angle
                auto R = RotationTranspose(theta[i], gamma[i], beta[i]);

                for (size_t j = 0; j < dims.n2; ++j) {
                    for (size_t k = 0; k < dims.n3; ++k) {

                        T qX = (k + 0.5) * dX - M_PI;
                        T qY = (j + 0.5) * dY - M_PI;
                        auto q_rot = matvec(R, {qX, qY, 0.0});

                        // non-uniform grid points
                        x[{i, j, k}] = q_rot[0];
                        y[{i, j, k}] = q_rot[1];
                        z[{i, j, k}] = q_rot[2];

                        // set weight to 0 if the point is outside the [-M_PI, M_PI]
                        if (std::abs(q_rot[0]) > M_PI || std::abs(q_rot[1]) > M_PI ||
                            std::abs(q_rot[2]) > M_PI) {
                            w[{i, j, k}] = (T)0;
                        } else {
                            w[{i, j, k}] = (T)1;
                        }
                    }
                }
            }
        }
    };
} // namespace tomocam::cpu

#endif // CPU_POLAR_GRID_H
