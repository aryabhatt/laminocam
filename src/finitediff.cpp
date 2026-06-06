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

#include "array.h"
#include "bregman.h"

namespace tomocam::opt {

    template <typename T>
    std::array<Array<T>, 3> grad_u(const Array<T> &u) {
        std::array<Array<T>, 3> grad;
        // grad w.r.t. z
        auto temp = Array<T>::zeros(u.dims());
#pragma omp parallel for collapse(3)
        for (size_t slc = 1; slc < u.nslices() - 1; ++slc) {
            for (size_t row = 1; row < u.nrows() - 1; ++row) {
                for (size_t col = 1; col < u.ncols() - 1; ++col) {
                    temp[{slc, row, col}] =
                        (u[{slc + 1, row, col}] - u[{slc - 1, row, col}]) / 2.0;
                }
            }
        }
        grad[0] = std::move(temp);
        // grad w.r.t. y
        temp = Array<T>::zeros(u.dims());
#pragma omp parallel for collapse(3)
        for (size_t slc = 1; slc < u.nslices() - 1; ++slc) {
            for (size_t row = 1; row < u.nrows() - 1; ++row) {
                for (size_t col = 1; col < u.ncols() - 1; ++col) {
                    temp[{slc, row, col}] =
                        (u[{slc, row + 1, col}] - u[{slc, row - 1, col}]) / 2.0;
                }
            }
        }
        grad[1] = std::move(temp);
        // grad w.r.t. x
        temp = Array<T>::zeros(u.dims());
#pragma omp parallel for collapse(3)
        for (size_t slc = 1; slc < u.nslices() - 1; ++slc) {
            for (size_t row = 1; row < u.nrows() - 1; ++row) {
                for (size_t col = 1; col < u.ncols() - 1; ++col) {
                    temp[{slc, row, col}] =
                        (u[{slc, row, col + 1}] - u[{slc, row, col - 1}]) / 2.0;
                }
            }
        }
        grad[2] = std::move(temp);
        return grad;
    }
    // explicit instantiation for float and double
    template std::array<Array<float>, 3> grad_u(const Array<float> &u);
    template std::array<Array<double>, 3> grad_u(const Array<double> &u);

    // divergence of the gradient of u
    template <typename T>
    Array<T> divergence(const std::array<Array<T>, 3> &du) {
        Array<T> div = Array<T>::zeros(du[0].dims());
#pragma omp parallel for collapse(3)
        for (size_t slc = 1; slc < du[0].nslices() - 1; ++slc) {
            for (size_t row = 1; row < du[0].nrows() - 1; ++row) {
                for (size_t col = 1; col < du[0].ncols() - 1; ++col) {
                    div[{slc, row, col}] =
                        (du[0][{slc + 1, row, col}] - du[0][{slc - 1, row, col}]) /
                            2.0 +
                        (du[1][{slc, row + 1, col}] - du[1][{slc, row - 1, col}]) /
                            2.0 +
                        (du[2][{slc, row, col + 1}] - du[2][{slc, row, col - 1}]) /
                            2.0;
                }
            }
        }
        return div;
    }
    // explicit instantiation for float and double
    template Array<float> divergence(const std::array<Array<float>, 3> &du);
    template Array<double> divergence(const std::array<Array<double>, 3> &du);

} // namespace tomocam::opt
