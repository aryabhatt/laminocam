
#include "array.h"
#include "bregman.h"

namespace tomocam::opt {

    template <typename T>
    std::array<Array<T>, 3> grad_u(const Array<T> &u) {
        std::array<Array<T>, 3> grad;
        // grad w.r.t. z
        auto temp = Array<T>::zeros_like(u);
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
        temp = Array<T>::zeros_like(u);
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
        temp = Array<T>::zeros_like(u);
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
        Array<T> div = Array<T>::zeros_like(du[0]);
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

    // laplacian of u
    template <typename T>
    Array<T> laplacian(const Array<T> &u) {
        Array<T> laplacian = Array<T>::zeros_like(u);
#pragma omp parallel for collapse(3)
        for (size_t slc = 1; slc < u.nslices() - 1; ++slc) {
            for (size_t row = 1; row < u.nrows() - 1; ++row) {
                for (size_t col = 1; col < u.ncols() - 1; ++col) {
                    laplacian[{slc, row, col}] =
                        u[{slc + 1, row, col}] + u[{slc - 1, row, col}] +
                        u[{slc, row + 1, col}] + u[{slc, row - 1, col}] +
                        u[{slc, row, col + 1}] + u[{slc, row, col - 1}] -
                        6 * u[{slc, row, col}];
                }
            }
        }
        return laplacian;
    }
    // explicit instantiation for float and double
    template Array<float> laplacian(const Array<float> &u);
    template Array<double> laplacian(const Array<double> &u);

} // namespace tomocam::opt
