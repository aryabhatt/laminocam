
#ifndef DENSITY_COMP_H
#define DENSITY_COMP_H

#include <complex>
#include <format>
#include <iostream>

#include "array.h"
#include "array_ops.h"
#include "nufft.h"

namespace tomocam {
    template <typename T>
    Array<T> pipe_menon_comp(PolarGrid<T> &grid, dims_t dims) {
        Array<T> weights = Array<T>::ones(grid.dims());

        Array<std::complex<T>> xcmplx = Array<std::complex<T>>::zeros(dims);
        for (int iter = 0; iter < 20; ++iter) {
            // rebuild complex weights from current real weights each iteration
            auto wcmplx = array::to_complex(weights);
            // Type 1 NUFFT: NU -> U
            nufft::nufft3d1(wcmplx, xcmplx, grid);
            // Type 2 NUFFT: U -> NU, p = G^H(G(w)) written into wcmplx
            nufft::nufft3d2(wcmplx, xcmplx, grid);

            // Pipe-Menon update: w_new[j] = w_old[j] / p[j]
            auto p = array::to_real(wcmplx);
            auto wnew = weights / p;

            T e = array::norm2(wnew - weights) / array::norm2(weights);
            std::cout << std::format("iter {}: error = {:.2e}", iter, e)
                      << std::endl;
            weights = std::move(wnew);
            if (e < 1e-6) { break; }
        }
        return weights;
    }
} // namespace tomocam
#endif // DENSITY_COMP_H
