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

#ifndef NUFFT__H
#define NUFFT__H

#include <array>
#include <complex>
#include <cstdint>

#include "array.h"
#include "dtypes.h"
#include "finufft_plan_cache.h"
#include "polar_grid.h"
#include "polar_grid2d.h"

namespace tomocam::nufft {

    // 3D Type-1 NUFFT: nonuniform points to uniform grid
    template <typename T>
    void nufft3d1(const Array<std::complex<T>> &cz, Array<std::complex<T>> &fz,
                  const cpu::PolarGrid<T> &pg) {
        std::array<int64_t, 3> n_modes = {static_cast<int64_t>(fz.ncols()),
                                          static_cast<int64_t>(fz.nrows()),
                                          static_cast<int64_t>(fz.nslices())};
        auto &plan = plans::cache<T>.get_plan(1, 3, n_modes, 1);
        plan.set_points(pg);
        plan.execute(const_cast<std::complex<T> *>(cz.begin()), fz.begin());
    }

    // 3D Type-2 NUFFT: uniform grid to nonuniform points
    template <typename T>
    void nufft3d2(Array<std::complex<T>> &cz, const Array<std::complex<T>> &fz,
                  const cpu::PolarGrid<T> &pg) {
        std::array<int64_t, 3> n_modes = {static_cast<int64_t>(fz.ncols()),
                                          static_cast<int64_t>(fz.nrows()),
                                          static_cast<int64_t>(fz.nslices())};
        auto &plan = plans::cache<T>.get_plan(2, 3, n_modes, -1);
        plan.set_points(pg);
        plan.execute(cz.begin(), const_cast<std::complex<T> *>(fz.begin()));
    }

    // 2D Type-1 NUFFT: nonuniform points to uniform grid, broadcast over the
    // outer ("slice") axis of cz/fz. The polar grid is identical for every
    // slice (vertical rotation axis), so set_points() runs once and
    // execute() runs once per slice.
    template <typename T>
    void nufft2d1(const Array<std::complex<T>> &cz, Array<std::complex<T>> &fz,
                  const cpu::PolarGrid2D<T> &pg) {
        std::array<int64_t, 3> n_modes = {static_cast<int64_t>(fz.ncols()),
                                          static_cast<int64_t>(fz.nrows()), 1};
        auto &plan = plans::cache<T>.get_plan(1, 2, n_modes, 1);
        plan.set_points(pg);

        size_t nvox = fz.nrows() * fz.ncols();
        for (size_t i = 0; i < fz.nslices(); ++i) {
            auto *cz_i = const_cast<std::complex<T> *>(cz.begin()) + i * pg.npts;
            auto *fz_i = fz.begin() + i * nvox;
            plan.execute(cz_i, fz_i);
        }
    }

    // 2D Type-2 NUFFT: uniform grid to nonuniform points, broadcast over slices.
    template <typename T>
    void nufft2d2(Array<std::complex<T>> &cz, const Array<std::complex<T>> &fz,
                  const cpu::PolarGrid2D<T> &pg) {
        std::array<int64_t, 3> n_modes = {static_cast<int64_t>(fz.ncols()),
                                          static_cast<int64_t>(fz.nrows()), 1};
        auto &plan = plans::cache<T>.get_plan(2, 2, n_modes, -1);
        plan.set_points(pg);

        size_t nvox = fz.nrows() * fz.ncols();
        for (size_t i = 0; i < fz.nslices(); ++i) {
            auto *cz_i = cz.begin() + i * pg.npts;
            auto *fz_i = const_cast<std::complex<T> *>(fz.begin()) + i * nvox;
            plan.execute(cz_i, fz_i);
        }
    }

} // namespace tomocam::nufft

#endif // NUFFT__H
