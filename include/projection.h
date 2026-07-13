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

#ifndef PROJECTION_H
#define PROJECTION_H

#include <string>

#include "array.h"
#include "dtypes.h"
#include "polar_grid.h"
#include "toeplitz.h"

namespace tomocam {

    template <typename T>
    Array<T> forward(const Array<T> &volume, const cpu::PolarGrid<T> &grid);

    template <typename T>
    Array<T> backproj(const Array<T> &projections, const cpu::PolarGrid<T> &grid,
                      const dims_t &dims);

    template <typename T>
    Array<T> sysmat(const Array<T> &x, const cpu::PolarGrid<T> &grid);

    template <typename T>
    Array<T> sysmat(const Array<T> &x, const cpu::PointSpreadFunction<T> &psf);

} // namespace tomocam

#endif // PROJECTION_H
