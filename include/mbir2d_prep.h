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

#ifndef TOMOCAM_MBIR2D_PREP_H
#define TOMOCAM_MBIR2D_PREP_H

#include <algorithm>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "recon_params.h"

namespace tomocam {

    // Host-side sinogram for the 2D (vertical rotation axis) reconstruction.
    template <typename T>
    struct Sinogram2D {
        Array<T> data;        // {n_slices, n_angles, n_cols}
        std::vector<T> theta; // projection angles (radians), one per angle
    };

    // Stack all datasets' projections and angles, normalize by the global
    // maximum, and reorder {n_angles, n_rows, n_cols} -> {n_rows, n_angles,
    // n_cols} so each detector row becomes one slice. Shared by the CPU and
    // GPU reconstruction paths.
    template <typename T>
    Sinogram2D<T> prepare_sinogram(const std::vector<Dataset_t<T>> &datasets) {
        size_t n_projs = 0;
        for (auto &ds : datasets) n_projs += ds.angles.size();
        dims_t proj_dims = datasets[0].projs.dims();
        proj_dims.n1 = n_projs;

        Sinogram2D<T> out;
        out.theta.reserve(n_projs);
        Array<T> stacked(proj_dims);
        size_t offset = 0;
        for (auto &ds : datasets) {
            std::copy(ds.projs.data(), ds.projs.data() + ds.projs.size(),
                      stacked.data() + offset);
            offset += ds.projs.size();
            for (auto &a : ds.angles) out.theta.push_back(a);
        }
        T max_val = array::max(stacked);
        if (max_val > 0) { stacked /= max_val; }
        out.data = array::transpose(stacked, {1, 0, 2});
        return out;
    }

} // namespace tomocam

#endif // TOMOCAM_MBIR2D_PREP_H
