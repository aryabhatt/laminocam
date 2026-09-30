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

#ifndef TOMOCAM_GPU_POLAR_GRID2D_H
#define TOMOCAM_GPU_POLAR_GRID2D_H

#include <cstddef>
#include <vector>

#include "dtypes.h"
#include "gpu/device_array.h"

namespace tomocam::gpu {

    /// Mirrors tomocam::cpu::PolarGrid2D but stores coordinates in device
    /// memory. Purely radial grid (x = r*cos(theta), y = r*sin(theta),
    /// r in [-pi, pi]) for 2D (vertical rotation axis) tomography -- no z
    /// or mask array is needed.
    ///
    /// @tparam T Floating-point element type (float or double)
    template <typename T>
    struct PolarGrid2D {
        size_t npts;
        DeviceArray<T> x;
        DeviceArray<T> y;

        [[nodiscard]] size_t nangles() const { return x.dims().n2; }
        [[nodiscard]] size_t nradial() const { return x.dims().n3; }
        [[nodiscard]] dims_t dims() const { return x.dims(); }
        [[nodiscard]] size_t size() const { return x.size(); }

        /// Constructs the polar grid on the GPU.
        ///
        /// @param theta  Host-side vector of projection angles (radians)
        /// @param ncols  Number of radial samples (== reconstructed image ncols)
        PolarGrid2D(const std::vector<T> &theta, size_t ncols);
    };
} // namespace tomocam::gpu

#endif // TOMOCAM_GPU_POLAR_GRID2D_H
