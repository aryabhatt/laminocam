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

#ifndef TOMOCAM_GPU_POLAR_GRID_H
#define TOMOCAM_GPU_POLAR_GRID_H

#include <cstddef>
#include <vector>

#include "dtypes.h"
#include "gpu/device_array.h"
#include "gpu/device_ptr.h"
#include "gpu/utils.h"

namespace tomocam::gpu {

    /// Mirrors the interface of tomocam::PolarGrid but stores coordinate
    /// arrays in device memory as DeviceArray<T>.
    ///
    /// Construction uploads the computed (x, y, z) grid points directly
    /// to the GPU via a CUDA kernel; no host-side coordinate arrays are retained.
    ///
    /// @tparam T Floating-point element type (float or double)
    template <typename T>
    struct PolarGrid {
        size_t npts;
        DeviceArray<T> x;
        DeviceArray<T> y;
        DeviceArray<T> z;

        /// Returns dimensions (nangles, nrows, ncols) of the coordinate arrays.
        [[nodiscard]] dims_t dims() const { return x.dims(); }

        /// Returns total number of non-uniform grid points.
        [[nodiscard]] size_t size() const { return x.size(); }

        /// Constructs the polar grid on the GPU.
        ///
        /// @param theta  Host-side vector of projection angles (radians)
        /// @param gamma  Out-of-plane tilt angle (radians)
        /// @param nrows  Number of radial samples
        /// @param ncols  Number of axial samples
        PolarGrid(const std::vector<T> &theta, T gamma, size_t nrows, size_t ncols);
    };
} // namespace tomocam::gpu

#endif // TOMOCAM_GPU_POLAR_GRID_H
