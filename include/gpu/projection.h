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

#ifndef TOMOCAM_GPU_PROJECTION_H
#define TOMOCAM_GPU_PROJECTION_H

#include "dtypes.h"
#include "gpu/device_array.h"
#include "gpu/polar_grid.h"
#include "gpu/toeplitz.h"

namespace tomocam::gpu {

    /**
     * @brief GPU forward projection: volume -> projections.
     * @param volume  Device array with shape (nz, ny, nx).
     * @param pg      GPU polar grid.
     * @return        Device array of projections with shape (ntheta, nrows, ncols).
     */
    template <typename T>
    DeviceArray<T> forward(const DeviceArray<T> &volume, const PolarGrid<T> &pg);

    /**
     * @brief GPU backward (adjoint) projection: projections -> volume.
     * @param proj        Device array of projections.
     * @param pg          GPU polar grid.
     * @param recon_dims  Dimensions of the output volume.
     * @return            Device array with shape recon_dims.
     */
    template <typename T>
    DeviceArray<T> backward(const DeviceArray<T> &proj, const PolarGrid<T> &pg,
                            const dims_t &recon_dims);

    /**
     * @brief GPU adjoint projection (no filter): projections -> volume.
     *        Inline alias for backward(), matching the CPU backproj() interface.
     * @param proj        Device array of projections.
     * @param pg          GPU polar grid.
     * @param recon_dims  Dimensions of the output volume.
     * @return            Device array with shape recon_dims.
     */
    template <typename T>
    inline DeviceArray<T> backproj(const DeviceArray<T> &proj, const PolarGrid<T> &pg,
                                   const dims_t &recon_dims) {
        return backward(proj, pg, recon_dims);
    }

    /**
     * @brief GPU system matrix: (A^T A) x.
     * @param x   Device array with shape (nz, ny, nx).
     * @param pg  GPU polar grid.
     * @return    Device array of shape (nz, ny, nx).
     */
    template <typename T>
    DeviceArray<T> sysmat(const DeviceArray<T> &x, const PolarGrid<T> &pg);

    template <typename T>
    DeviceArray<T> sysmat(const DeviceArray<T> &x, const PointSpreadFunction<T> &psf);
} // namespace tomocam::gpu

#endif // TOMOCAM_GPU_PROJECTION_H
