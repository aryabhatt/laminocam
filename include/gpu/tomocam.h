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

#ifndef TOMOCAM_GPU_H
#define TOMOCAM_GPU_H

#include <string>
#include <vector>

#include "dtypes.h"
#include "gpu/device_array.h"
#include "gpu/polar_grid.h"
#include "gpu/projection.h"
#include "recon_params.h"

namespace tomocam::gpu {

    /**
     * @brief GPU filtered backprojection: reconstruct a 3D volume from projections
     *        with frequency-domain filtering.
     * @param projections Device array of projections with shape (ntheta, nrows,
     * ncols).
     * @param grid        GPU polar grid defining projection geometry.
     * @param dims        Dimensions of the output volume.
     * @param filter_type Name of the filter to apply (e.g. "ramp", "ram-lak").
     * @return            Reconstructed volume as a DeviceArray.
     *
     */

    template <typename T>
    DeviceArray<T> fbp(const DeviceArray<T> &projections, const PolarGrid<T> &grid,
                       const dims_t &dims, const std::string &filter_type = "ramp");

    /**
     * @brief GPU model-based iterative reconstruction (MBIR): reconstruct a 3D
     *        volume from projections by iteratively minimizing a cost function
     *        that includes a data fidelity term and a regularization term.
     * @param projs   DeviceArray of projections with shape (ntheta, nrows, ncols).
     * @param angles  std::vector of projection angles in radians, with length
     * ntheta.
     * @param gamma   Inplane orientation of the sample in radians
     * @param recon_dims Dimensions of the output volume (nx, ny, nz).
     * @param params  Reconstruction parameters (e.g. number of iterations,
     * regularization weight, etc.).
     * @return        Reconstructed volume as a DeviceArray.
     */
    template <typename T>
    DeviceArray<T> MBIR(const DeviceArray<T> &projs, const std::vector<T> &angles,
                        T gamma, const dims_t &recon_dims,
                        const ReconParams &params);

    /**
     * TODO: genrealize to accept multiple datasets one per each gamma orientation
     *
     * @brief Host wrapper for GPU MBIR: takes projections and angles on the host,
     *        moves data to GPU, calls the GPU MBIR function, and returns the
     *        reconstructed volume back on the host.
     * @param dataset  Dataset containing projections, angles, and gamma reference.
     * @param params   Reconstruction parameters (e.g. number of iterations,
     * regularization weight, etc.).
     * @return         Reconstructed volume as a host array (std::vector or
     * similar).
     */
    template <typename T>
    Array<T> MBIR(const std::vector<Dataset_t<T>> &datasets,
                  const dims_t &recon_dims, const ReconParams &params) {

        auto &[projs, angles, gamma_ref] = datasets[0];
        T gamma = gamma_ref;

        // Move data to GPU
        DeviceArray<T> d_projs(projs);

        // Call GPU MBIR
        DeviceArray<T> d_recon = MBIR(d_projs, angles, gamma, recon_dims, params);

        // Move result back to host and return
        return d_recon.to_host();
    }
} // namespace tomocam::gpu

#endif // TOMOCAM_GPU_H
