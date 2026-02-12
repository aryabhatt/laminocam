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

#ifndef TOMOCAM__H
#define TOMOCAM__H

#include <vector>

#include "array.h"
#include "dtypes.h"
#include "optimize.h"
#include "polar_grid.h"
#include "tiff.h"

namespace tomocam {
    /**
     * @brief Performs a forward projection of 3D volume data onto a polar grid.
     * @param volume The input 3D volume data as an Array.
     * @param grid The polar grid defining the projection geometry.
     * @param gamma orientation of the polar grid (default is 0).
     * @return The projected data as an Array.
     */
    template <typename T>
    Array<T> forward(const Array<T> &volume, const PolarGrid<T> &grid,
                     T gamma = T(0));

    /**
     * @brief Performs a backward projection from polar grid data to reconstruct a 3D
     * volume.
     * @param projections The input polar grid data as an Array.
     * @param grid The polar grid defining the projection geometry.
     * @param dims The dimensions of the output volume.
     * @param gamma orientation of the polar grid (ALS specific)
     * @param boolean filter Whether to apply filtering
     * @param filter_type The type of filter to apply
     * @return The backprojected polar data as an Array
     */
    template <typename T>
    Array<T> backward(const Array<T> &projections, const PolarGrid<T> &grid,
                      const dims_t &dims, T gamma, bool filter,
                      const std::string &filter_type);

    /**
     * @brief Adjoint of the Radon operator for polar grid data.
     * @param projections The input polar grid data as an Array.
     * @param grid The polar grid defining the projection geometry.
     * @param dims The dimensions of the output volume.
     * @param gamma orientation of the polar grid (default is 0).
     * @return The adjoint projected data as an Array.
     */
    template <typename T>
    Array<T> backproj(const Array<T> &projections, const PolarGrid<T> &grid,
                      const dims_t &dims, T gamma = T(0)) {
        std::string filter_type = "";
        return backward(projections, grid, dims, gamma, false, filter_type);
    }
    /**
     * @brief Filtered-backprojection of polar grid data to reconstruct a 3D volume.
     * @param projections The input polar grid data as an Array.
     * @param grid The polar grid defining the projection geometry.
     * @param dims The dimensions of the output volume.
     * @param gamma orientation of the polar grid (default is 0).
     * @pram filter_type The type of filter to apply (default is "ram-lak").
     * @return The reconstructed volume data as an Array.
     */
    template <typename T>
    Array<T> fbp(const Array<T> &projections, const PolarGrid<T> &grid,
                 const dims_t &dims, const std::string &filter_type = "ramp",
                 T gamma = T(0)) {
        return backward(projections, grid, dims, gamma, true, filter_type);
    }
    /**
     * @brief Function equivalent of system matrix y = A*x
     * @param x Input volume data as an Array.
     * @param grid The polar grid defining the projection geometry.
     * @return The equivalent of A^T(A*x) result as an Array.
     */
    template <typename T>
    Array<T> sysmat(const Array<T> &x, const PolarGrid<T> &grid);

    /**
     * @brief Computes the gradient of the objective function for iterative
     * reconstruction.
     * @param x Current estimate of the volume data as an Array.
     * @param b backprojection data as an Array.
     * @param grid The polar grid defining the projection geometry.
     * @return The gradient as an Array.
     */
    template <typename T>
    Array<T> gradient(const Array<T> &x, const Array<T> &b,
                      const PolarGrid<T> &grid);

    /**
     * @brief Computes the residual between the projected data and the measured data.
     * @param x Current estimate of the volume data as an Array.
     * @param b backprojection data as an Array.
     * @param grid The polar grid defining the projection geometry.
     * @param yTy Precomputed inner product of the measured data.
     * @return The computed residual as a scalar value of type T.
     */
    template <typename T>
    T residual(const Array<T> &x, const Array<T> &b, const PolarGrid<T> &grid,
               T yTy);

    /**
     * @brief Performs Model-Based Iterative Reconstruction (MBIR) of volume data.
     * @param projections The input projection data as an Array.
     * @param theta std::vector containing the projection angles.
     * @param recon_dims Dimensions of the output reconstructed volume.
     * @param opt_config Configuration parameters for the optimizer.
     * @return The reconstructed volume data as an Array.
     */
    template <typename T>
    Array<T> MBIR(const Array<T> &projections, const std::vector<T> &theta,
                  const dims_t &recon_dims,
                  const opt::OptimizerConfig<T> &opt_config);

} // namespace tomocam

#endif // TOMOCAM__H
