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

#ifndef TOMOCAM_GPU_OPT_H
#define TOMOCAM_GPU_OPT_H

#include <array>
#include <functional>

#include "gpu/device_array.h"

namespace tomocam::gpu::opt {

    template <typename T>
    using gpuFunction = std::function<DeviceArray<T>>(const DeviceArray<T> &) > ;

    /**
     * @brief GPU Conjugate Gradient solver for the preconditioned linear system
     *        A x = y, where A is a self-adjoint positive-definite operator.
     *
     * @param A        GPU operator (Compostion of RadonAdjoint and Radon)
     * @param y        Right-hand side: GPU array
     * @param x0       Initial guess
     * @param max_iter Maximum number of CG iterations
     * @param tol      Convergence tolerance (residual norm)
     * @return         Approximate solution x on the GPU
     */
    template <typename T>
    DeviceArray<T> cgsolver(const gpuFunction<T> &A, const DeviceArray<T> &y,
                            const DeviceArray<T> &x0, size_t max_iter, T tol);

} // namespace tomocam::gpu::opt

#endif // TOMOCAM_GPU_OPT_H
