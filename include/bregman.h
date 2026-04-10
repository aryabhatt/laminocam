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
#ifndef BREGMAN_H
#define BREGMAN_H

#include <array>

#include "array.h"

namespace tomocam::opt {
    /**
     * @brief Compute the Laplacian of an array.
     * @tparam T The type of the array elements.
     * @param u The input array.
     * @return The Laplacian of the input array.
     */
    template <typename T>
    Array<T> laplacian(const Array<T> &u);

    /**
     * @brief Compute the gradient of an array.
     * @tparam T The type of the array elements.
     * @param u The input array.
     * @return The gradient of the input array.
     */
    template <typename T>
    std::array<Array<T>, 3> grad_u(const Array<T> &u);

    /**
     * @brief Compute the divergence of an array.
     * @tparam T The type of the array elements.
     * @param u The input array.
     * @return The divergence of the input array.
     */
    template <typename T>
    Array<T> divergence(const std::array<Array<T>, 3> &u);
} // namespace tomocam::opt

#endif // BREGMAN_H
