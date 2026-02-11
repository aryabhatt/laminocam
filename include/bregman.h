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
