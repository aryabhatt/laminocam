// clang-format off
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
 //clang-format on

#ifndef TOMOCAM_MASK_H
#define TOMOCAM_MASK_H

#include <cmath>
#include <execution>
#include <stdexcept>

#include "array.h"
#include <algorithm>

namespace tomocam {
    /**
     * @brief Creates a mask identifying finite values in an array.
     *
     * Returns a masked array where all the NaNs and Infs are set to 0.
     *
     * @tparam T Numeric type of the projection array elements
     * @param projs Input array to check for finite values
     * @return Array<T> Masked array with NaNs and Infs replaced by 0
     */
    template <typename T>
    inline Array<T> mask_infs_nans(const Array<T> &projs) {
        Array<T> masked = projs.clone();
        std::for_each(std::execution::par_unseq, masked.begin(), masked.end(),
                      [](T &val) {
                          if (!std::isfinite(val)) {
                              val = 0; // Set NaNs and Infs to 0
                          }
                      });
        return masked;
    }
    /**
     * @brief Creates a mask for laminography/tomography support region.
     *
     * Ptychographic projections are "infinite" in the sense that they
     * add successively add more date to the projection view as the object is
     * rotated. This extra data is missing from projections which are more normal to
     * the beam. This function assumes that the object is contained with rectangular
     * support.
     *
     * @tparam T Numeric type for coordinates and angles
     * @param projs Input projection array
     * @param dims Dimensions of the object being reconstructed (n1, n2, n3)
     * @param theta Vector of tomographic rotation angles (in radians)
     * @param gamma Laminography tilt angle (in radians)
     * @return Array<T> with values outside the support replaced by 0
     */
    template <typename T>
    Array<T> mask_support(const Array<T> &projs, const dims_t dims,
                          const std::vector<T> &theta, const T gamma) {

        // precompute sine and cosine of tilt angle
        const T cos_g = std::cos(gamma);
        const T sin_g = std::sin(gamma);

        // setup coordinates relative to object center
        const T xcen = projs.ncols() / 2;
        const T ycen = projs.nrows() / 2;

        // maximum extent of the object projection in the detector plane at zero tilt
        const T ylim = dims.n2 / 2;
        const T xlim = dims.n3 / 2;

        Array<T> mask = projs.clone();
        for (size_t p = 0; p < theta.size(); ++p) {
            const T angle = theta[p];
            const T cos_t = std::cos(angle);
            if (std::abs(cos_t) < 1e-06) {
                throw std::runtime_error("Projection angle too close to 90 "
                                         "degrees, support mask is undefined.");
            }
            for (size_t y = 0; y < projs.nrows(); ++y) {
                for (size_t x = 0; x < projs.ncols(); ++x) {
                    // coordinates relative to center
                    const T xrot = x - xcen;
                    const T yrot = y - ycen;

                    // un-rotate the coordinates
                    const T xcrd = (xrot * cos_g * cos_t + yrot * sin_g) / cos_t;
                    const T ycrd = (-xrot * sin_g * cos_t + yrot * cos_g) / cos_t;

                    // check if the projected point is within the support region
                    if (std::abs(xcrd) > xlim || std::abs(ycrd) > ylim) {
                        mask[{p, y, x}] = 0;
                    }
                }
            }
        }
        return mask;
    }

} // namespace tomocam

#endif // TOMOCAM_MASK_H
