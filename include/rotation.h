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
#ifndef ROTATION_H
#define ROTATION_H

#include <array>
#include <cmath>

namespace tomocam {

    // Returns transpose( Rx(theta) * Ry(beta) * Rz(gamma) )
    template <typename T>
    std::array<std::array<T, 3>, 3> RotationTranspose(T theta, T gamma, T beta) {
        using std::cos;
        using std::sin;

        const T ct = cos(theta);
        const T st = sin(theta);
        const T cb = cos(beta);
        const T sb = sin(beta);
        const T cg = cos(gamma);
        const T sg = sin(gamma);

        std::array<std::array<T, 3>, 3> R{};

        R[0][0] = cb * cg;
        R[0][1] = ct * sg + st * sb * cg;
        R[0][2] = st * sg - ct * sb * cg;

        R[1][0] = -cb * sg;
        R[1][1] = ct * cg - st * sb * sg;
        R[1][2] = st * cg + ct * sb * sg;

        R[2][0] = sb;
        R[2][1] = -st * cb;
        R[2][2] = ct * cb;

        return R;
    }

    template <typename T>
    std::array<T, 3> matvec(const std::array<std::array<T, 3>, 3> &A,
                            const std::array<T, 3> &x) {
        std::array<T, 3> y{};
        for (int i = 0; i < 3; ++i) {
            y[i] = A[i][0] * x[0] + A[i][1] * x[1] + A[i][2] * x[2];
        }
        return y;
    }

} // namespace tomocam

#endif // ROTATION_H
