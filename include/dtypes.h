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

#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <tuple>

#ifndef DTYPES__H
#define DTYPES__H

namespace tomocam {

    struct dims_t {
        size_t n1;
        size_t n2;
        size_t n3;

      public:
        dims_t() : n1(0), n2(0), n3() {}
        dims_t(size_t a, size_t b, size_t c) : n1(a), n2(b), n3(c) {}

        [[nodiscard]] std::tuple<size_t, size_t, size_t>
        unravel_idx(size_t idx) const {
            size_t i = idx / n2 / n3;
            size_t j = (idx / n3) % n2;
            size_t k = idx % n3;
            return std::make_tuple(i, j, k);
        }

        // change individual dims (needed in padding)
        void set_x(size_t v) { n1 = v; }
        void set_y(size_t v) { n2 = v; }
        void set_z(size_t v) { n3 = v; }

        [[nodiscard]] size_t size() const { return (n1 * n2 * n3); }

        [[nodiscard]] size_t flat_idx(size_t i, size_t j, size_t k) const {
            return (((i * n2 + j) * n3) + k);
        }
        [[nodiscard]] size_t x() const { return n1; }
        [[nodiscard]] size_t y() const { return n2; }
        [[nodiscard]] size_t z() const { return n3; }

        dims_t operator+(const dims_t &v) const {
            return {n1 + v.n1, n2 + v.n2, n3 + v.n3};
        }
        dims_t operator-(const dims_t &v) const {
            return {n1 - v.n1, n2 - v.n2, n3 - v.n3};
        }
        dims_t operator*(int v) const { return {n1 * v, n2 * v, n3 * v}; }
        dims_t operator/(int v) const {
            if (v == 0) { throw std::runtime_error("divide by zeros"); }
            return {n1 / v, n2 / v, n3 / v};
        }
    };

    inline std::ostream &operator<<(std::ostream &outs, dims_t d) {
        outs << "dims_t(" << d.x() << ", " << d.y() << ", " << d.z() << ")";
        return outs;
    }
} // namespace tomocam
#endif // DTYPES__H
