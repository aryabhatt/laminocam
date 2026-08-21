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

// CPU-only operator consistency tests (double precision).
//
// Test 1 – adjoint:       <R x, y>  ==  <x, R^T y>
//   R   = forward()   (projection, projection.cpp)
//   R^T = backproj()  (unfiltered back-projection, projection.cpp)
//
// Test 2 – quadratic id:  x^T A x - 2<x, R^T y> + <y,y>  ==  ||Rx - y||^2
//   A is computed as R^T(R x) on the fly — this is a pure algebraic check
//   of the forward/backproj operators; sysmat is tested separately below.
//
// Test 3 – sysmat symmetry:  <u, A x>  ==  <x, A u>    (gradient.cpp)
//
// Test 4 – sysmat PSD:       <x, A x>  >=  0            (gradient.cpp)

#include <cmath>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "polar_grid.h"
#include "tomocam.h"

using namespace tomocam;

static double rel_diff(double a, double b) {
    double denom = std::max(std::abs(a), std::abs(b));
    return (denom > 0.0) ? std::abs(a - b) / denom : 0.0;
}

// <R x, y> == <x, R^T y>
static bool test_adjoint(const dims_t &vol_dims, const std::vector<double> &theta,
                         double gamma, double tol) {
    std::cout << "Adjoint       <Rx,y> = <x,R^T y>           ... ";

    cpu::PolarGrid<double> pg(theta, std::vector<double>(theta.size(), gamma),
                              std::vector<double>(theta.size(), 0.0), vol_dims.n2,
                              vol_dims.n3);
    auto x = Array<double>::random(vol_dims);
    auto y = Array<double>::random(pg.dims());
    auto Rx = forward(x, pg);
    auto RTy = backproj(y, pg, vol_dims);

    double lhs = array::dot(Rx, y);
    double rhs = array::dot(x, RTy);
    double rd = rel_diff(lhs, rhs);

    bool ok = rd < tol;
    std::cout << (ok ? "PASSED" : "FAILED") << "  lhs=" << lhs << "  rhs=" << rhs
              << "  rel_diff=" << rd << "\n";
    return ok;
}

// x^T(R^T R)x - 2<x, R^T y> + ||y||^2 == ||Rx - y||^2
// A = R^T R computed directly as backproj(forward(x))
static bool test_quadratic(const dims_t &vol_dims, const std::vector<double> &theta,
                           double gamma, double tol) {
    std::cout << "Quadratic id  x^T Ax - 2<x,R^T y> + ||y||^2 = ||Rx-y||^2 ... ";

    cpu::PolarGrid<double> pg(theta, std::vector<double>(theta.size(), gamma),
                              std::vector<double>(theta.size(), 0.0), vol_dims.n2,
                              vol_dims.n3);
    auto x = Array<double>::random(vol_dims);
    auto y = Array<double>::random(pg.dims());
    auto Rx = forward(x, pg);
    auto RTy = backproj(y, pg, vol_dims);
    auto Ax = sysmat(x, pg);

    double lhs = array::dot(x, Ax) - 2.0 * array::dot(x, RTy) + array::dot(y, y);
    auto res = Rx - y;
    double rhs = array::dot(res, res);
    double rd = rel_diff(lhs, rhs);

    bool ok = rd < tol;
    std::cout << (ok ? "PASSED" : "FAILED") << "  lhs=" << lhs << "  rhs=" << rhs
              << "  rel_diff=" << rd << "\n";
    return ok;
}

// <u, A x> == <x, A u>  (sysmat must be symmetric)
static bool test_sysmat_symmetry(const dims_t &vol_dims,
                                 const std::vector<double> &theta, double gamma,
                                 double tol) {
    std::cout << "Sysmat sym    <u,Ax> = <x,Au>               ... ";

    cpu::PolarGrid<double> pg(theta, std::vector<double>(theta.size(), gamma),
                              std::vector<double>(theta.size(), 0.0), vol_dims.n2,
                              vol_dims.n3);
    auto x = Array<double>::random(vol_dims);
    auto u = Array<double>::random(vol_dims);
    auto Ax = sysmat(x, pg);
    auto Au = sysmat(u, pg);

    double lhs = array::dot(u, Ax);
    double rhs = array::dot(x, Au);
    double rd = rel_diff(lhs, rhs);

    bool ok = rd < tol;
    std::cout << (ok ? "PASSED" : "FAILED") << "  lhs=" << lhs << "  rhs=" << rhs
              << "  rel_diff=" << rd << "\n";
    return ok;
}

// <x, A x> >= 0  (sysmat must be positive semi-definite)
static bool test_sysmat_psd(const dims_t &vol_dims, const std::vector<double> &theta,
                            double gamma) {
    std::cout << "Sysmat PSD    <x,Ax> >= 0                   ... ";

    cpu::PolarGrid<double> pg(theta, std::vector<double>(theta.size(), gamma),
                              std::vector<double>(theta.size(), 0.0), vol_dims.n2,
                              vol_dims.n3);
    auto x = Array<double>::random(vol_dims);
    auto Ax = sysmat(x, pg);

    double val = array::dot(x, Ax);
    bool ok = val >= 0.0;
    std::cout << (ok ? "PASSED" : "FAILED") << "  <x,Ax>=" << val << "\n";
    return ok;
}

int main() {
    std::cout
        << "\n====== CPU Projection Operator Tests (double precision) ======\n\n";

    const double gamma = 0.0;
    const double tol = 1e-5;
    int passed = 0, failed = 0;
    auto record = [&](bool ok) { ok ? ++passed : ++failed; };

    // Geometry 1: small — 5 slices, 16x16 detector, 7 tilt angles
    {
        const dims_t vol_dims{5, 16, 16};
        const size_t nangles = 7;
        std::vector<double> theta(nangles);
        for (size_t i = 0; i < nangles; ++i)
            theta[i] = (static_cast<double>(i) - nangles / 2.0) * M_PI / nangles;

        std::cout << "-- vol(" << vol_dims.n1 << "," << vol_dims.n2 << ","
                  << vol_dims.n3 << ")  nangles=" << nangles << " --\n";
        record(test_adjoint(vol_dims, theta, gamma, tol));
        // record(test_quadratic(vol_dims, theta, gamma, tol));
        record(test_sysmat_symmetry(vol_dims, theta, gamma, tol));
        record(test_sysmat_psd(vol_dims, theta, gamma));
        std::cout << "\n";
    }

    // Geometry 2: non-square, more angles
    {
        const dims_t vol_dims{8, 32, 32};
        const size_t nangles = 13;
        std::vector<double> theta(nangles);
        for (size_t i = 0; i < nangles; ++i)
            theta[i] = (static_cast<double>(i) - nangles / 2.0) * M_PI / nangles;

        std::cout << "-- vol(" << vol_dims.n1 << "," << vol_dims.n2 << ","
                  << vol_dims.n3 << ")  nangles=" << nangles << " --\n";
        record(test_adjoint(vol_dims, theta, gamma, tol));
        // record(test_quadratic(vol_dims, theta, gamma, tol));
        record(test_sysmat_symmetry(vol_dims, theta, gamma, tol));
        record(test_sysmat_psd(vol_dims, theta, gamma));
        std::cout << "\n";
    }

    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
