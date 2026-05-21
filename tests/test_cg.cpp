#include <array>
#include <cstddef>
#include <format>
#include <iostream>
#include <vector>

#include "array.h"
#include "optimize.h"
#include "tiff.h"
#include "tomocam.h"

typedef tomocam::Array<float> Array;

int main(int argc, char **argv) {

    size_t N = 100;
    if (argc == 2) { N = (size_t)atoi(argv[1]); }
    tomocam::dims_t dims = {1, 1, N};

    // create vector representing a diagonal matrix of size N
    std::vector<float> diagMat(N);
    for (size_t i = 0; i < N; i++) { diagMat[i] = static_cast<float>(i + 1); }

    tomocam::Array<float> xref(dims);
    auto x_true = tomocam::Array<float>::random(xref.dims());

    // create b = A*x_true
    tomocam::Array<float> b(dims);
    for (size_t i = 0; i < N; i++) { b[{0, 0, i}] = diagMat[i] * x_true[{0, 0, i}]; }

    // Conjugate Gradient solver for diagonal matrix
    std::function<tomocam::Array<float>(const tomocam::Array<float> &)> A =
        [&](const tomocam::Array<float> &v) {
            tomocam::Array<float> Av(dims);
            for (size_t i = 0; i < N; i++) {
                Av[{0, 0, i}] = diagMat[i] * v[{0, 0, i}];
            }
            return Av;
        };

    // call CG solver
    // initialize guess
    auto x = tomocam::Array<float>::random(xref.dims());
    auto x_sol = tomocam::opt::cgsolver(A, b, x, 1000, 1.0e-06f, 1.0e-04f);

    return 0;
}
