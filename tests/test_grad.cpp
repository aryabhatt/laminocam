#include <format>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "nufft.h"
#include "timer.h"
#include "tomocam.h"

using namespace tomocam;

int main() {

    dims_t dims = {21, 511, 511};
    Array<double> x = Array<double>::random(dims);

    size_t ntheta = 141;
    std::vector<double> theta(ntheta, 0.0);
    for (size_t i = 0; i < ntheta; i++) { theta[i] = (i - 70.0) * M_PI / 180.0; }

    auto pg = PolarGrid<double>(theta, 0, dims.n2, dims.n3);
    auto y = Array<double>::random(pg.dims());

    // FFTW does not normalize: fft2(ifft2(x)) = N^2 * x
    double N2 = static_cast<double>(dims.n2) * static_cast<double>(dims.n3);

    Timer timer;

    // Method 1: R^T(Rx - y) computed directly via projection.cpp
    timer.start();
    auto residual = forward(x, pg) - y;
    auto grad_direct = backproj(residual, pg, dims);
    timer.stop();
    std::cout << std::format("Direct R^T(Rx-y) time: {:.3f} s\n", timer.seconds());

    // Method 2: N^2 * sysmat(x) - R^T y (scaled expanded form via gradient.cpp)
    timer.start();
    auto grad_expanded = sysmat(x, pg) - backproj(y, pg, dims);
    timer.stop();
    std::cout << std::format("Expanded sysmat(x) - R^Ty time: {:.3f} s\n",
                             timer.seconds());

    // Compare
    auto scale = array::dot(grad_direct, grad_expanded) /
                 array::dot(grad_expanded, grad_expanded);
    auto diff = grad_direct - grad_expanded * scale;
    std::cout << "Scale factor: " << scale << "\n";
    std::cout << "Relative difference: "
              << array::norm2(diff) / array::norm2(grad_direct) << "\n";

    return 0;
}
