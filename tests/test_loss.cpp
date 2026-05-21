#include <fstream>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "nufft.h"
#include "timer.h"
#include "tomocam.h"

using namespace tomocam;
int main() {

    // Define random f
    dims_t dims = {21, 511, 511};
    Array<float> f = Array<float>(dims);
    f = Array<float>::random(f.dims());

    // Define theta
    size_t ntheta = 141;
    std::vector<float> theta(ntheta, 0.0f);
    for (size_t i = 0; i < ntheta; i++) { theta[i] = (i - 70.f) * M_PI / 180.f; }

    auto pg = PolarGrid<float>(theta, 0, dims.n2, dims.n3);
    auto y = Array<float>::random(pg.x.dims());
    auto yTy = array::dot(y, y);

    Timer timer;

    // direct method
    timer.start();
    auto diff = forward(f, pg) - y;
    auto err = array::norm2(diff) / static_cast<float>(y.size());
    timer.stop();
    std::cout << std::format("Direct method time: {:.3f} s\n", timer.seconds());
    std::cout << std::format("Direct method loss: {:.6f}\n", err);

    // Ajdoint method

    auto yT = backproj(y, pg, dims);
    timer.start();
    auto err2 = residual(f, yT, pg, yTy) / static_cast<float>(y.size());
    timer.stop();
    std::cout << std::format("Adjoint method time: {:.3f} s\n", timer.seconds());
    std::cout << std::format("Adjoint method loss: {:.6f}\n", err2);

    return 0;
}
