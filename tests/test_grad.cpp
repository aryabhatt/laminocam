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
    f = Array<float>::random_like(f);

    // Define theta
    size_t ntheta = 141;
    std::vector<float> theta(ntheta, 0.0f);
    for (size_t i = 0; i < ntheta; i++) { theta[i] = (i - 70.f) * M_PI / 180.f; }

    auto pg = PolarGrid<float>(theta, 0, dims.n2, dims.n3);

    Timer timer;

    // direct method
    timer.start();
    auto fwd = forward(f, pg);
    auto grad_direct = backproj(fwd, pg, dims);
    timer.stop();
    std::cout << std::format("Direct method time: {:.3f} s\n", timer.seconds());

    // Ajdoint method
    auto yT = Array<float>::zeros_like(fwd);
    timer.start();
    auto grad_adj = gradient(f, yT, pg);
    timer.stop();
    std::cout << std::format("Adjoint method time: {:.3f} s\n", timer.seconds());

    // Compare
    auto scale = array::dot(grad_direct, grad_adj) / array::dot(grad_adj, grad_adj);
    auto diff = grad_direct - grad_adj * scale;
    std::cout << "Scale factor: " << scale << std::endl;
    std::cout << "Relative difference: "
              << array::norm2(diff) / array::norm2(grad_direct) << std::endl;

    return 0;
}
