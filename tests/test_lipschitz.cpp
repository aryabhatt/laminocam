#include <fstream>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "nufft.h"
#include "optimize.h"
#include "timer.h"
#include "tomocam.h"

using namespace tomocam;
int main() {

    // Define random f
    dims_t dims = {21, 721, 721};
    Array<float> f = Array<float>(dims);
    f = Array<float>::random(f.dims());

    // Define theta
    size_t ntheta = 141;
    std::vector<float> theta(ntheta, 0.0f);
    for (size_t i = 0; i < ntheta; i++) { theta[i] = (i - 70.f) * M_PI / 180.f; }

    auto pg = PolarGrid<float>(theta, 0, dims.n2, dims.n3);

    Timer timer;
    // hacky method to esitmate Lipschitz constant
    auto f1 = Array<float>::ones(dims);
    auto yt = Array<float>::zeros(pg.dims());
    auto g1 = gradient(f1, yt, pg);
    std::cout << std::format("Max gradient: {:.6f}\n", array::max(g1));
    std::exit(1);

    // Compute Lipschitz constant
    opt::Function<float> sys = [&pg](const Array<float> &x) {
        return sysmat(x, pg);
    };
    float L = opt::lipschitz<float>(sys, f, 20, 1.e-04);
    std::cout << std::format("Lipschitz constant: {:.6f}\n", L);

    return 0;
}
