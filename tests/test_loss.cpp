#include <fstream>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "gpu/device_array_ops.h"
#include "gpu/tomocam.h"
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

    Timer timer;

    // CPU
    timer.start();
    auto diff = forward(f, pg) - y;
    auto err = array::norm2(diff) / static_cast<float>(y.size());
    timer.stop();
    std::cout << std::format("CPU time: {:.3f} s\n", timer.seconds());
    std::cout << std::format("CPU loss: {:.6f}\n", err);

    // GPU
    auto d_y = tomocam::gpu::DeviceArray<float>(y);
    auto d_f = tomocam::gpu::DeviceArray<float>(f);
    auto d_pg = tomocam::gpu::PolarGrid<float>(theta, 0, dims.n2, dims.n3);
    timer.start();
    auto d_diff = tomocam::gpu::forward(d_f, d_pg) - d_y;
    auto d_err = tomocam::gpu::array::norm2(d_diff) / static_cast<float>(y.size());
    timer.stop();
    std::cout << std::format("GPU time: {:.3f} s\n", timer.seconds());
    std::cout << std::format("GPU loss: {:.6f}\n", d_err);

    return 0;
}
