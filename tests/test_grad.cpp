#include <fstream>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/projection.h"
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

    Timer timer;

    // direct method
    timer.start();
    auto fwd = forward(f, pg);
    auto grad_direct = backproj(fwd, pg, dims);
    timer.stop();
    std::cout << std::format("Direct method time: {:.3f} s\n", timer.seconds());

    // Ajdoint method
    auto yT = Array<float>::zeros(fwd.dims());
    timer.start();
    auto grad_adj = sysmat(f, pg) - yT;
    timer.stop();
    std::cout << std::format("Adjoint method time: {:.3f} s\n", timer.seconds());

    // Compare
    auto scale = array::dot(grad_direct, grad_adj) / array::dot(grad_adj, grad_adj);
    auto diff = grad_direct - grad_adj * scale;
    std::cout << "Scale factor: " << scale << std::endl;
    std::cout << "Relative difference: "
              << array::norm2(diff) / array::norm2(grad_direct) << std::endl;

    // test GPU implementation
    gpu::PolarGrid<float> d_pg(theta, 0, dims.n2, dims.n3);

    Timer gpu_timer;
    gpu_timer.start();
    gpu::DeviceArray<float> d_f(f);
    auto d_fwd = gpu::forward(d_f, d_pg);
    auto d_grad_direct = gpu::backproj(d_fwd, d_pg, dims);
    auto h_grad_direct = d_grad_direct.to_host();
    gpu_timer.stop();
    std::cout << std::format("GPU direct method time: {:.3f} s\n",
                             gpu_timer.seconds());
    scale = array::dot(grad_direct, h_grad_direct) /
            array::dot(h_grad_direct, h_grad_direct);
    diff = grad_direct - h_grad_direct * scale;
    std::cout << "GPU Scale factor: " << scale << std::endl;
    std::cout << "GPU Relative difference: "
              << array::norm2(diff) / array::norm2(grad_direct) << std::endl;

    // GPU adjoint method
    gpu_timer.start();
    auto gpu_grad_adj = gpu::sysmat(d_f, d_pg) - gpu::DeviceArray<float>(fwd.dims());
    auto h_grad_adj = gpu_grad_adj.to_host();
    gpu_timer.stop();
    std::cout << std::format("GPU adjoint method time: {:.3f} s\n",
                             gpu_timer.seconds());
    scale = array::dot(grad_direct, h_grad_adj) / array::dot(h_grad_adj, h_grad_adj);
    diff = grad_direct - h_grad_adj * scale;
    std::cout << "GPU Adjoint Scale factor: " << scale << std::endl;
    std::cout << "GPU Adjoint Relative difference: "
              << array::norm2(diff) / array::norm2(grad_direct) << std::endl;
    return 0;
}
