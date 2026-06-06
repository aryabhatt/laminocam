#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <vector>

#include "array.h"
#include "dtypes.h"
#include "optimize.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/gpu_opt.h"

int main(int argc, char **argv) {

    size_t N = 100;
    if (argc == 2) { N = (size_t)atoi(argv[1]); }
    tomocam::dims_t dims = {1, 1, N};

    // diagonal matrix with fixed condition number κ = 100, independent of N
    constexpr float kappa = 100.f;
    std::vector<float> diagMat(N);
    for (size_t i = 0; i < N; i++)
        diagMat[i] = 1.f + (kappa - 1.f) * static_cast<float>(i) / static_cast<float>(std::max(N, size_t(2)) - 1);

    // true solution and rhs b = A * x_true
    auto x_true = tomocam::Array<float>::random(dims);
    tomocam::Array<float> b(dims);
    for (size_t i = 0; i < N; i++) b[{0, 0, i}] = diagMat[i] * x_true[{0, 0, i}];

    // shared initial guess
    auto x0 = tomocam::Array<float>::random(dims);

    // ---- CPU CG ----
    std::function<tomocam::Array<float>(const tomocam::Array<float> &)> A_cpu =
        [&](const tomocam::Array<float> &v) {
            tomocam::Array<float> Av(dims);
            for (size_t i = 0; i < N; i++) Av[{0, 0, i}] = diagMat[i] * v[{0, 0, i}];
            return Av;
        };

    std::cout << "Running CPU CG solver...\n";
    auto x_cpu = tomocam::opt::cgsolver(A_cpu, b, x0, 1000, 1.0e-06f, 1.0e-04f);

    // ---- GPU CG ----
    tomocam::Array<float> h_diag(dims);
    for (size_t i = 0; i < N; i++) h_diag[{0, 0, i}] = diagMat[i];
    tomocam::gpu::DeviceArray<float> d_diag(h_diag);

    tomocam::gpu::opt::gpuFunction<float> A_gpu =
        [&d_diag](const tomocam::gpu::DeviceArray<float> &v) {
            return v * d_diag;
        };

    tomocam::gpu::DeviceArray<float> d_b(b), d_x0(x0);
    std::cout << "Running GPU CG solver...\n";
    auto d_sol = tomocam::gpu::opt::cgsolver(A_gpu, d_b, d_x0, 1000, 1.0e-06f, 1.0e-04f);
    auto x_gpu = d_sol.to_host();

    // ---- Compare solutions ----
    float diff_sq = 0.f, ref_sq = 0.f;
    for (size_t i = 0; i < N; i++) {
        float d = x_cpu[{0, 0, i}] - x_gpu[{0, 0, i}];
        diff_sq += d * d;
        ref_sq  += x_cpu[{0, 0, i}] * x_cpu[{0, 0, i}];
    }
    float rel_err = (ref_sq > 0.f) ? std::sqrt(diff_sq / ref_sq) : std::sqrt(diff_sq);

    std::cout << std::format("CPU vs GPU CG rel_err = {:.6e}\n", rel_err);
    bool pass = rel_err < 1.0e-4f;
    std::cout << (pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
