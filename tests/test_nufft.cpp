#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <vector>

#include "array.h"
#include "array_ops.h"
#include "dtypes.h"
#include "fft.h"
#include "nufft.h"
#include "polar_grid.h"

using namespace tomocam;

template <typename T>
T rand_val() {
    return static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
}

// Map FFTW array index i (in a dimension of size N) to centered frequency.
// FFTW ordering: k = 0, 1, ..., floor((N-1)/2), -floor(N/2), ..., -1
static int centered_freq(size_t i, size_t N) {
    return (i < (N + 1) / 2) ? static_cast<int>(i) : static_cast<int>(i) - static_cast<int>(N);
}

bool test_nufft3d1_vs_ifft_double(std::array<size_t, 3> dims,
                                  PolarGrid<double> &ug) {

    size_t nz = dims[0];
    size_t ny = dims[1];
    size_t nx = dims[2];
    std::cout << "Testing NUFFT3D1 vs IFFT3 (double): dims=(" << nz << "," << ny
              << "," << nx << ") ... ";

    // Random spatial domain data
    Array<double> spatial(dims_t{nz, ny, nx});
    for (size_t i = 0; i < spatial.size(); ++i) { spatial[i] = rand_val<double>(); }
    auto c = array::to_complex(spatial);

    // NUFFT Type-1: values at nonuniform (here uniform) points -> Fourier grid
    // Signature: nufft3d1(cz, fz, pg)
    //   cz = source VALUES at the M nonuniform points  (const, input)
    //   fz = output uniform FOURIER grid               (non-const, output)
    Array<std::complex<double>> nufft_result(dims_t{nz, ny, nx});
    nufft::nufft3d1(c, nufft_result, ug);

    // Reference: FFTW BACKWARD (unnormalized IFFT) of the same spatial data.
    //
    // With uniform points  x_j = 2π j/N - π  and iflag=+1, Type-1 gives:
    //
    //   F[n] = (-1)^(k_z+k_y+k_x) * IFFT(c)[n]
    //
    // where (k_z, k_y, k_x) are the centered FFTW frequencies at output
    // index n.  The (-1)^k phase comes from the [-π,π) coordinate shift in
    // FINUFFT's convention:
    //   exp(+i k (2π j/N - π)) = (-1)^k * exp(+2πi k j/N)
    // and exp(+2πi k j/N) summed over j is exactly FFTW_BACKWARD at index k.
    auto ref = fft::ifft3(c);

    double max_diff = 0.0;
    for (size_t iz = 0; iz < nz; ++iz) {
        int k_z = centered_freq(iz, nz);
        for (size_t iy = 0; iy < ny; ++iy) {
            int k_y = centered_freq(iy, ny);
            for (size_t ix = 0; ix < nx; ++ix) {
                int k_x = centered_freq(ix, nx);
                // (-1)^(k_z+k_y+k_x): sum%2 works for negative k in C++11
                // (truncation-toward-zero guarantees correct parity)
                double phase = ((k_z + k_y + k_x) % 2 == 0) ? 1.0 : -1.0;
                size_t idx   = iz * ny * nx + iy * nx + ix;
                auto expected = phase * ref[idx];
                double diff   = std::abs(nufft_result[idx] - expected);
                if (diff > max_diff) max_diff = diff;
            }
        }
    }

    // FINUFFT tolerance is 1e-14; allow 4 orders of margin for combined
    // FFTW + FINUFFT floating-point error.
    double tol = 1e-10;
    if (max_diff < tol) {
        std::cout << "PASSED (max diff = " << max_diff << ")" << std::endl;
        return true;
    } else {
        std::cout << "FAILED (max diff = " << max_diff << ")" << std::endl;
        return false;
    }
}

int main() {
    std::cout << std::endl;
    std::cout << "====== NUFFT Type 1 Correctness Tests ======" << std::endl;
    std::cout << "Comparing NUFFT3D1 with IFFT3 at uniform grid points" << std::endl;
    std::cout << "Theory: NUFFT3D1(c, uniform) = (-1)^(k_z+k_y+k_x) * IFFT3(c)" << std::endl;
    std::cout << "  where k_l are centered FFTW frequencies at each output index" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int failed = 0;

    // Test cases with different dimension combinations
    std::vector<std::tuple<size_t, size_t, size_t>> test_dims = {
        {31, 15, 7},   // rectangular
        {15, 15, 23},  // mixed sizes
        {23, 31, 15},  // another mix
    };

    std::cout << "--- Testing NUFFT3D1 vs IFFT3 (Double Precision) ---" << std::endl;
    for (const auto &[nz, ny, nx] : test_dims) {
        // Uniform grid in [-π, π) — FINUFFT coordinate convention
        PolarGrid<double> ug;
        ug.npts = nz * ny * nx;
        ug.x    = Array<double>(dims_t{nz, ny, nx});
        ug.y    = Array<double>(ug.x.dims());
        ug.z    = Array<double>(ug.x.dims());

        double dx = 2.0 * M_PI / nx;
        double dy = 2.0 * M_PI / ny;
        double dz = 2.0 * M_PI / nz;
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t idx  = z * ny * nx + y * nx + x;
                    ug.x[idx]   = x * dx - M_PI;
                    ug.y[idx]   = y * dy - M_PI;
                    ug.z[idx]   = z * dz - M_PI;
                }
            }
        }

        if (test_nufft3d1_vs_ifft_double({nz, ny, nx}, ug)) {
            passed++;
        } else {
            failed++;
        }
    }

    std::cout << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    return (failed == 0) ? 0 : 1;
}
