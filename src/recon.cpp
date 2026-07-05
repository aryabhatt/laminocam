// clang-format off
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
//clang-format on


#include <format>
#include <iostream>

#include "array_ops.h"
#include "config.h"
#include "tomocam.h"
#include "timer.h"

#ifdef USE_GPU
#include "gpu/tomocam.h"
#include "gpu/cufinufft_plan_cache.h"
#include "gpu/cufft_plan_cache.h"
#endif

using namespace tomocam;

int main(int argc, char **argv) {

    // parse TOML input
    if (argc < 2) {
        std::cerr << std::format("Usage: {} <input.toml>\n", argv[0]);
        std::cerr << "Please see config_template.toml for an example input file.\n";
        tomocam::dump_config("config_template.toml");
        return 1;
    }

    // read and parse TOML file
    toml::table config = tomocam::read_toml_file(argv[1]);
    // parse input datasets
    auto datasets = tomocam::parse_input_datasets<float>(config);
    // parse reconstruction parameters
    auto params = tomocam::parse_recon_params(config);
    // output parameters
    auto output = tomocam::parse_output_params(config);

    // set reconstruction dimensions
    dims_t recon_dims = params.recon_dims;

    // print parameters
    params.print(std::cout);

    // flush the output
    std::cout << std::endl;

    tomocam::Timer t0;
    t0.start();

#ifdef USE_GPU
    std::cout << "Running reconstruction on GPU...\n";
    auto recon = tomocam::gpu::MBIR<float>(datasets, params);
    tomocam::gpu::nufft::plans::cache<float>.clear();
    tomocam::gpu::fft::cache::plans<float>.clear();
    cudaDeviceReset();
#else
    std::cout << "Running reconstruction on CPU...\n";
    auto recon = tomocam::MBIR<float>(datasets, params);
#endif
    t0.stop();
    std::cout << std::format("Reconstruction completed in {:.2f} seconds.\n",
                             t0.seconds());

    auto base_dir = std::filesystem::path(output.filepath).parent_path();
    if (!std::filesystem::exists(base_dir)) {
        std::filesystem::create_directories(base_dir);
    }
    if (output.has_format("tiff")) {
        tomocam::tiff::write(output.filepath, recon);
        std::cout << std::format("Saved reconstruction to {}\n", output.filepath);
    }

    return 0;
}
