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
#include <fstream>
#include <iostream>
#include <string>
#include <toml++/toml.h>

#include "tomocam.h"
#include "timer.h"

void dump_config() {
    std::ofstream outf("config_template.toml", std::ios::out);
    outf << "[input]\n";
    outf << "filename = \"/path/to/dataset.tif\"\n";
    outf << "angles = \"/path/to/angles.txt\"\n";
    outf << "\n";
    outf << "[output]\n";
    outf << "filename = \"output.tiff\"\n";
    outf << "\n";
    outf << "[recon_params]\n";
    outf << "max_iters = 50\n";
    outf << "lambda = 0.1\n";
    outf << "mu = 5.0\n";
    outf << "tol = 1e-5\n";
    outf << "xtol = 1e-5\n";
    outf << "thickness = 21\n";
    outf.close();
}

int main(int argc, char **argv) {

    // parse TOML input
    if (argc < 2) {
        std::cerr << std::format("Usage: {} <input.toml>\n", argv[0]);
        std::cerr << "Please see config_template.toml for an example input file.\n";
        dump_config();
        return 1;
    }

    std::string input_file = argv[1];
    toml::table config;
    try {
        config = toml::parse_file(input_file);
    } catch (const toml::parse_error& err) {
        std::cerr << "Error parsing TOML file:\n" << err << "\n";
        return 1;
    }

    // get the input entry
    auto input = config["input"].as_table();
    std::string dataset = (*input)["filename"].value_or("");
    if (dataset.empty()) {
        std::cerr << "Error: projection file not specified in input\n";
        return 1;
    }
    auto projs = tomocam::tiff::read(dataset);

    // read projection angles 
    std::string angles_file = (*input)["angles"].value_or("");
    if (angles_file.empty()) {
        std::cerr << "Error: angles file not specified in input\n";
        return 1;
    }
    std::ifstream fp(angles_file);
    if (!fp.is_open()) {
        std::cerr << std::format("Error: Could not open angles file {}\n", angles_file);
        return 1;
    }
    std::vector<float> angles;
    float angle;
    while (fp >> angle) {
        angles.push_back(angle);
    }
    fp.close();

    // get recon_params with defaults
    auto params = config["recon_params"].as_table();
    size_t max_iter = params ? (*params)["max_iters"].value_or(50) : 50;
    float lambda = params ? (*params)["lambda"].value_or(0.1f) : 0.1f;
    float mu = params ? (*params)["mu"].value_or(5.0f) : 5.0f;
    float tol = params ? (*params)["tol"].value_or(1e-5f) : 1e-5f;
    float xtol = params ? (*params)["xtol"].value_or(1e-5f) : 1e-5f;
    size_t thickness = (*params)["thickness"].value_or(21);

    // ensure angles are in radians
    auto max_angle = *std::max_element(angles.begin(), angles.end());
    if (std::abs(max_angle) > 2 * M_PI) {
        for (auto &angle : angles) { angle = angle * M_PI / 180.0f; }
    }
    // print parameters
    std::cout << "Reconstruction parameters:\n";
    std::cout << std::format("  Dataset: {}\n", dataset);
    std::cout << std::format("  Projections: {} x {} x {}\n", projs.nslices(),
                             projs.nrows(), projs.ncols());
    std::cout << std::format("  Angles: {} values from {:.2f} to {:.2f} radians\n",
                             angles.size(), angles.front(), angles.back());
    std::cout << std::format("  Recon Dimensions: {} x {} x {}\n", thickness,
                             projs.nrows(), projs.ncols());
    std::cout << std::format("  Max iterations: {}\n", max_iter);
    std::cout << std::format("  \u03BB: {:.2f}\n", lambda);
    std::cout << std::format("  \u03BC: {:.2f}\n", mu);
    std::cout << std::format("  Tolerance: {:.2e}\n", tol);
    std::cout << std::format("  X-tolerance: {:.2e}\n", xtol);

    // set reconstruction dimensions
    tomocam::dims_t img_dims = {thickness, projs.nrows(), projs.ncols()};

    tomocam::Timer t0;
    t0.start();
    auto recon =
        tomocam::MBIR(projs, angles, img_dims, max_iter, lambda, mu, tol, xtol);
    t0.stop();
    std::cout << std::format("Reconstruction completed in {:.2f} seconds.\n",
                             t0.seconds());

    // save result to tiff
    // parse output filename from config
    auto output = config["output"].as_table();
    std::string output_file = (*output)["filename"].value_or("recon.tiff");
    tomocam::tiff::write(output_file, recon);
    return 0;
}
