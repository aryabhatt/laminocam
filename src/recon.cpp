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

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "tomocam.h"
#include "timer.h"

void dump_config() {
    std::ofstream outf("config_template.json", std::ios::out);
    outf << "{\n";
    outf << "  \"filename\": \"path/to/dataset.tif\",\n";
    outf << "  \"angles\": \"path/to/angles.txt\",\n";
    outf << "  \"max_iter\": 50,\n";
    outf << "  \"sigma\": 100.0,\n";
    outf << "  \"p\": 1.2,\n";
    outf << "  \"tol\": 1e-5,\n";
    outf << "  \"xtol\": 1e-5,\n";
    outf << "  \"thickness\": 21\n";
    outf << "}\n";
    outf.close();
}

int main(int argc, char **argv) {

    // paser JSON input
    if (argc < 2) {
        std::cerr << std::format("Usage: {} <input.json>\n", argv[0]);
        std::cerr << "Please see config_template.json for an example input file.\n";
        dump_config();
        return 1;
    }

    std::string input_file = argv[1];
    std::ifstream ifs(input_file);
    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open input file " << input_file << "\n";
        return 1;
    }
    auto config = json::parse(ifs);
    ifs.close();
    auto dataset = config["filename"].get<std::string>();
    auto projs = tomocam::tiff::read(dataset);

    // read projection angles 
    std::string angles_file = config["angles"];
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

    // find keys in json, else set default values
    size_t max_iter =
        config.contains("max_iter") ? config["max_iter"].get<size_t>() : 50;
    float lambda = config.contains("lambda") ? config["lambda"].get<float>() : 0.1f;
    float mu = config.contains("mu") ? config["mu"].get<float>() : 5.0f;
    float tol = config.contains("tol") ? config["tol"].get<float>() : 1e-5f;
    float xtol = config.contains("xtol") ? config["xtol"].get<float>() : 1e-5f;
    size_t thickness =
        config.contains("thickness") ? config["thickness"].get<size_t>() : 21;

    // ensure angles are in radians
    auto max_angle = *std::max_element(angles.begin(), angles.end());
    if (std::abs(max_angle) > 2 * M_PI) {
        for (auto &angle : angles) { angle = angle * M_PI / 180.0f; }
    }
    // print parameters
    std::cout << "Reconstruction parameters:\n";
    std::cout << std::format("  Dataset: {}\n", dataset);
    std::cout << std::format("  Projections: {} x {} x {}\n", projs.nrows(),
                             projs.ncols(), projs.nslices());
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
    tomocam::tiff::write("recon.tif", recon);
    return 0;
}
