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

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <toml++/toml.h>
#include <vector>

#include "array_ops.h"
#include "config.h"
#include "padding.h"
#include "projection.h"
#include "tiff.h"
#include "timer.h"

#ifdef USE_GPU
#include "gpu/projection.h"
#endif

using namespace tomocam;

int main(int argc, char **argv) {

    // parse TOML input
    if (argc < 2) {
        std::cerr << std::format("Usage: {} <input.toml>\n", argv[0]);
        return 1;
    }
    /*
     * read TOML file
     * [input]
     * filename = "path/to/volume.tiff"
     *
     * [output]
     * basedir    = "path/to/output"
     * pad_factor = 1.5
     * proj_angles = { min = -90, max = 90, n = 180 }
     *
     * [[projections]]
     * gamma    = 0
     * filename = "gamma0.tiff"
     */
    auto cfg = toml::parse_file(argv[1]);

    if (!cfg.contains("input") || !cfg.contains("output")) {
        std::cerr
            << "Error: input and output sections are required in the TOML file.\n";
        return 1;
    }
    auto input = cfg["input"];
    // read 3D volume from input file
    auto input_file = input["filename"].value<std::string>().value_or("");
    if (!std::filesystem::exists(input_file)) {
        std::cerr << std::format("Error: input file {} does not exist.\n",
                                 input_file);
        return 1;
    }
    std::cout << std::format("Reading volume from: {}\n", input_file);
    auto volume = tomocam::tiff::read(input_file);
    std::cout << std::format("Volume loaded: {}x{}x{} (slices x rows x cols)\n",
                             volume.nslices(), volume.nrows(), volume.ncols());

    // parse output section
    auto output = cfg["output"];
    auto output_path =
        output["basedir"].value<std::string>().value_or(std::string{"."});
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directories(output_path);
    }
    auto pad_factor = output["pad_factor"].value<float>().value_or(1.0f);

    // generate angles from proj_angles table: [min, max) with n steps
    auto ang_cfg = output["proj_angles"];
    float ang_min = ang_cfg["min"].value<float>().value_or(-90.0f);
    float ang_max = ang_cfg["max"].value<float>().value_or(90.0f);
    int ang_n = ang_cfg["n"].value<int>().value_or(180);
    float ang_step = (ang_max - ang_min) / ang_n;
    std::vector<float> angles(ang_n);
    for (int i = 0; i < ang_n; i++) { angles[i] = ang_min + i * ang_step; }

    std::cout << std::format("Generating {} angles from {} to {} degrees\n", ang_n,
                             ang_min, ang_max);

    // write angles to angles.txt in output dir
    {
        std::ofstream angles_out(std::filesystem::path(output_path) / "angles.txt");
        for (auto a : angles) { angles_out << a << "\n"; }
    }
    std::cout << std::format(
        "Angles written to: {}\n",
        (std::filesystem::path(output_path) / "angles.txt").string());

    // parse [[projections]] array of tables
    struct Projection {
        float gamma;
        std::string filename;
    };
    std::vector<Projection> projections;
    if (auto arr = cfg["projections"].as_array()) {
        for (auto &elem : *arr) {
            if (auto tbl = elem.as_table()) {
                auto g_node = tbl->get("gamma");
                auto f_node = tbl->get("filename");
                if (g_node && f_node) {
                    projections.push_back(
                        {g_node->value<float>().value_or(0.0f),
                         f_node->value<std::string>().value_or("proj.tiff")});
                }
            }
        }
    }
    if (projections.empty()) { projections.push_back({0.0f, "proj.tiff"}); }

    auto pad = [pad_factor](size_t dim) {
        auto new_dim = static_cast<size_t>(dim * pad_factor);
        if (new_dim % 2 == 0) {
            new_dim -= 1; // make it odd
        }
        return new_dim;
    };

    // pad the volume
    dims_t padded_dims = {pad(volume.nslices()), pad(volume.nrows()),
                          pad(volume.ncols())};
    if (pad_factor > 1.0f) {
        std::cout << std::format("Padding volume to: {}x{}x{}\n", padded_dims.n1,
                                 padded_dims.n2, padded_dims.n3);
        volume = tomocam::pad3d(volume, padded_dims, tomocam::PadType::SYMMETRIC);
        std::cout << "Padding complete\n";
    }

    std::cout << std::format("Computing {} projection(s)\n", projections.size());

    // loop over projections
    int proj_idx = 0;
    for (auto &[gamma, filename] : projections) {
        proj_idx++;
        std::cout << std::format("[{}/{}] gamma={:.2f} deg -> {}\n", proj_idx,
                                 projections.size(), gamma, filename);

        std::vector<float> gamma_vec(angles.size(), gamma * M_PI / 180.0f);
        std::vector<float> theta_vec;
        for (const auto &a : angles) { theta_vec.push_back(a * M_PI / 180.0f); }
        auto pg = tomocam::cpu::PolarGrid(theta_vec, gamma_vec, volume.nrows(),
                                          volume.ncols());
        auto proj = tomocam::forward(volume, pg);

        std::string output_file = std::filesystem::path(output_path) / filename;
        tomocam::tiff::write(output_file, proj);
        std::cout << std::format("  Written: {}\n", output_file);
    }
    std::cout << "Done.\n";
    return 0;
}
