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

#ifndef CONFIG_H
#define CONFIG_H

#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <toml++/toml.h>
#include <tuple>
#include <vector>

#include "array.h"
#include "mask.h"
#include "recon_params.h"
#include "tiff.h"

namespace tomocam {

    // Function to read and parse a TOML file
    inline toml::table read_toml_file(const std::string &filepath) {
        if (!std::filesystem::exists(filepath)) {
            throw std::runtime_error(
                std::format("TOML file does not exist: {}", filepath));
        }

        try {
            return toml::parse_file(filepath);
        } catch (const toml::parse_error &err) {
            throw std::runtime_error(std::format(
                "Failed to parse TOML file '{}': {}", filepath, err.description()));
        }
    }

    // Function to read angles from a text file
    template <typename T>
    inline std::vector<T> read_angles_file(const std::string &filepath) {
        std::ifstream fp(filepath);
        if (!fp.is_open()) {
            throw std::runtime_error(
                std::format("Could not open angles file: {}", filepath));
        }

        std::vector<T> angles;
        T angle;
        while (fp >> angle) { angles.push_back(angle); }
        if (angles.empty()) {
            throw std::runtime_error(
                std::format("No angles found in file: {}", filepath));
        }

        // Convert to radians if necessary
        auto max_angle = *std::max_element(angles.begin(), angles.end());
        if (std::abs(max_angle) > 2 * M_PI) {
            for (auto &a : angles) { a = a * M_PI / (T)180.0; }
        }
        return angles;
    }

    // Function to parse input datasets from TOML config
    template <typename T>
    [[nodiscard]] std::vector<Dataset_t<T>>
    parse_input_datasets(const toml::table &config) {

        auto input_array = config["input"].as_array();
        if (!input_array) {
            throw std::runtime_error("Missing [[input]] array in TOML file");
        }

        std::vector<Dataset_t<T>> datasets;

        for (auto &elem : *input_array) {
            auto input_table = elem.as_table();
            if (!input_table) {
                throw std::runtime_error("Invalid [[input]] entry");
            }

            //  check for required fields: filename, angles, gamma
            if (!input_table->contains("filename") ||
                !input_table->contains("angles") ||
                !input_table->contains("gamma")) {
                throw std::runtime_error("[[input]] entry must have 'filename', "
                                         "'angles', and 'gamma' fields");
            }
            auto filename = (*input_table)["filename"].value<std::string>();
            if (!filename.has_value()) {
                throw std::runtime_error("[[input]] 'filename' must be a string");
            }
            // check if file exists
            if (!std::filesystem::exists(*filename)) {
                throw std::runtime_error(
                    std::format("Projection file does not exist: {}", *filename));
            }
            auto angles_file = (*input_table)["angles"].value<std::string>();
            if (!angles_file.has_value()) {
                throw std::runtime_error("[[input]] 'angles' must be a string");
            }
            // check if file exists
            if (!std::filesystem::exists(*angles_file)) {
                throw std::runtime_error(
                    std::format("Angles file does not exist: {}", *angles_file));
            }
            auto gamma = (*input_table)["gamma"].value<T>();
            if (!gamma.has_value()) {
                throw std::runtime_error("[[input]] 'gamma' field must be a number");
            }

            auto projs = tomocam::tiff::read(*filename);
            projs = tomocam::mask_infs_nans(projs);
            auto angles = read_angles_file<T>(*angles_file);
            auto gamma_rad = *gamma * M_PI / (T)180.0; // convert to radians
            datasets.push_back(
                std::make_tuple(std::move(projs), std::move(angles), gamma_rad));
        }
        return datasets;
    }

    /***

      enum class Regularizer { SPLIT_BREGMAN, UNCONSTRAINED };
    // Reconstruction parameters
    struct ReconParams {
        Regularizer regularizer =
            Regularizer::UNCONSTRAINED;               // Regularization method
        std::array<size_t, 3> recon_dims = {0, 0, 0}; // Reconstruction dimensions
        size_t maxIters = 50;                         // Maximum number of iterations
        size_t innerIters = 3;     // Number of inner iterations for Split-Bregman
        float lambda = 0.1f;       // Regularization weight (Split-Bregman)
        float mu = 10.0f;          // Augmented Lagrangian parameter (Split-Bregman)
        float tol = 1e-5f;         // Tolerance for convergence
        float xtol = 1e-5f;        // Tolerance for solution change
        float PAD_FACTOR = 1.4142; // sqrt(2) padding factor

        void print(std::ostream &os) const {

            std::string reg_str = (regularizer == Regularizer::SPLIT_BREGMAN)
                                      ? "Split-Bregman"
                                      : "Unconstrained";
            os << "Reconstruction Parameters:\n";
            os << "  max_outer_iters: " << maxIters << "\n";
            os << std::format("  recon_dims: [{}, {}, {}]\n", recon_dims[0],
                              recon_dims[1], recon_dims[2]);
            os << "  tol: " << tol << "\n";
            os << "  xtol: " << xtol << "\n";
            os << "  regularizer: " << reg_str << "\n";
            if (regularizer == Regularizer::SPLIT_BREGMAN) {
                os << "    inner_iters: " << innerIters << "\n";
                os << "    lambda: " << lambda << "\n";
                os << "    mu: " << mu << "\n";
            }
        }
    };
    ****/

    inline ReconParams parse_recon_params(const toml::table &config) {
        ReconParams p;

        // Read [recon_params] section
        auto recon = config["recon_params"].as_table();
        if (!recon) {
            throw std::runtime_error(
                "Missing [recon_params] section in config file");
        }
        p.maxIters = (*recon)["max_iters"].value_or<size_t>(50);

        const auto *dims = (*recon)["recon_dims"].as_array();
        if (dims && dims->size() == 3) {
            for (size_t i = 0; i < 3; ++i) {
                size_t temp = (*dims)[i].value_or<size_t>(0);
                if (temp == 0) {
                    throw std::runtime_error(
                        std::format("[recon_params] 'recon_dims[{}]' must be a "
                                    "positive integer",
                                    i));
                }
                if (temp % 2 == 0) temp -= 1;
                p.recon_dims[i] = temp;
            }
        } else {
            throw std::runtime_error("[recon_params] 'recon_dims' must be an "
                                     "array of three integers");
        }
        float ratio = static_cast<float>(p.recon_dims[0]) /
                      static_cast<float>(p.recon_dims[2]);
        if (ratio > 0.15f) {
            std::cerr << std::format(
                "\033[31mWarning\033[0m: recon_dims[0] / recon_dims[2] = "
                "{:.2f} > 0.15.\n"
                "laminography thickness (recon_dims[0]) "
                "is expected to be much smaller than the in-plane "
                "dimensions (recon_dims[1], recon_dims[2]).\n",
                ratio);
        }
        p.tol = (*recon)["tol"].value_or<float>(1e-5f);
        p.xtol = (*recon)["xtol"].value_or<float>(1e-5f);
        if (recon->contains("regularizer")) {
            auto reg = (*recon)["regularizer"].as_table();
            if (reg) {
                auto reg_str =
                    (*reg)["method"].value_or<std::string>("split_bregman");
                if (reg_str == "split_bregman") {
                    p.regularizer = Regularizer::SPLIT_BREGMAN;
                    auto params = (*reg)["split_bregman"].as_table();
                    if (!params) {
                        throw std::runtime_error(
                            "Missing [recon_params.regularizer.split_bregman] "
                            "section in config file");
                    }
                    p.lambda = (*params)["lambda"].value_or<float>(0.1f);
                    p.mu = (*params)["mu"].value_or<float>(10.0f);
                    p.innerIters = (*params)["inner_iters"].value_or<size_t>(3);
                } else {
                    throw std::runtime_error(
                        "[recon_params] 'regularizer' must be 'split_bregman'");
                }
            }
        }
        return p;
    };

    // Output parameters
    inline OutputParams parse_output_params(const toml::table &config) {

        OutputParams params;
        // Read [output] section
        auto output = config["output"];
        if (!output) {
            throw std::runtime_error("Missing [output] section in config file");
        }
        params.filepath = output["filename"].value_or<std::string>("./recon.tiff");

        // Read formats array
        std::vector<std::string> formats;
        auto formats_array = output["formats"].as_array();
        if (formats_array) {
            for (auto &elem : *formats_array) {
                auto fmt = elem.value<std::string>();
                if (!fmt.has_value()) {
                    throw std::runtime_error(
                        "[output] 'formats' array must contain strings");
                }
                if (*fmt != "tiff" && *fmt != "vti") {
                    throw std::runtime_error(std::format(
                        "[output] invalid format '{}'. Must be 'tiff' or 'vti'",
                        *fmt));
                }
                formats.push_back(*fmt);
            }
        } else {
            formats = {"tiff"};
        }

        // Remove duplicates while preserving order
        std::vector<std::string> unique_formats;
        for (const auto &fmt : formats) {
            if (std::find(unique_formats.begin(), unique_formats.end(), fmt) ==
                unique_formats.end()) {
                unique_formats.push_back(fmt);
            }
        }
        params.formats = std::move(unique_formats);
        return params;
    }

    // Function to dump an example configuration file
    inline void dump_config(const std::string &filepath = "config.toml") {
        std::ofstream outfile(filepath);
        if (!outfile.is_open()) {
            throw std::runtime_error(
                std::format("Could not open file for writing: {}", filepath));
        }

        outfile << "[[input]]\n";
        outfile << "filename = \"/path/to/projections.tiff\"\n";
        outfile << "angles = \"/path/to/angles.txt\"\n";
        outfile << "gamma = 0\n";
        outfile << "\n";
        outfile << "# add more [[input]] sections for additional datasets\n";
        outfile << "# [[input]]\n";
        outfile << "# filename = \"/path/to/projections2.tiff\"\n";
        outfile << "# angles = \"/path/to/angles.txt\"\n";
        outfile << "# gamma = 45\n";
        outfile << "\n";
        outfile << "[output]\n";
        outfile << "filename = \"output.tiff\"\n";
        outfile << "formats = [\"tiff\", \"vti\"]  # available: \"tiff\", \"vti\"\n";
        outfile << "\n";
        outfile << "[recon_params]\n";
        outfile << "max_iters = 50\n";
        outfile << "tol = 1e-5\n";
        outfile << "xtol = 1e-5\n";
        outfile << "recon_dims = [51, 511, 511]\n";
        outfile << "\n";
        outfile << "[recon_params.regularizer]\n";
        outfile << "method = \"split_bregman\"\n";
        outfile << "\n";
        outfile << "[recon_params.regularizer.split_bregman]\n";
        outfile << "lambda = 0.1\n";
        outfile << "mu = 10.0\n";
        outfile << "\n";
        outfile.close();
    }
} // namespace tomocam
#endif // CONFIG_H
