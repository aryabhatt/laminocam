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

#include <algorithm>
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
#include "hdfread.h"
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

    // Read per-projection shifts from a two-column text file (dx dy per line, pixels).
    template <typename T>
    inline std::vector<std::array<T, 2>> read_shifts_file(const std::string &filepath) {
        std::ifstream fp(filepath);
        if (!fp.is_open()) {
            throw std::runtime_error(
                std::format("Could not open shifts file: {}", filepath));
        }
        std::vector<std::array<T, 2>> shifts;
        T dx, dy;
        while (fp >> dx >> dy) { shifts.push_back({dx, dy}); }
        if (shifts.empty()) {
            throw std::runtime_error(
                std::format("No shifts found in file: {}", filepath));
        }
        return shifts;
    }

    // parse aligment parameters from TOML
    template <typename T>
    inline std::vector<std::array<T, 6>> parse_alignment(const toml::table &align,
                                                         const char *section) {
        std::vector<std::array<T, 6>> rots;
        auto *projs = align[section]["projections"].as_array();
        if (!projs)
            throw std::runtime_error(std::string("missing [[") + section +
                                     ".projections]]");
        for (auto &elem : *projs) {
            auto *row = elem.as_table();
            auto *rt = row->get("rotation_T")->as_array();
            std::array<T, 6> r;
            for (int i = 0; i < 6; ++i) r[i] = (*rt)[i].value<T>().value();
            rots.push_back(r);
        }
        return rots;
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

            //  check for required fields: filename, gamma
            if (!input_table->contains("filename") ||
                !input_table->contains("gamma")) {
                throw std::runtime_error(
                    "[[input]] entry must have 'filename' and 'gamma' fields");
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
            auto gamma = (*input_table)["gamma"].value<T>();
            if (!gamma.has_value()) {
                throw std::runtime_error("[[input]] 'gamma' field must be a number");
            }
            // read beta (misalignment with y-axis) if provided,
            T beta = (*input_table)["beta"].value_or<T>(0);
            T beta_rad = beta * T(M_PI) / T(180);

            // Build per-projection shifts vector.
            // 'shifts' (path to a two-column file) takes priority over the legacy
            // 'cor-offset' scalar, which is broadcast to all projections.
            std::vector<std::array<T, 2>> per_proj_shifts;
            if (input_table->contains("shifts")) {
                auto shifts_path = (*input_table)["shifts"].value<std::string>();
                if (!shifts_path.has_value())
                    throw std::runtime_error(
                        "[[input]] 'shifts' must be a string path");
                auto resolved = std::filesystem::path(*shifts_path);
                if (!std::filesystem::exists(resolved))
                    throw std::runtime_error(
                        std::format("Shifts file does not exist: {}", resolved.string()));
                per_proj_shifts = read_shifts_file<T>(resolved.string());
            } else if (input_table->contains("cor-offset")) {
                auto offsets_array = (*input_table)["cor-offset"].as_array();
                if (!offsets_array || offsets_array->size() != 2) {
                    throw std::runtime_error(
                        "[[input]] 'cor-offset' must be in form [dx, dy]");
                }
                std::array<T, 2> buf;
                for (size_t i = 0; i < 2; ++i) {
                    auto val = (*offsets_array)[i].value<T>();
                    if (!val.has_value()) {
                        throw std::runtime_error(
                            "[[input]] 'cor-offset' must be an array of numbers");
                    }
                    buf[i] = *val;
                }
                // broadcast to all projections after we know N (filled below)
                per_proj_shifts = {buf};  // sentinel: one element means broadcast
            }

            // deg→rad conversion applied to any angles vector
            auto to_radians = [](std::vector<T> &a) {
                auto mx = *std::max_element(a.begin(), a.end());
                if (std::abs(mx) > 2 * M_PI)
                    for (auto &v : a) v = v * M_PI / T(180);
            };

            // dispatch on filename extension
            auto ext = std::filesystem::path(*filename).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            Array<float> projs;
            std::vector<T> angles;

            if (ext == ".tiff" || ext == ".tif") {
                auto angles_path = (*input_table)["angles"].value<std::string>();
                if (!angles_path.has_value())
                    throw std::runtime_error(
                        "[[input]] 'angles' is required for TIFF files");
                if (!std::filesystem::exists(*angles_path))
                    throw std::runtime_error(
                        std::format("Angles file does not exist: {}", *angles_path));
                projs = tomocam::tiff::read(*filename);
                angles = read_angles_file<T>(*angles_path);
            } else if (ext == ".h5" || ext == ".hdf5") {
                auto angles_ds =
                    (*input_table)["angles"].value_or<std::string>("/coords/alpha");
                projs = tomocam::h5::read_images(*filename);
                auto raw = tomocam::h5::read_angles(*filename, angles_ds);
                angles.assign(raw.begin(), raw.end());
                to_radians(angles);
            } else {
                throw std::runtime_error(std::format(
                    "Unsupported file extension '{}': {}", ext, *filename));
            }

            projs = tomocam::mask_infs_nans(projs);
            // alignlsq uses Rz(γ)=[[c,s],[−s,c]] (passive/CW);
            // RotationTranspose uses standard CCW Rz. Negate γ to reconcile.
            T gamma_rad = -(*gamma) * T(M_PI) / T(180);

            // broadcast scalar cor-offset to all N projections if needed
            if (per_proj_shifts.size() == 1) {
                std::array<T, 2> scalar = per_proj_shifts[0];
                per_proj_shifts.assign(angles.size(), scalar);
            } else if (per_proj_shifts.empty()) {
                per_proj_shifts.assign(angles.size(), std::array<T, 2>{T(0), T(0)});
            } else if (per_proj_shifts.size() != angles.size()) {
                throw std::runtime_error(std::format(
                    "shifts file has {} entries but {} projections were loaded",
                    per_proj_shifts.size(), angles.size()));
            }

            datasets.push_back(
                {std::move(projs), std::move(angles), gamma_rad, beta_rad,
                 std::move(per_proj_shifts)});
        }
        return datasets;
    }

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
        outfile << "# HDF5 alternative (angles read from /coords/alpha inside the "
                   "file):\n";
        outfile << "# [[input]]\n";
        outfile << "# filename = \"/path/to/projections.h5\"\n";
        outfile << "# gamma = 0\n";
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
