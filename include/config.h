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
#include "config_templates.h"
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

    // Resolve a path from the config: relative paths are taken relative to the
    // config file's directory, falling back to the current directory.
    inline std::string resolve_path(const std::string &p,
                                    const std::filesystem::path &base_dir) {
        std::filesystem::path fp(p);
        if (fp.is_relative() && !base_dir.empty()) {
            auto candidate = base_dir / fp;
            if (std::filesystem::exists(candidate)) return candidate.string();
        }
        return p;
    }

    // Warn about keys in a table that are not in the allowed list.
    inline void warn_unknown_keys(const toml::table &tbl,
                                  std::initializer_list<const char *> allowed,
                                  const std::string &section) {
        for (auto &&[key, val] : tbl) {
            bool known = false;
            for (auto *a : allowed) {
                if (key.str() == a) {
                    known = true;
                    break;
                }
            }
            if (!known) {
                std::cerr << std::format(
                    "\033[33mWarning\033[0m: unknown key '{}' in {} (ignored)\n",
                    std::string(key.str()), section);
            }
        }
    }

    // Convert angles to radians according to "deg" or "rad".
    template <typename T>
    inline void angles_to_radians(std::vector<T> &angles, const std::string &units) {
        if (units == "deg") {
            for (auto &a : angles) a = a * T(M_PI) / T(180);
        } else if (units != "rad") {
            throw std::runtime_error(std::format(
                "[[input]] 'angle_units' must be \"deg\" or \"rad\", got '{}'",
                units));
        }
    }

    // Function to read angles from a text file (values in the given units,
    // returned in radians)
    template <typename T>
    inline std::vector<T> read_angles_file(const std::string &filepath,
                                           const std::string &units = "deg") {
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
        angles_to_radians(angles, units);
        return angles;
    }

    // Read per-projection shifts from a two-column text file (dx dy per line,
    // pixels).
    template <typename T>
    inline std::vector<std::array<T, 2>>
    read_shifts_file(const std::string &filepath) {
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
    parse_input_datasets(const toml::table &config,
                         const std::filesystem::path &base_dir = {}) {

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

            //  check for required fields: filename
            if (!input_table->contains("filename")) {
                throw std::runtime_error(
                    "[[input]] entry must have a 'filename' field");
            }
            warn_unknown_keys(*input_table,
                              {"filename", "angles", "angle_units", "gamma", "beta",
                               "shifts", "cor-offset", "alignment", "projs_dataset"},
                              "[[input]]");
            auto filename = (*input_table)["filename"].value<std::string>();
            if (!filename.has_value()) {
                throw std::runtime_error("[[input]] 'filename' must be a string");
            }
            *filename = resolve_path(*filename, base_dir);
            auto angle_units =
                (*input_table)["angle_units"].value_or<std::string>("deg");
            // check if file exists
            if (!std::filesystem::exists(*filename)) {
                throw std::runtime_error(
                    std::format("Projection file does not exist: {}", *filename));
            }
            // gamma and beta default to 0; overwritten by alignment block if present
            T gamma_rad = -(*input_table)["gamma"].value_or<T>(0) * T(M_PI) / T(180);
            T beta_rad = (*input_table)["beta"].value_or<T>(0) * T(M_PI) / T(180);

            // Build per-projection shifts vector.
            // 'shifts' (path to a two-column file) takes priority over the legacy
            // 'cor-offset' scalar, which is broadcast to all projections.
            std::vector<std::array<T, 2>> per_proj_shifts;
            if (input_table->contains("shifts")) {
                auto shifts_path = (*input_table)["shifts"].value<std::string>();
                if (!shifts_path.has_value())
                    throw std::runtime_error(
                        "[[input]] 'shifts' must be a string path");
                auto resolved =
                    std::filesystem::path(resolve_path(*shifts_path, base_dir));
                if (!std::filesystem::exists(resolved))
                    throw std::runtime_error(std::format(
                        "Shifts file does not exist: {}", resolved.string()));
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
                per_proj_shifts = {buf}; // sentinel: one element means broadcast
            }

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
                *angles_path = resolve_path(*angles_path, base_dir);
                if (!std::filesystem::exists(*angles_path))
                    throw std::runtime_error(
                        std::format("Angles file does not exist: {}", *angles_path));
                projs = tomocam::tiff::read(*filename);
                angles = read_angles_file<T>(*angles_path, angle_units);
                if (projs.nslices() != angles.size())
                    throw std::runtime_error(std::format(
                        "'{}' has {} pages but '{}' has {} entries", *filename,
                        projs.nslices(), *angles_path, angles.size()));
            } else if (ext == ".h5" || ext == ".hdf5") {
                auto angles_ds =
                    (*input_table)["angles"].value_or<std::string>("/coords/alpha");
                auto projs_ds =
                    (*input_table)["projs_dataset"].value_or<std::string>("/images");
                projs = tomocam::h5::read_images(*filename, projs_ds);
                auto raw = tomocam::h5::read_angles(*filename, angles_ds);
                angles.assign(raw.begin(), raw.end());
                angles_to_radians(angles, angle_units);
                if (projs.nslices() != angles.size())
                    throw std::runtime_error(std::format(
                        "'{}' images has {} slices but angles dataset '{}' has {} "
                        "entries",
                        *filename, projs.nslices(), angles_ds, angles.size()));
            } else {
                throw std::runtime_error(std::format(
                    "Unsupported file extension '{}': {}", ext, *filename));
            }

            projs = tomocam::mask_infs_nans(projs);
            // alignlsq uses Rz(γ)=[[c,s],[−s,c]] (passive/CW);
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

            // Override angles and shifts from an alignment TOML if provided.
            if (input_table->contains("alignment")) {
                auto align_path = (*input_table)["alignment"].value<std::string>();
                if (!align_path.has_value())
                    throw std::runtime_error(
                        "[[input]] 'alignment' must be a string path");
                *align_path = resolve_path(*align_path, base_dir);
                if (!std::filesystem::exists(*align_path))
                    throw std::runtime_error(std::format(
                        "Alignment file does not exist: {}", *align_path));
                auto align_tbl = read_toml_file(*align_path);

                // Override gamma and beta from alignment TOML
                if (auto gv = align_tbl["gamma_deg"].value<T>())
                    gamma_rad = -(*gv) * T(M_PI) / T(180);
                if (auto bv = align_tbl["beta_deg"].value<T>())
                    beta_rad = (*bv) * T(M_PI) / T(180);

                auto *ang_arr = align_tbl["corrected_angles_deg"].as_array();
                if (!ang_arr)
                    throw std::runtime_error(
                        "Alignment TOML missing 'corrected_angles_deg'");
                angles.clear();
                for (auto &elem : *ang_arr)
                    angles.push_back(elem.value<T>().value() * T(M_PI) / T(180));

                auto *spx = align_tbl["shifts_px"].as_array();
                if (spx) {
                    per_proj_shifts.clear();
                    for (auto &elem : *spx) {
                        auto *pair = elem.as_array();
                        if (!pair || pair->size() < 2)
                            throw std::runtime_error(
                                "Alignment TOML 'shifts_px' entries must be [dx, "
                                "dy] pairs");
                        T dx = (*pair)[0].value<T>().value();
                        T dy = (*pair)[1].value<T>().value();
                        per_proj_shifts.push_back({dx, dy});
                    }
                }

                if (per_proj_shifts.size() != angles.size())
                    throw std::runtime_error(
                        std::format("Alignment TOML: shifts_px has {} entries but "
                                    "corrected_angles_deg has {}",
                                    per_proj_shifts.size(), angles.size()));
            }

            datasets.push_back({std::move(projs), std::move(angles), gamma_rad,
                                beta_rad, std::move(per_proj_shifts)});
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
        warn_unknown_keys(*recon,
                          {"max_iters", "tol", "xtol", "recon_dims", "regularizer"},
                          "[recon_params]");
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
        p.tol = (*recon)["tol"].value_or<float>(1e-5f);
        p.xtol = (*recon)["xtol"].value_or<float>(1e-5f);
        if (recon->contains("regularizer")) {
            auto reg = (*recon)["regularizer"].as_table();
            if (reg) {
                warn_unknown_keys(*reg, {"method", "split_bregman"},
                                  "[recon_params.regularizer]");
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
                    warn_unknown_keys(*params, {"lambda", "mu", "inner_iters"},
                                      "[recon_params.regularizer.split_bregman]");
                    p.lambda = (*params)["lambda"].value_or<float>(0.1f);
                    p.mu = (*params)["mu"].value_or<float>(10.0f);
                    p.innerIters = (*params)["inner_iters"].value_or<size_t>(1);
                } else {
                    throw std::runtime_error(
                        "[recon_params] 'regularizer' must be 'split_bregman'");
                }
            }
        }
        return p;
    };

    // Laminography-only check: the sample thickness (recon_dims[0]) is expected
    // to be much smaller than the in-plane dimensions.
    inline void warn_laminography_dims(const ReconParams &p) {
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
    }

    // Output parameters
    inline OutputParams parse_output_params(const toml::table &config) {

        OutputParams params;
        // Read [output] section
        auto output = config["output"];
        if (!output) {
            throw std::runtime_error("Missing [output] section in config file");
        }
        if (auto *out_tbl = output.as_table())
            warn_unknown_keys(*out_tbl, {"filename", "formats"}, "[output]");
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

    // Write an example configuration file from the template embedded at build
    // time (config_template_lamino.toml or config_template_tomo.toml).
    inline void dump_config(const char *tmpl, const std::string &filepath) {
        std::ofstream outfile(filepath);
        if (!outfile.is_open()) {
            throw std::runtime_error(
                std::format("Could not open file for writing: {}", filepath));
        }
        outfile << tmpl;
    }
} // namespace tomocam
#endif // CONFIG_H
