#include <format>
#include <iostream>

#include "config.h"
#include "padding.h"
#include "polar_grid.h"
#include "tiff.h"
#include "timer.h"
#include "tomocam.h"

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cerr << std::format("Usage: {} <input.toml>\n", argv[0]);
        std::cerr << "Please see config_template.toml for an example input file.\n";
        return 1;
    }

    toml::table config = tomocam::read_toml_file(argv[1]);
    auto datasets = tomocam::parse_input_datasets<float>(config);
    auto params = tomocam::ReconParams(config);
    auto output = tomocam::OutputParams(config);

    auto &projs = std::get<0>(datasets[0]);
    auto &theta = std::get<1>(datasets[0]);
    float gamma = std::get<2>(datasets[0]);
    size_t thickness = params.recon_dims[0];

    std::cerr << std::format("Projections size: ({}, {})\n", projs.nrows(), projs.ncols());
    std::cerr << std::format("No. of projections: {}\n", projs.nslices());
    std::cerr << std::format("Number of angles: {}\n", theta.size());

    // pad projections
    tomocam::Timer t0;
    t0.start();
    auto projs2 = tomocam::pad2d<float>(projs, params.PAD_FACTOR, tomocam::PadType::SYMMETRIC);
    t0.stop();
    std::cerr << std::format("Time to pad projections: {}(s)\n", t0.seconds());
    std::cerr << std::format("Padded projections size: ({}, {})\n", projs2.ncols(),
                             projs2.nrows());

    // create a polar grid
    t0.start();
    auto nrows = projs2.nrows();
    auto ncols = projs2.ncols();
    tomocam::PolarGrid<float> pgrid(theta, 0, nrows, ncols);
    t0.stop();
    std::cerr << std::format("Polar grid size: ({}, {})\n", nrows, ncols);

    // do the backprojection
    tomocam::dims_t dims = {thickness, nrows, ncols};
    std::cerr << std::format("Backprojecting to size: ({}, {}, {})\n", dims.n1,
                             dims.n2, dims.n3);
    t0.start();
    auto img = tomocam::backproj(projs2, pgrid, dims, gamma);
    t0.stop();
    std::cerr << std::format("Time to backproject: {}(s)\n", t0.seconds());

    // crop the image to original size
    tomocam::dims_t crop_dims = {thickness, projs.nrows(), projs.ncols()};
    t0.start();
    img = tomocam::crop2d<float>(img, crop_dims, tomocam::PadType::SYMMETRIC);
    t0.stop();
    std::cerr << std::format("Cropped image size: ({}, {}, {})\n", img.nslices(),
                             img.nrows(), img.ncols());
    std::cerr << std::format("Time to crop image: {}(s)\n", t0.seconds());

    // save data to tiff-stack
    tomocam::tiff::write(output.filepath, img);

    return 0;
}
