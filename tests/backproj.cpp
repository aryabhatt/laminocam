#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <ostream>
#include <string>
using json = nlohmann::json;

#include "padding.h"
#include "polar_grid.h"
#include "tiff.h"
#include "timer.h"
#include "tomocam.h"

constexpr double PADDING = 1.4142;

void usage(char **argv) {
    std::cout << "Usage: " << argv[0] << " JSON input configuration" << std::endl;
    exit(0);
}

int main(int argc, char **argv) {

    // sanity check
    if (argc < 2) usage(argv);

    // read data
    std::ifstream json_file(argv[1]);
    if (!json_file.is_open()) {
        std::cerr << std::format("Usage: {} JSON input configuration\n", argv[0]);
        return 1;
    }

    // get input data
    json config = json::parse(json_file);
    std::string filename = config["filename"];
    std::string angles = config["angles"];
    std::string output = config["output"];
    float gamma = config["gamma"];
    size_t thickness = config["thickness"];
    json_file.close();

    tomocam::Timer t0;
    t0.start();
    auto projs = tomocam::tiff::read(filename);
    t0.stop();
    std::cerr << std::format("Time to read data: {}(s)\n", t0.seconds());
    std::cerr << std::format("Projections size: ({}, {})\n", projs.nrows(),
                             projs.ncols());
    std::cerr << std::format("No. of projections: {}\n", projs.nslices());

    std::vector<float> theta;
    std::ifstream angles_file(angles);
    if (!angles_file.is_open()) {
        std::cerr << std::format("Could not open angles file: {}\n", angles);
        return 1;
    }
    float angle;
    while (angles_file >> angle) { theta.push_back(angle); }
    angles_file.close();
    // convert degrees to radians
    for (auto &a : theta) { a = a * M_PI / 180.0f; }
    std::cerr << std::format("Number of angles: {}\n", theta.size());

    // padd projections
    t0.start();
    auto projs2 = tomocam::pad2d<float>(projs, PADDING, tomocam::PadType::SYMMETRIC);
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

    // do the forward projection
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
    tomocam::tiff::write(output, img);

    return 0;
}
