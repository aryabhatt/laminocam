#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <ostream>
#include <string>
using json = nlohmann::json;

#include "padding.h"
#include "tiff.h"
#include "timer.h"
#include "tomocam.h"

constexpr double PADDING = 1.4142;

void usage(char **argv) {
    std::cout << "Usage: " << argv[0] << " JSON input configuration\n";
    exit(0);
}

int main(int argc, char **argv) {

    // sanity check
    if (argc < 2) { usage(argv); }

    // read data
    std::ifstream json_file(argv[1]);
    if (!json_file.is_open()) {
        std::cerr << "error: can't open JSON file: " << argv[1] << '\n';
        return 1;
    }

    // get input data
    json config = json::parse(json_file);
    std::string filename = config["filename"];
    std::string angles = config["angles"];
    std::string output = config["output"];
    float gamma = config["gamma"];

    tomocam::Timer t0;
    t0.start();
    auto sample = tomocam::tiff::read(filename);
    t0.stop();
    std::cerr << "Time to read data: " << t0.seconds() << "(s)\n";

    // read angles
    std::vector<float> theta;
    std::ifstream fp(angles);
    if (!fp.is_open()) {
        std::cerr << "error: can't open angles file: " << angles << '\n';
        return 1;
    }
    float angle;
    while (fp >> angle) { theta.push_back(angle); }
    fp.close();
    // convert angles to radians
    for (auto &a : theta) { a = a * M_PI / 180.0f; }

    // pad the sample
    t0.start();
    auto sample2 =
        tomocam::pad3d<float>(sample, PADDING, tomocam::PadType::SYMMETRIC);
    t0.stop();
    std::cerr << "Time to pad data: " << t0.seconds() << "(s)\n";

    // create a polar grid
    t0.start();
    // oversample polar-grid
    auto nrows = sample2.nrows();
    auto ncols = sample2.ncols();
    tomocam::PolarGrid<float> pgrid(theta, 0, nrows, ncols);
    t0.stop();
    std::cerr << "Time to build a polar grid: " << t0.seconds() << "(s)\n";

    // do the forward projection
    t0.start();
    auto proj = tomocam::forward(sample2, pgrid, gamma);
    t0.stop();
    std::cerr << "time to do forward projection: " << t0.seconds() << "(s)\n";

    // crop the projection to original size
    tomocam::dims_t crop_dims = {theta.size(), sample.nrows(), sample.ncols()};
    t0.start();
    auto proj2 =
        tomocam::crop2d<float>(proj, crop_dims, tomocam::PadType::SYMMETRIC);
    t0.stop();
    std::cerr << "Time to crop data: " << t0.seconds() << "(s)\n";
    // save data to tiff-stack
    tomocam::tiff::write(output, proj2);

    return 0;
}
