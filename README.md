![Documentation Status](https://readthedocs.org/projects/laminocam/badge/?version=latest)

![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)

![CI](https://github.com/username/laminocam/actions/workflows/ci.yml/badge.svg)

# LaminoCam

A C++ library for tomographic reconstruction of laminographic data, with missing cone correction and Model-Based Iterative Reconstruction (MBIR) capabilities. Developed at Lawrence Berkeley National Laboratory.

## Overview

LaminoCam is a high-performance library developed at Lawrence Berkeley National Laboratory for advanced reconstruction of laminographic data. It provides forward and backward projection operators, iterative reconstruction algorithms, and Model-Based Iterative Reconstruction.
The library is optimized for performance using:
- OpenMP parallelization
- Intel TBB (Threading Building Blocks)
- FFTW for fast Fourier transforms
- FINUFFT for non-uniform FFTs
- Optional GPU acceleration via CUDA

## Features

- **Forward/Backward Projection**: Efficient projection operators on polar grids
- **Iterative Reconstruction**: Conjugate gradient and Nesterov accelerated gradient methods
- **MBIR**: Model-Based Iterative Reconstruction L1 regularization via Split Bregman method.
- **TIFF I/O**: Read and write reconstruction data

## Requirements

### Build Dependencies

- CMake ≥ 3.20
- C++ compiler with C++20 support (llvm recommended)
- OpenMP
- Intel TBB
- FFTW3 (single and double precision)
- libtiff
- FINUFFT
- CUDA for GPU acceleration

### Supported Platforms

- Linux (tested on Arch Linux)
- macOS (15.5+)

## Building

### Configure and Build

```bash
# Linux
cmake --preset arch
cmake --build --preset arch

# macOS
cmake --preset macos
cmake --build --preset macos
```

## Usage

### Running Reconstruction

The reconstruction tool uses a TOML configuration file to specify input data and parameters:

```bash
./build/recon_scalar <config.toml>
```

### TOML Configuration File

Create a TOML file with the following structure:

```toml
[input]
filename = "/path/to/projections.tif"   # Input TIFF file with projection data
angles = "/path/to/angles.txt"              # Text file with projection angles

[output]
filename = "output.tiff"                       # Output filename for reconstruction

[recon_params]
max_iters = 100                          # Maximum outer iterations
inner_iters = 1                         # Inner iterations (for split_bregman, and line-serach in NAG)
tol = 1e-5                              # Convergence tolerance
xtol = 1e-5                             # X-tolerance for convergence
recon_dims = [21, 511, 511]                          # Reconstruction volume dimensions

[recon_params.optimizer]
method = "split_bregman"                # Optimizer: "split_bregman", "conjugate_gradient", or "nag_optimizer"

# Parameters for split_bregman method
[recon_params.optimizer.split_bregman]
lambda = 0.1                           # Regularization parameter
mu = 10                                 # Penalty parameter

# Parameters for nag_optimizer method (if using)
# [recon_params.optimizer.nag_optimizer]
# sigma = 0.1
# p = 1.2
```

**Notes:**
- Angles are expected be in degrees
- The angles file should be is ASCII format and contain one angle per line
- A template configuration file (`config_template.toml`) is generated automatically if no input is provided

## Documentation
Documentation is a work in progress. The latest version is available in readthedocs: https://laminocam.readthedocs.io/en/latest/

## License

Copyright (c) 2018, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.

This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so.

## Contributing

## Contact

For questions or issues, please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
