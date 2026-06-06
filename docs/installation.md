# Installation

## Build Dependencies

Tomocam requires the following dependencies:

- CMake ≥ 3.20
- C++ compiler with C++20 support (LLVM recommended)
- OpenMP
- Intel TBB (Threading Building Blocks)
- FFTW3 (single and double precision)
- libtiff
- FINUFFT
- CUDA (optional, for GPU acceleration)

## Supported Platforms

- Linux (tested on Arch Linux)
- macOS (15.5+)

## Building from Source

### Linux (Arch)

```bash
cmake --preset arch
cmake --build --preset arch
```

### macOS

```bash
cmake --preset macos
cmake --build --preset macos
```

## Building with Tests

To enable tests during the build:

```bash
# Linux
cmake --preset arch -DENABLE_TESTS=ON
cmake --build --preset arch
ctest --preset arch

# macOS
cmake --preset macos -DENABLE_TESTS=ON
cmake --build --preset macos
ctest --preset macos
```

## Installation

After building, install the library:

```bash
cmake --install . --prefix /path/to/install
```

This installs the binary and libraries to the specified prefix. Then set up your environment:

```bash
source setup.sh
```

which configures `PATH` and `LD_LIBRARY_PATH` for the installed layout.
