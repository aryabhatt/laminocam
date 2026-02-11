
cmake_minimum_required(VERSION 3.20)

project(Testbed CXX)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Add cuda
option(USE_GPU "Use GPU Acceleration" TRUE)
if (${USE_GPU})
  enable_language(CUDA)
endif()

# C++ std
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -march=native")
endif()

set(SRC
    src/conjgrad.cpp
    src/bregman.cpp
    src/nagopt.cpp
    src/finitediff.cpp
    src/projection.cpp
    src/gradient.cpp
    src/lipschitz.cpp
    src/mbir.cpp
    src/qggmrf.cpp
   )
#if (${USE_GPU})
#  list(APPEND SRC src/gpu/polar_grid.cu)
#endif()
include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(OpenMP)
if(OpenMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OpenMP_FOUND)

# tbb
find_package(TBB REQUIRED)
find_package(FFTW REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(TIFF REQUIRED)

# toml++
include(FetchContent)
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

# finufft
find_package(finufft REQUIRED)
add_library(tomocam SHARED ${SRC})
target_link_libraries(tomocam PUBLIC FFTW::double)
target_link_libraries(tomocam PUBLIC FFTW::float)
target_link_libraries(tomocam PUBLIC finufft::finufft)
target_link_libraries(tomocam PUBLIC TBB::tbb)
target_link_libraries(tomocam PUBLIC TIFF::TIFF)
set_property(TARGET tomocam PROPERTY CXX_STANDARD 20)

# recon executable
add_executable(recon src/recon.cpp)
target_link_libraries(recon PRIVATE tomocam tomlplusplus::tomlplusplus)


option(ENABLE_TESTS "Enable tests" OFF)
if (${ENABLE_TESTS})
    find_package(nlohmann_json REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()
