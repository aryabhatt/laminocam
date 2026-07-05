
cmake_minimum_required(VERSION 3.20)

project(Testbed CXX)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Add cuda
option(USE_GPU "Use GPU Acceleration" TRUE)
if (${USE_GPU})
  add_definitions(-DUSE_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
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
    src/finitediff.cpp
    src/projection.cpp
    src/gradient.cpp
    src/mbir.cpp
   )
if (${USE_GPU})
  list(APPEND SRC
    src/gpu/array_ops.cu
    src/gpu/device_array.cu
    src/gpu/bregman.cu
    src/gpu/conjgrad.cu
    src/gpu/gradient.cu
    src/gpu/mbir.cu
    src/gpu/projection.cu
    src/gpu/fftshift.cu
    src/gpu/finitediff.cu
    src/gpu/padding.cu
    src/gpu/polar_grid.cu
  )
endif()
include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(OpenMP)
if(OpenMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OpenMP_FOUND)

# tbb
find_package(TBB REQUIRED)
find_package(FFTW REQUIRED)
find_package(TIFF REQUIRED)
find_package(HDF5 REQUIRED)
include_directories(${HDF5_C_INCLUDE_DIR})

# toml++
include(FetchContent)
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)
include_directories(${tomlplusplus_SOURCE_DIR}/include)

# finufft
find_package(finufft REQUIRED)
add_library(tomocam_scalar SHARED ${SRC})
target_link_libraries(tomocam_scalar PUBLIC FFTW::double)
target_link_libraries(tomocam_scalar PUBLIC FFTW::float)
target_link_libraries(tomocam_scalar PUBLIC finufft::finufft)
target_link_libraries(tomocam_scalar PUBLIC TBB::tbb)
if (${USE_GPU})
  target_link_libraries(tomocam_scalar PUBLIC finufft::cufinufft)
  target_link_libraries(tomocam_scalar PUBLIC CUDA::cufft)
  target_include_directories(tomocam_scalar PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(tomocam_scalar PUBLIC CUDA::cudart)
endif()
set_property(TARGET tomocam_scalar PROPERTY CXX_STANDARD 20)

# recon executable
add_executable(recon_scalar src/recon.cpp)
target_link_libraries(recon_scalar PRIVATE tomocam_scalar tomlplusplus::tomlplusplus)
target_link_libraries(recon_scalar PRIVATE TIFF::TIFF ${HDF5_LIBRARIES})

install(TARGETS recon_scalar RUNTIME DESTINATION bin)
install(TARGETS tomocam_scalar LIBRARY DESTINATION lib)

option(ENABLE_TESTS "Enable tests" OFF)

# GPU memory verification checkpoints (guarded by #ifdef TOMOCAM_MEMORY_VERIFY)
option(TOMOCAM_MEMORY_VERIFY "Enable GPU memory verification checkpoints in bregman/cgsolver" OFF)
if (TOMOCAM_MEMORY_VERIFY)
    target_compile_definitions(tomocam_scalar PRIVATE TOMOCAM_MEMORY_VERIFY)
    message(STATUS "TOMOCAM_MEMORY_VERIFY: GPU memory checkpoints enabled")
endif()

if (${ENABLE_TESTS})
    enable_testing()
    add_subdirectory(tests)
endif()
