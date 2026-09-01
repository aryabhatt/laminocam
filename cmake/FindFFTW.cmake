
# Find FFTW library
#
# This module defines:
#   FFTW_FOUND
#   FFTW_INCLUDE_DIR
#   FFTW_LIBRARIES
#   FFTWF_LIBRARIES
#
# Targets:
#   FFTW::double   (double precision, links fftw3)
#   FFTW::float    (single precision, links fftw3f)
#
# Set FFTW_DIR to the directory containing FFTW3Config.cmake, e.g.:
#   cmake -DFFTW_DIR=~/fftw/lib/cmake/fftw3 ..
#
# For module-mode fallback set FFTW_ROOT or FFTW_DIR to the installation
# prefix (NERSC sets $FFTW_ROOT via environment modules).

include(FindPackageHandleStandardArgs)

# --- Config mode (preferred when available) ---
# Recent FFTW3 releases ship FFTW3Config.cmake; try the user-supplied dir first.
find_package(FFTW3 CONFIG QUIET
    PATHS
        "${FFTW_DIR}"
    NO_DEFAULT_PATH
)
if (NOT FFTW3_FOUND)
    find_package(FFTW3 CONFIG QUIET)
endif()

if (FFTW3_FOUND)
    # Alias to our conventional target names if the config didn't create them.
    if (TARGET FFTW3::fftw3 AND NOT TARGET FFTW::double)
        add_library(FFTW::double ALIAS FFTW3::fftw3)
    endif()
    if (TARGET FFTW3::fftw3f AND NOT TARGET FFTW::float)
        add_library(FFTW::float ALIAS FFTW3::fftw3f)
    endif()
    message(STATUS "Found FFTW3 (config): ${FFTW3_DIR}")
    return()
endif()

# --- Module mode fallback ---
# FFTW_ROOT (CMake 3.12+) is honoured automatically by find_path/find_library.
# $ENV{FFTW_DIR} is the conventional HPC environment variable for the install prefix.
set(FFTW_SEARCH_PATHS
    /usr/local
    /opt/homebrew
    /opt/local
    /usr
    /opt
)

find_path(FFTW_INCLUDE_DIR
    NAMES
        fftw3.h
    HINTS
        $ENV{FFTW_ROOT}
        $ENV{FFTW_DIR}
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES
        include
    PATHS
        ${FFTW_SEARCH_PATHS}
)

find_library(FFTW_LIBRARY
    NAMES
        fftw3
    HINTS
        $ENV{FFTW_ROOT}
        $ENV{FFTW_DIR}
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES
        lib64
        lib
    PATHS
        ${FFTW_SEARCH_PATHS}
)

find_library(FFTWF_LIBRARY
    NAMES
        fftw3f
    HINTS
        $ENV{FFTW_ROOT}
        $ENV{FFTW_DIR}
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES
        lib64
        lib
    PATHS
        ${FFTW_SEARCH_PATHS}
)

find_package_handle_standard_args(FFTW
    REQUIRED_VARS
        FFTW_INCLUDE_DIR
        FFTW_LIBRARY
        FFTWF_LIBRARY
)

if (FFTW_FOUND)
    set(FFTW_LIBRARIES  ${FFTW_LIBRARY})
    set(FFTWF_LIBRARIES ${FFTWF_LIBRARY})

    mark_as_advanced(
        FFTW_INCLUDE_DIR
        FFTW_LIBRARY
        FFTWF_LIBRARY
    )

    if (NOT TARGET FFTW::double)
        add_library(FFTW::double INTERFACE IMPORTED GLOBAL)
        target_include_directories(FFTW::double INTERFACE ${FFTW_INCLUDE_DIR})
        target_link_libraries(FFTW::double INTERFACE ${FFTW_LIBRARY})
    endif()
    message(STATUS "Found FFTW::double (module): ${FFTW_LIBRARY}")

    if (NOT TARGET FFTW::float)
        add_library(FFTW::float INTERFACE IMPORTED GLOBAL)
        target_include_directories(FFTW::float INTERFACE ${FFTW_INCLUDE_DIR})
        target_link_libraries(FFTW::float INTERFACE ${FFTWF_LIBRARY})
    endif()
    message(STATUS "Found FFTW::float (module): ${FFTWF_LIBRARY}")
else()
    message(FATAL_ERROR
        "FFTW not found.\n"
        "  Config mode:  set FFTW_DIR to the directory containing FFTW3Config.cmake\n"
        "                e.g. -DFFTW_DIR=~/fftw/lib/cmake/fftw3\n"
        "  Module mode:  set FFTW_ROOT to the installation prefix\n"
        "                e.g. -DFFTW_ROOT=~/fftw\n"
        "                or export FFTW_DIR=~/fftw (environment variable)"
    )
endif()
