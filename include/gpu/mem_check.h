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

#ifndef TOMOCAM_MEM_CHECK_H
#define TOMOCAM_MEM_CHECK_H

/// @file mem_check.h
/// Opt-in GPU memory verification checkpoints.
///
/// Enable by building with -DTOMOCAM_MEMORY_VERIFY (CMake option
/// TOMOCAM_MEMORY_VERIFY=ON).  When enabled, MEM_CHECK(tag, predicted_bytes)
/// calls cudaMemGetInfo, prints the actual device memory in use alongside the
/// analytically predicted value, and flags a warning when actual usage exceeds
/// the prediction by more than 10%.
///
/// When the macro is disabled (default), all MEM_CHECK calls compile away to
/// nothing with zero overhead.

#ifdef TOMOCAM_MEMORY_VERIFY

#include <cstdio>
#include <cuda_runtime.h>

namespace tomocam::gpu::detail {

inline void mem_check_impl(const char *tag, std::size_t predicted_bytes) {
    std::size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::size_t used = total_mem - free_mem;
    double used_mib      = static_cast<double>(used)            / (1 << 20);
    double predicted_mib = static_cast<double>(predicted_bytes) / (1 << 20);
    const char *warn = (predicted_bytes > 0 &&
                        used > predicted_bytes * 1.1) ? "  *** OVER PREDICTED ***" : "";
    std::printf("[MEM] %-48s  actual=%8.1f MiB  predicted=%8.1f MiB%s\n",
                tag, used_mib, predicted_mib, warn);
}

} // namespace tomocam::gpu::detail

/// Print actual GPU memory usage vs. predicted at a named checkpoint.
/// @param tag            Human-readable label (string literal).
/// @param predicted_bytes Expected bytes in use at this point (0 = no prediction).
#define MEM_CHECK(tag, predicted_bytes) \
    ::tomocam::gpu::detail::mem_check_impl((tag), static_cast<std::size_t>(predicted_bytes))

#else // TOMOCAM_MEMORY_VERIFY not defined

#define MEM_CHECK(tag, predicted_bytes) ((void)0)

#endif // TOMOCAM_MEMORY_VERIFY

#endif // TOMOCAM_MEM_CHECK_H
