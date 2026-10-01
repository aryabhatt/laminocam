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

#ifndef TOMOCAM_GPU_SLICE_CHUNKS_H
#define TOMOCAM_GPU_SLICE_CHUNKS_H

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tomocam::gpu {

    /// A contiguous range of slices (the first array axis) that one GPU works on.
    /// The chunk owns the interior slices [begin, end) and additionally reads
    /// `h0` slices before `begin` and `h1` slices after `end` (halo). Halos are
    /// clipped at the ends of the volume, so they are 0 there, which matches the
    /// zero-padded boundary of the finite-difference kernels.
    struct SliceChunk {
        size_t begin = 0; // first interior slice
        size_t end = 0;   // one past the last interior slice
        size_t h0 = 0;    // halo slices before `begin`
        size_t h1 = 0;    // halo slices after `end`

        size_t size() const { return end - begin; }       // interior slices
        size_t first() const { return begin - h0; }       // first slice read
        size_t total() const { return h0 + size() + h1; } // slices read
    };

    /// Split `nslices` into chunks of at most `chunk_slices` interior slices, each
    /// with `halo` extra slices on either side (clipped to the volume). The
    /// interiors tile [0, nslices) exactly once and are as even as possible.
    inline std::vector<SliceChunk> make_chunks(size_t nslices, size_t chunk_slices,
                                               size_t halo) {
        if (nslices == 0) { throw std::invalid_argument("make_chunks: no slices"); }
        if (chunk_slices == 0) {
            throw std::invalid_argument("make_chunks: chunk_slices must be > 0");
        }
        const size_t nchunks = (nslices + chunk_slices - 1) / chunk_slices;
        const size_t base = nslices / nchunks;
        const size_t extra = nslices % nchunks;

        std::vector<SliceChunk> chunks;
        chunks.reserve(nchunks);
        size_t offset = 0;
        for (size_t i = 0; i < nchunks; ++i) {
            SliceChunk c;
            c.begin = offset;
            c.end = offset + base + (i < extra ? 1 : 0);
            c.h0 = std::min(halo, c.begin);
            c.h1 = std::min(halo, nslices - c.end);
            chunks.push_back(c);
            offset = c.end;
        }
        return chunks;
    }

} // namespace tomocam::gpu

#endif // TOMOCAM_GPU_SLICE_CHUNKS_H
