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

// Host-only checks of make_chunks(): the interiors tile [0, N) exactly once,
// halos are clipped at the ends of the volume, and sizes are balanced.

#include <format>
#include <iostream>
#include <stdexcept>

#include "gpu/slice_chunks.h"

using tomocam::gpu::make_chunks;
using tomocam::gpu::SliceChunk;

static bool check(bool cond, const char *what) {
    if (!cond) std::cout << std::format("  FAIL  {}\n", what);
    return cond;
}

static bool test_layout(size_t n, size_t chunk, size_t halo) {
    auto chunks = make_chunks(n, chunk, halo);
    const std::string tag = std::format("N={} chunk={} halo={}", n, chunk, halo);
    bool ok = true;

    size_t expect_begin = 0, smallest = n, largest = 0;
    for (const SliceChunk &c : chunks) {
        ok &= check(c.begin == expect_begin, (tag + ": contiguous").c_str());
        ok &= check(c.end > c.begin, (tag + ": non-empty").c_str());
        ok &= check(c.size() <= chunk, (tag + ": respects chunk size").c_str());
        ok &= check(c.h0 == std::min(halo, c.begin), (tag + ": left halo").c_str());
        ok &=
            check(c.h1 == std::min(halo, n - c.end), (tag + ": right halo").c_str());
        ok &= check(c.first() + c.total() <= n,
                    (tag + ": reads inside volume").c_str());
        expect_begin = c.end;
        smallest = std::min(smallest, c.size());
        largest = std::max(largest, c.size());
    }
    ok &= check(expect_begin == n, (tag + ": covers all slices").c_str());
    ok &= check(largest - smallest <= 1, (tag + ": balanced").c_str());
    ok &= check(chunks.front().h0 == 0 && chunks.back().h1 == 0,
                (tag + ": no halo at volume ends").c_str());
    std::cout << std::format("  {}  {} -> {} chunks\n", ok ? "PASS" : "FAIL", tag,
                             chunks.size());
    return ok;
}

int main() {
    std::cout << "=== make_chunks tests ===\n";
    int failed = 0;
    for (size_t n : {1, 2, 7, 64, 745}) {
        for (size_t chunk : {1, 2, 3, 10, 1000}) {
            for (size_t halo : {0, 1, 2, 5}) {
                failed += !test_layout(n, chunk, halo);
            }
        }
    }

    bool threw = false;
    try {
        make_chunks(5, 0, 0);
    } catch (const std::invalid_argument &) { threw = true; }
    failed += !check(threw, "chunk_slices == 0 is rejected");

    std::cout << std::format("\nResults: {} failed\n", failed);
    return failed == 0 ? 0 : 1;
}
