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

#ifndef CUFFT_PLAN_CACHE_H
#define CUFFT_PLAN_CACHE_H

#include <mutex>
#include <unordered_map>

#include "cufft_plan.h"

namespace tomocam::gpu::fft {
    template <typename T>
    class cuFFTPlanCache {
      private:
        std::unordered_map<uint64_t, cuFFTPlanWrapper<T>> plan_cache_;
        std::mutex cache_mutex_;

      public:
        cuFFTPlanCache() = default;
        ~cuFFTPlanCache() = default;

        // delele copy, assignment, move, and move assignment constructors
        cuFFTPlanCache(const cuFFTPlanCache &) = delete;
        cuFFTPlanCache &operator=(const cuFFTPlanCache &) = delete;
        cuFFTPlanCache(cuFFTPlanCache &&) = delete;
        cuFFTPlanCache &operator=(cuFFTPlanCache &&) = delete;

        void clear() {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            plan_cache_.clear();
        }

        cuFFTPlanWrapper<T> &get_plan(int dim, std::array<int, 3> n_modes,
                                      cufftType type, int gpu_id) {
            std::lock_guard<std::mutex> lock(cache_mutex_);

            uint64_t key = genKey(dim, n_modes, type, gpu_id);
            auto it = plan_cache_.find(key);
            if (it != plan_cache_.end()) {
                return it->second;
            } else {
                auto [new_it, _] = plan_cache_.emplace(
                    key, cuFFTPlanWrapper<T>(dim, n_modes.data(), type, gpu_id));
                return new_it->second;
            }
        }

        uint64_t genKey(int dim, std::array<int, 3> n, cufftType type,
                        int gpu_id) const {
            // Bit layout (63 -> 0):
            //   63:60 - gpu_id  (4 bits, 0-15)
            //   59:58 - dim     (2 bits, 1-3)
            //   57:42 - n[0] (16 bits, 0-65535)
            //   41:26 - n[1] (16 bits, 0-65535)
            //   25:10 - n[2] (16 bits, 0-65535)
            //    9:2  - type    (8 bits, covers all cufftType values up to 0x6c)
            uint64_t key = 0;
            key |= (static_cast<uint64_t>(gpu_id) & 0xF) << 60;
            key |= (static_cast<uint64_t>(dim) & 0x3) << 58;
            key |= (static_cast<uint64_t>(n[0]) & 0xFFFF) << 42;
            key |= (static_cast<uint64_t>(n[1]) & 0xFFFF) << 26;
            key |= (static_cast<uint64_t>(n[2]) & 0xFFFF) << 10;
            key |= (static_cast<uint64_t>(type) & 0xFF) << 2;
            return key;
        }
    };

    namespace cache {
        template <typename T>
        inline cuFFTPlanCache<T> plans;
    }
} // namespace tomocam::gpu::fft

#endif // CUFFT_PLAN_CACHE_H
