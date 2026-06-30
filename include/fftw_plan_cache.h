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

#ifndef FFTW_PLAN_CACHE_H
#define FFTW_PLAN_CACHE_H

#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "fftw_plan.h"

namespace tomocam::fft {

    template <typename T>
    class FftwPlanCache {
      private:
        std::unordered_map<uint64_t, FftwPlanWrapper<T>> plan_cache_;
        std::mutex cache_mutex_;

      public:
        FftwPlanCache() = default;
        ~FftwPlanCache() = default;

        FftwPlanCache(const FftwPlanCache &) = delete;
        FftwPlanCache &operator=(const FftwPlanCache &) = delete;
        FftwPlanCache(FftwPlanCache &&) = delete;
        FftwPlanCache &operator=(FftwPlanCache &&) = delete;

        void clear() {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            plan_cache_.clear();
        }

        FftwPlanWrapper<T> &get_plan(int ndim, std::array<int, 3> n, int howmany,
                                     FftwType type) {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            uint64_t key = genKey(ndim, n, howmany, type);
            auto it = plan_cache_.find(key);
            if (it != plan_cache_.end()) {
                return it->second;
            } else {
                auto [new_it, _] = plan_cache_.emplace(
                    key, FftwPlanWrapper<T>(ndim, n, howmany, type));
                return new_it->second;
            }
        }

        uint64_t genKey(int ndim, std::array<int, 3> n, int howmany,
                        FftwType type) const {
            // Bit layout (63 -> 0):
            //   63:62 - ndim    (2 bits, 1-3)
            //   61:46 - n[0]    (16 bits, 0-65535)
            //   45:30 - n[1]    (16 bits, 0-65535)
            //   29:14 - n[2]    (16 bits, 0-65535)
            //   13:2  - howmany (12 bits, 0-4095)
            //    1:0  - type    (2 bits, FftwType 0-3)
            // Total: 2 + 16 + 16 + 16 + 12 + 2 = 64 bits
            uint64_t key = 0;
            key |= (static_cast<uint64_t>(ndim) & 0x3) << 62;
            key |= (static_cast<uint64_t>(n[0]) & 0xFFFF) << 46;
            key |= (static_cast<uint64_t>(n[1]) & 0xFFFF) << 30;
            key |= (static_cast<uint64_t>(n[2]) & 0xFFFF) << 14;
            key |= (static_cast<uint64_t>(howmany) & 0xFFF) << 2;
            key |= (static_cast<uint64_t>(type) & 0x3);
            return key;
        }
    };

    namespace plans {
        template <typename T>
        inline FftwPlanCache<T> cache;
    }

} // namespace tomocam::fft

#endif // FFTW_PLAN_CACHE_H
