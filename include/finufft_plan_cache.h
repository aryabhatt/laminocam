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
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
 * Software to reproduce, distribute copies to the public, prepare derivative
 * works, and perform publicly and display publicly, and to permit other to do
 * so.
 *---------------------------------------------------------------------------------
 */

#ifndef FINUFFT_PLAN_CACHE_H
#define FINUFFT_PLAN_CACHE_H

#include <array>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "finufft_plan.h"

namespace tomocam::nufft {

    template <typename T>
    class FinufftPlanCache {
      private:
        std::unordered_map<uint64_t, FinufftPlanWrapper<T>> plan_cache_;
        std::mutex cache_mutex_;

      public:
        FinufftPlanCache() = default;
        ~FinufftPlanCache() = default;

        FinufftPlanCache(const FinufftPlanCache &) = delete;
        FinufftPlanCache &operator=(const FinufftPlanCache &) = delete;
        FinufftPlanCache(FinufftPlanCache &&) = delete;
        FinufftPlanCache &operator=(FinufftPlanCache &&) = delete;

        void clear() {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            plan_cache_.clear();
        }

        FinufftPlanWrapper<T> &get_plan(int type, int dim,
                                        std::array<int64_t, 3> n_modes, int iflag) {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            uint64_t key = genKey(type, dim, n_modes, iflag);
            auto it = plan_cache_.find(key);
            if (it != plan_cache_.end()) {
                return it->second;
            } else {
                auto [new_it, _] = plan_cache_.emplace(
                    key, FinufftPlanWrapper<T>(type, dim, n_modes, iflag));
                return new_it->second;
            }
        }

        uint64_t genKey(int type, int dim, std::array<int64_t, 3> n_modes,
                        int iflag) const {
            // type: 2 bits (1-2)
            // dim: 2 bits (1-3)
            // n_modes: 16 bits each (0-65535)
            // iflag: 1 bit (0 for +1, 1 for -1)
            // Total: 2 + 2 + 16 + 16 + 16 + 1 = 53 bits, fits in uint64_t
            uint64_t key = 0;
            key |= (static_cast<uint64_t>(type) & 0x3) << 62;
            key |= (static_cast<uint64_t>(dim) & 0x3) << 60;
            key |= (static_cast<uint64_t>(n_modes[0]) & 0xFFFF) << 44;
            key |= (static_cast<uint64_t>(n_modes[1]) & 0xFFFF) << 28;
            key |= (static_cast<uint64_t>(n_modes[2]) & 0xFFFF) << 12;
            key |= (static_cast<uint64_t>(iflag > 0 ? 1 : 0) & 0x1) << 11;
            return key;
        }
    };

    namespace plans {
        template <typename T>
        inline FinufftPlanCache<T> cache;
    }

} // namespace tomocam::nufft

#endif // FINUFFT_PLAN_CACHE_H
