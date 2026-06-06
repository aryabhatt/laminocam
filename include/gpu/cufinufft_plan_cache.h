
#ifndef CUFINUFFT_PLAN_CACHE_H
#define CUFINUFFT_PLAN_CACHE_H

#include <array>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "gpu/cufinufft_plan.h"

namespace tomocam::gpu::nufft {

    template <typename T>
    class cuFinufftPlanCache {
      private:
        std::unordered_map<uint64_t, cuFinfftPlanWrapper<T>> plan_cache_;
        std::mutex cache_mutex_;

      public:
        cuFinufftPlanCache() = default;
        ~cuFinufftPlanCache() = default;

        // delele copy, assignment, move, and move assignment constructors
        cuFinufftPlanCache(const cuFinufftPlanCache &) = delete;
        cuFinufftPlanCache &operator=(const cuFinufftPlanCache &) = delete;
        cuFinufftPlanCache(cuFinufftPlanCache &&) = delete;
        cuFinufftPlanCache &operator=(cuFinufftPlanCache &&) = delete;

        // Destroy all cached plans while the CUDA context is still live.
        // Call before cudaDeviceReset() to avoid cudaErrorCudartUnloading on exit.
        void clear() {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            plan_cache_.clear();
        }

        cuFinfftPlanWrapper<T> &get_plan(int type, int dim,
                                         std::array<int64_t, 3> n_modes, int iflag,
                                         int gpu_id) {

            std::lock_guard<std::mutex> lock(cache_mutex_);
            uint64_t key = genKey(type, dim, n_modes, iflag, gpu_id);
            auto it = plan_cache_.find(key);
            if (it != plan_cache_.end()) {
                return it->second;
            } else {
                auto [new_it, _] = plan_cache_.emplace(
                    key, cuFinfftPlanWrapper<T>(type, dim, n_modes, iflag, gpu_id));
                return new_it->second;
            }
        }

        uint64_t genKey(int type, int dim, std::array<int64_t, 3> n_modes, int iflag,
                        int gpu_id) const {
            // gpu_id: 2 bits (0-3)
            // type: 2 bits (1-2)
            // dim: 2 bits (1-3)
            // n_modes: 16 bits each (0-65535)
            // iflag: 1 bit (0 for +1, 1 for -1)
            // Total: 2 + 2 + 2 + 16 + 16 + 16 + 1 = 55 bits, fits in uint64_t
            uint64_t key = 0;
            key |= (static_cast<uint64_t>(gpu_id) & 0x3) << 62; // 2 bits for gpu_id
            key |= (static_cast<uint64_t>(type) & 0x3) << 60;   // 2 bits for type
            key |= (static_cast<uint64_t>(dim) & 0x3) << 58;    // 2 bits for dim
            key |= (static_cast<uint64_t>(n_modes[0]) & 0xFFFF)
                   << 42; // 16 bits for n_modes[0]
            key |= (static_cast<uint64_t>(n_modes[1]) & 0xFFFF)
                   << 26; // 16 bits for n_modes[1]
            key |= (static_cast<uint64_t>(n_modes[2]) & 0xFFFF)
                   << 10; // 16 bits for n_modes[2]
            key |= (static_cast<uint64_t>(iflag > 0 ? 1 : 0) & 0x1)
                   << 9; // 1 bit for iflag
            return key;
        }
    };

    namespace plans {
        template <typename T>
        inline cuFinufftPlanCache<T> cache;
    }

} // namespace tomocam::gpu::nufft

#endif // CUFINUFFT_PLAN_CACHE_H
