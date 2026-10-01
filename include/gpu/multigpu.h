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

#ifndef TOMOCAM_GPU_MULTIGPU_H
#define TOMOCAM_GPU_MULTIGPU_H

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "array.h"
#include "gpu/device_array.h"
#include "gpu/memory.h"
#include "gpu/polar_grid2d.h"
#include "gpu/slice_chunks.h"
#include "gpu/toeplitz2d.h"
#include "gpu/utils.h"

namespace tomocam::gpu {

    /// Number of CUDA devices to use: `requested` (0 = all), capped at the number
    /// of devices present.
    inline int resolve_num_gpus(size_t requested) {
        int n = 0;
        SAFE_CALL(cudaGetDeviceCount(&n));
        if (n < 1) { throw std::runtime_error("no CUDA device available"); }
        return (requested == 0) ? n : std::min(n, static_cast<int>(requested));
    }

    /// Copy the slices read by chunk `c` (interior + halo) of a host array to a
    /// new device array of dims {c.total(), n2, n3} on the current device.
    template <typename T>
    DeviceArray<T> upload_chunk(const Array<T> &host, const SliceChunk &c) {
        const dims_t d = host.dims();
        const size_t plane = d.n2 * d.n3;
        DeviceArray<T> dev(dims_t{c.total(), d.n2, d.n3});
        copyH2D(dev.data(), host.begin() + c.first() * plane, dev.bytes());
        return dev;
    }

    /// Copy only the interior slices of chunk `c` from a device array holding
    /// c.total() slices (interior + halo) back into the host array. Interiors of
    /// different chunks are disjoint, so concurrent calls need no locking.
    template <typename T>
    void download_interior(const DeviceArray<T> &dev, Array<T> &host,
                           const SliceChunk &c) {
        const dims_t d = host.dims();
        const size_t plane = d.n2 * d.n3;
        copyD2H(host.begin() + c.begin * plane, dev.data() + c.h0 * plane,
                c.size() * plane * sizeof(T));
    }

    /// Copy the interior slices of a halo-carrying device array into a new device
    /// array of c.size() slices.
    template <typename T>
    DeviceArray<T> crop_interior(const DeviceArray<T> &dev, const SliceChunk &c) {
        const dims_t d = dev.dims();
        const size_t plane = d.n2 * d.n3;
        DeviceArray<T> out(dims_t{c.size(), d.n2, d.n3});
        copyD2D(out.data(), dev.data() + c.h0 * plane, out.bytes());
        return out;
    }

    /// Per-device state that is built once and reused for every chunk.
    template <typename T>
    struct DeviceContext {
        int id;
        PolarGrid2D<T> pg;
        PointSpreadFunction2D<T> psf;

        // pg and psf allocate device memory, so the device must be current
        // while they are constructed.
        DeviceContext(int gpu, const std::vector<T> &theta, size_t ncols,
                      size_t image_size)
            : id(gpu), pg(make_grid(gpu, theta, ncols)),
              psf(make_psf(gpu, pg, image_size)) {
            // fix the PSF slab size once instead of querying free memory on
            // every convolve() call; the rest of the budget is for the chunks
            DeviceGuard guard(gpu);
            size_t free_b = 0, total_b = 0;
            SAFE_CALL(cudaMemGetInfo(&free_b, &total_b));
            psf.set_memory_budget(static_cast<size_t>(0.3 * free_b));
        }

      private:
        static PolarGrid2D<T> make_grid(int gpu, const std::vector<T> &theta,
                                        size_t ncols) {
            DeviceGuard guard(gpu);
            return PolarGrid2D<T>(theta, ncols);
        }
        static PointSpreadFunction2D<T> make_psf(int gpu, const PolarGrid2D<T> &pg,
                                                 size_t image_size) {
            DeviceGuard guard(gpu);
            return PointSpreadFunction2D<T>(pg, image_size);
        }
    };

    /// Runs one streamed pass over a set of slice chunks on several GPUs.
    ///
    /// One worker thread per GPU pulls the next chunk index from a shared atomic
    /// counter (dynamic load balancing), makes its device current and calls the
    /// user step. Exactly one worker uses each device, which matters because the
    /// cuFFT / cuFINUFFT plan caches hand out shared mutable plans.
    template <typename T>
    class ChunkExecutor {
      private:
        std::vector<std::unique_ptr<DeviceContext<T>>> ctx_;

      public:
        /// @param theta       projection angles
        /// @param ncols       detector columns (radial samples)
        /// @param image_size  square in-plane size of the reconstructed slices
        /// @param ngpus       number of devices to use
        ChunkExecutor(const std::vector<T> &theta, size_t ncols, size_t image_size,
                      int ngpus) {
            for (int g = 0; g < ngpus; ++g) {
                ctx_.push_back(
                    std::make_unique<DeviceContext<T>>(g, theta, ncols, image_size));
            }
        }

        int ngpus() const { return static_cast<int>(ctx_.size()); }

        /// Calls step(DeviceContext<T>&, chunk_index, const SliceChunk&) for every
        /// chunk, spread across the GPUs. Returns when all chunks are done. The
        /// first exception thrown by any step is rethrown here after every worker
        /// has stopped.
        template <typename Fn>
        void run(const std::vector<SliceChunk> &chunks, Fn &&step) {
            std::atomic<size_t> next{0};
            std::atomic<bool> failed{false};
            std::exception_ptr error;
            std::mutex error_mutex;

            auto worker = [&](DeviceContext<T> &ctx) {
                try {
                    DeviceGuard guard(ctx.id);
                    while (!failed) {
                        size_t i = next.fetch_add(1);
                        if (i >= chunks.size()) break;
                        step(ctx, i, chunks[i]);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error) { error = std::current_exception(); }
                    failed = true;
                }
            };

            std::vector<std::thread> threads;
            for (size_t g = 1; g < ctx_.size(); ++g) {
                threads.emplace_back(worker, std::ref(*ctx_[g]));
            }
            worker(*ctx_[0]); // the calling thread drives device 0
            for (auto &t : threads) { t.join(); }
            if (error) { std::rethrow_exception(error); }
        }
    };

} // namespace tomocam::gpu

#endif // TOMOCAM_GPU_MULTIGPU_H
