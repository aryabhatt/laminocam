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

#ifndef TOMOCAM_GPU_PRECOND_H
#define TOMOCAM_GPU_PRECOND_H

#include "dtypes.h"
#include "gpu/device_array.h"
#include "gpu/polar_grid.h"

namespace tomocam::gpu::opt {

    // -------------------------------------------------------------------------
    // Base interface: z = P * r  (out-of-place, r unchanged)
    // -------------------------------------------------------------------------
    template <typename T>
    class IPrecond {
      public:
        virtual ~IPrecond() = default;
        virtual void apply(const DeviceArray<T> &r, DeviceArray<T> &z) const = 0;
    };

    template <typename T>
    class IdentityPrecond : public IPrecond<T> {
      public:
        void apply(const DeviceArray<T> &r, DeviceArray<T> &z) const override {
            z = r.clone();
        }
    };

    // -------------------------------------------------------------------------
    // Density-compensation (Jacobi) preconditioner — GPU version.
    //   w_[i] ~ diag(A^T A)_i, built via Pipe-Menon iterations on GPU.
    //   apply: z[i] = r[i] / w_[i]  — O(N), no NUFFT in hot path.
    // -------------------------------------------------------------------------
    template <typename T>
    class DensityComp : public IPrecond<T> {
      public:
        DensityComp(const gpu::PolarGrid<T> &pg, dims_t recon_dims,
                    size_t n_iters = 10);
        void apply(const DeviceArray<T> &r, DeviceArray<T> &z) const override;

      private:
        DeviceArray<T> w_;
    };

} // namespace tomocam::gpu::opt

#endif // TOMOCAM_GPU_PRECOND_H
