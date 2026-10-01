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

// Multi-GPU 2D tomography. The solver loops (CG, Split-Bregman) run on the host
// and the volume vectors live in host memory. Every operation on the volume is
// one streamed pass: the slices are cut into chunks, GPU workers pull chunks
// from a shared queue, apply the per-chunk step and write the interior slices
// back. Device memory is therefore bounded by the chunk size, not the volume.
//
// Halo requirements along the slice axis (the finite-difference kernels use
// central differences, zero-padded at the ends of the volume):
//   A (Toeplitz PSF), dot products, axpys ......... 0 slices (slices independent)
//   grad, neg_divergence .......................... 1 slice
//   neg_laplacian = divergence(grad(.)) ........... 2 slices
// With these halos every pass is exact, not an approximation near chunk seams.

#include <algorithm>
#include <array>
#include <cmath>
#include <execution>
#include <format>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

#include "array.h"
#include "gpu/bregman.h"
#include "gpu/device_array.h"
#include "gpu/device_array_ops.h"
#include "gpu/finitediff.h"
#include "gpu/multigpu.h"
#include "gpu/projection.h"
#include "gpu/slice_chunks.h"
#include "gpu/tomocam.h"
#include "recon_params.h"

namespace tomocam::gpu {

    namespace {

        constexpr double EPS = 1e-8;
        // slice-axis reach of neg_laplacian (grad followed by divergence)
        constexpr size_t TV_HALO = 2;

        // ---------------------------------------------------------------------
        // host-side vector ops (volume vectors live in host memory)
        // ---------------------------------------------------------------------
        // Deterministic parallel reduction. The array is cut into fixed-size
        // blocks, the blocks are processed in parallel, and the per-block partial
        // sums are added in block order, so the result never depends on thread
        // scheduling. body(lo, hi, acc) accumulates elements [lo, hi) into acc.
        constexpr size_t REDUCE_BLOCK = size_t(1) << 16;

        template <size_t K, typename Body>
        std::array<double, K> block_reduce(size_t n, Body body) {
            const size_t nblocks = (n + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
            std::vector<std::array<double, K>> part(nblocks);
            std::vector<size_t> ids(nblocks);
            std::iota(ids.begin(), ids.end(), size_t(0));
            std::for_each(std::execution::par, ids.begin(), ids.end(),
                          [&](size_t blk) {
                              const size_t lo = blk * REDUCE_BLOCK;
                              body(lo, std::min(n, lo + REDUCE_BLOCK), part[blk]);
                          });
            std::array<double, K> total{};
            for (const auto &p : part) {
                for (size_t k = 0; k < K; ++k) total[k] += p[k];
            }
            return total;
        }

        template <typename T>
        double dot_h(const Array<T> &a, const Array<T> &b) {
            const T *pa = a.begin();
            const T *pb = b.begin();
            return block_reduce<1>(a.size(), [&](size_t lo, size_t hi, auto &acc) {
                double s = 0;
                for (size_t i = lo; i < hi; ++i) s += double(pa[i]) * double(pb[i]);
                acc[0] = s;
            })[0];
        }

        template <typename T>
        double norm_h(const Array<T> &a) {
            return std::sqrt(dot_h(a, a));
        }

        template <typename T>
        void copy_h(const Array<T> &src, Array<T> &dst) {
            std::copy(std::execution::par_unseq, src.begin(), src.end(),
                      dst.begin());
        }

        // one CG step on the host, in a single pass over the vectors:
        //   x += alpha * p;  r -= alpha * Ap
        // returns |r|^2 (new), |p|^2 (before p is updated) and |x|^2 (updated)
        struct CgSums {
            double rr, pp, xx;
        };
        template <typename T>
        CgSums cg_update(Array<T> &x, Array<T> &r, const Array<T> &p,
                         const Array<T> &Ap, double alpha) {
            T *px = x.begin();
            T *pr = r.begin();
            const T *pp = p.begin();
            const T *pap = Ap.begin();
            auto s = block_reduce<3>(x.size(), [&](size_t lo, size_t hi, auto &acc) {
                double rr = 0, pn = 0, xx = 0;
                for (size_t i = lo; i < hi; ++i) {
                    const T xv = T(px[i] + alpha * pp[i]);
                    const T rv = T(pr[i] - alpha * pap[i]);
                    px[i] = xv;
                    pr[i] = rv;
                    rr += double(rv) * double(rv);
                    pn += double(pp[i]) * double(pp[i]);
                    xx += double(xv) * double(xv);
                }
                acc = {rr, pn, xx};
            });
            return {s[0], s[1], s[2]};
        }

        // p = r + beta * p
        template <typename T>
        void axpy_h(Array<T> &p, double beta, const Array<T> &r) {
            std::transform(std::execution::par_unseq, p.begin(), p.end(), r.begin(),
                           p.begin(),
                           [beta](T pv, T rv) { return T(rv + beta * pv); });
        }

        // sets x_old = x and returns {|x - x_old|^2, |x_old|^2} (old x_old)
        template <typename T>
        std::array<double, 2> diff_and_copy(const Array<T> &x, Array<T> &x_old) {
            const T *px = x.begin();
            T *po = x_old.begin();
            return block_reduce<2>(x.size(), [&](size_t lo, size_t hi, auto &acc) {
                double dd = 0, oo = 0;
                for (size_t i = lo; i < hi; ++i) {
                    const double d = double(px[i]) - double(po[i]);
                    dd += d * d;
                    oo += double(po[i]) * double(po[i]);
                    po[i] = px[i];
                }
                acc = {dd, oo};
            });
        }

        // CG vectors, allocated once and reused by every solve
        template <typename T>
        struct CgWorkspace {
            Array<T> r, p, Ap;
            explicit CgWorkspace(dims_t d) : r(d), p(d), Ap(d) {}
        };

        // sum of the partials in chunk order, so the result does not depend on
        // which GPU happened to process which chunk
        double ordered_sum(const std::vector<double> &partial) {
            return std::accumulate(partial.begin(), partial.end(), 0.0);
        }

        // <a, b> over the interior slices of chunk c of two halo-carrying arrays
        template <typename T>
        double interior_dot(const DeviceArray<T> &a, const DeviceArray<T> &b,
                            const SliceChunk &c) {
            const size_t plane = a.nrows() * a.ncols();
            const size_t off = c.h0 * plane;
            const size_t n = c.size() * plane;
            thrust::device_ptr<const T> pa(a.data() + off);
            thrust::device_ptr<const T> pb(b.data() + off);
            return thrust::inner_product(pa, pa + n, pb, 0.0);
        }

        // ---------------------------------------------------------------------
        // streamed operators
        // ---------------------------------------------------------------------

        // out = A in                      (tv == false, chunks need halo 0)
        // out = (A + mu * (-laplacian)) in (tv == true,  chunks need halo >= 2)
        // Returns <in, out> over the whole volume.
        template <typename T>
        class StreamedOperator {
          private:
            ChunkExecutor<T> &exec_;
            const std::vector<SliceChunk> &chunks_;
            bool tv_;
            T mu_;

          public:
            StreamedOperator(ChunkExecutor<T> &exec,
                             const std::vector<SliceChunk> &chunks, bool tv, T mu)
                : exec_(exec), chunks_(chunks), tv_(tv), mu_(mu) {}

            double operator()(const Array<T> &in, Array<T> &out) const {
                std::vector<double> partial(chunks_.size(), 0.0);
                exec_.run(chunks_,
                          [&](DeviceContext<T> &ctx, size_t i, const SliceChunk &c) {
                              DeviceArray<T> d_in = upload_chunk(in, c);
                              DeviceArray<T> d_out = sysmat2d(d_in, ctx.psf);
                              if (tv_) {
                                  auto lap = opt::neg_laplacian(d_in);
                                  array::xpay(d_out, lap, mu_);
                              }
                              partial[i] = interior_dot(d_in, d_out, c);
                              download_interior(d_out, out, c);
                          });
                return ordered_sum(partial);
            }
        };

        // ---------------------------------------------------------------------
        // streamed conjugate gradient: solves A x = y, x is the initial guess on
        // entry and the solution on exit. Same algorithm and stopping rules as
        // gpu::opt::cgsolver (identity preconditioner).
        // ---------------------------------------------------------------------
        template <typename T, typename Op>
        void streamed_cg(const Op &A, const Array<T> &y, Array<T> &x,
                         CgWorkspace<T> &ws, bool zero_start, size_t max_iter, T tol,
                         T xtol) {

            Array<T> &r = ws.r;
            Array<T> &p = ws.p;
            Array<T> &Ap = ws.Ap;

            // r = y - A x  (no operator pass needed when starting from zero)
            copy_h(y, r);
            if (!zero_start) {
                A(x, Ap);
                r -= Ap;
            }
            const double y_norm = norm_h(y);

            copy_h(r, p);
            double rs_old = dot_h(r, r);
            if (rs_old == 0) return;

            for (size_t iter = 0; iter < max_iter; ++iter) {
                const double pAp = A(p, Ap);
                const double alpha = rs_old / pAp;
                const auto sums = cg_update(x, r, p, Ap, alpha);

                // relative step size, from p before it is updated
                const double dx = std::abs(alpha) * std::sqrt(sums.pp) /
                                  (std::sqrt(sums.xx) + EPS);

                const double beta = sums.rr / rs_old;
                axpy_h(p, beta, r); // p = r + beta * p
                rs_old = sums.rr;

                const double res = std::sqrt(sums.rr) / (y_norm + EPS);
                std::cout << std::format(
                    "\t CG iter {:3d}: residual = {:.6e}, dx = {:.6e}\n", iter + 1,
                    res, dx);
                if (res < tol) break;
                if (dx < xtol) {
                    std::cout << "CG converged based on solution change\n";
                    break;
                }
            }
        }

        // ---------------------------------------------------------------------
        // streamed Split-Bregman (same algorithm as gpu::opt::split_bregman)
        // `chunks` must carry a halo of at least TV_HALO slices.
        // ---------------------------------------------------------------------
        template <typename T>
        Array<T> streamed_split_bregman(ChunkExecutor<T> &exec,
                                        const std::vector<SliceChunk> &chunks,
                                        const Array<T> &yT, T lambda, T mu,
                                        size_t outer_max, size_t inner_max, T tol,
                                        T xtol) {
            const dims_t dims = yT.dims();
            Array<T> x = Array<T>::zeros(dims);
            Array<T> x_old = x.clone();
            Array<T> rhs(dims);
            std::array<Array<T>, 3> b, d;
            for (size_t k = 0; k < 3; ++k) {
                b[k] = Array<T>::zeros(dims);
                d[k] = Array<T>::zeros(dims);
            }

            StreamedOperator<T> Ap(exec, chunks, true, mu);
            CgWorkspace<T> ws(dims);

            for (size_t iter = 0; iter < outer_max; ++iter) {

                // rhs = A^T y + mu * (-div)(d - b). Reads d, b with halo and writes
                // only rhs.
                exec.run(chunks,
                         [&](DeviceContext<T> &, size_t, const SliceChunk &c) {
                             std::array<DeviceArray<T>, 3> d_b;
                             for (size_t k = 0; k < 3; ++k) {
                                 d_b[k] = upload_chunk(d[k], c);
                                 d_b[k] -= upload_chunk(b[k], c);
                             }
                             auto neg_div = opt::neg_divergence(d_b);
                             DeviceArray<T> r = upload_chunk(yT, c);
                             array::xpay(r, neg_div, mu);
                             download_interior(r, rhs, c);
                         });

                // x-update: (A^T A + mu * grad^T grad) x = rhs
                streamed_cg<T>(Ap, rhs, x, ws, iter == 0, inner_max, tol, xtol);

                // isotropic TV shrinkage and Bregman update. x is read with halo;
                // d and b are elementwise, so only their interiors are touched.
                exec.run(chunks,
                         [&](DeviceContext<T> &, size_t, const SliceChunk &c) {
                             const SliceChunk in{c.begin, c.end, 0, 0};
                             auto dx_full = opt::grad(upload_chunk(x, c));

                             std::array<DeviceArray<T>, 3> dx, bk;
                             for (size_t k = 0; k < 3; ++k) {
                                 dx[k] = crop_interior(dx_full[k], c);
                                 bk[k] = upload_chunk(b[k], in);
                             }
                             auto sk = opt::compute_sk(dx, bk);
                             for (size_t k = 0; k < 3; ++k) {
                                 auto dk = opt::shrink(dx[k], bk[k], sk, lambda, mu);
                                 bk[k] += dx[k] - dk; // b += dx - d
                                 download_interior(dk, d[k], in);
                                 download_interior(bk[k], b[k], in);
                             }
                         });

                // convergence: ||x_k+1 - x_k|| / ||x_k||
                const auto nd = diff_and_copy(x, x_old); // also sets x_old = x
                const double norm_diff = std::sqrt(nd[0]) / (std::sqrt(nd[1]) + EPS);
                std::cout << std::format(
                    "Outer iter: {}, ‖xᵏ⁺¹ − xᵏ‖₂ / ‖xᵏ‖₂: {:.6e}\n", iter,
                    norm_diff);
                if (norm_diff < xtol) { break; }
            }
            return x;
        }

        // free memory of the most constrained device among the first ngpus
        size_t min_free_bytes(int ngpus) {
            size_t best = std::numeric_limits<size_t>::max();
            for (int g = 0; g < ngpus; ++g) {
                DeviceGuard guard(g);
                size_t free_b = 0, total_b = 0;
                SAFE_CALL(cudaMemGetInfo(&free_b, &total_b));
                best = std::min(best, free_b);
            }
            return best;
        }

    } // namespace

    // ------------------------------------------------------------------------
    // Pick the interior slices per chunk: the largest that fits in device
    // memory, but no more than an even share per GPU.
    // ------------------------------------------------------------------------
    template <typename T>
    size_t auto_chunk_slices(const ReconParams &params, int ngpus) {
        const dims_t dims = params.recon_dims;
        const bool tv = (params.regularizer == Regularizer::SPLIT_BREGMAN);

        // full-size device arrays alive at once for one chunk: Split-Bregman's
        // x-update / shrink passes hold about 16; CG's operator pass about 4.
        // (the PSF convolution sizes its own slabs from the memory left over)
        const size_t arrays = tv ? 16 : 4;
        const size_t halo = tv ? TV_HALO : 0;
        const size_t per_slice = arrays * dims.n2 * dims.n3 * sizeof(T);

        // leave room for the PSF slab, cuFFT workspace and the CUDA context
        const size_t budget = static_cast<size_t>(0.4 * min_free_bytes(ngpus));
        size_t mem_bound = budget / per_slice;
        mem_bound = (mem_bound > 2 * halo) ? mem_bound - 2 * halo : 1;

        const size_t per_gpu = (dims.n1 + ngpus - 1) / ngpus;
        return std::clamp<size_t>(std::min(per_gpu, mem_bound), 1, dims.n1);
    }

    template <typename T>
    Array<T> MBIR2D_streamed(const Array<T> &sinogram, const std::vector<T> &theta,
                             const ReconParams &params) {

        const dims_t recon_dims = params.recon_dims;
        // like the single-device path, only the first recon_dims[0] rows are used
        // (the config parser rounds even sizes down, so rows may exceed slices)
        if (sinogram.nslices() < recon_dims.n1) {
            throw std::invalid_argument(
                "MBIR2D_streamed: sinogram has fewer rows than recon_dims[0]");
        }
        const bool tv = (params.regularizer == Regularizer::SPLIT_BREGMAN);
        if (!tv && params.regularizer != Regularizer::UNCONSTRAINED) {
            throw std::invalid_argument("Unsupported optimizer type");
        }

        const int ngpus = resolve_num_gpus(params.numGpus);
        const size_t chunk_slices = (params.chunkSlices > 0)
                                        ? params.chunkSlices
                                        : auto_chunk_slices<T>(params, ngpus);
        // halo 0 for backprojection / CG; TV passes need halo along the slices
        const auto plain = make_chunks(recon_dims.n1, chunk_slices, 0);
        const auto halo = make_chunks(recon_dims.n1, chunk_slices, TV_HALO);
        std::cout << std::format(
            "Streaming {} slices in {} chunks of up to {} slices across {} GPU(s)\n",
            recon_dims.n1, plain.size(), chunk_slices, ngpus);

        std::cout << "Building 2D Toeplitz PSF on each GPU ...\n";
        ChunkExecutor<T> exec(theta, sinogram.ncols(), recon_dims.n3, ngpus);

        // right-hand side: backprojection of the measured sinogram
        std::cout << "Computing backprojection (right-hand side) ...\n";
        Array<T> rhs(recon_dims);
        exec.run(plain, [&](DeviceContext<T> &ctx, size_t, const SliceChunk &c) {
            DeviceArray<T> d_sino = upload_chunk(sinogram, c);
            auto bp = backproj2d(d_sino, ctx.pg,
                                 dims_t{c.size(), recon_dims.n2, recon_dims.n3});
            download_interior(bp, rhs, c);
        });

        if (tv) {
            std::cout << "Starting MBIR with Split-Bregman method ...\n";
            return streamed_split_bregman<T>(
                exec, halo, rhs, params.lambda, params.mu, params.maxIters,
                params.innerIters, params.tol, params.xtol);
        }
        std::cout << "Starting unconstrained reconstruction with CG ...\n";
        Array<T> x = Array<T>::zeros(recon_dims);
        StreamedOperator<T> A(exec, plain, false, T(0));
        CgWorkspace<T> ws(recon_dims);
        streamed_cg<T>(A, rhs, x, ws, true, params.maxIters, params.tol,
                       params.xtol);
        return x;
    }

    template Array<float> MBIR2D_streamed(const Array<float> &,
                                          const std::vector<float> &,
                                          const ReconParams &);
    template Array<double> MBIR2D_streamed(const Array<double> &,
                                           const std::vector<double> &,
                                           const ReconParams &);

    // ------------------------------------------------------------------------
    // Decide whether the whole volume can live on one device. Peak device memory
    // of the single-device solvers is about 27 N (Split-Bregman) / 11 N (CG)
    // elements for a volume of N elements.
    // ------------------------------------------------------------------------
    template <typename T>
    bool volume_fits_one_gpu(const ReconParams &params) {
        const dims_t d = params.recon_dims;
        const size_t N = d.n1 * d.n2 * d.n3;
        const size_t factor =
            (params.regularizer == Regularizer::SPLIT_BREGMAN) ? 27 : 11;
        return factor * N * sizeof(T) < static_cast<size_t>(0.8 * min_free_bytes(1));
    }

    template bool volume_fits_one_gpu<float>(const ReconParams &);
    template bool volume_fits_one_gpu<double>(const ReconParams &);

} // namespace tomocam::gpu
