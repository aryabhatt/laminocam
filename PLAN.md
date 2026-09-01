# PLAN: PCG Preconditioners for NUFFT-based Laminography Reconstruction

**Goal:** add Jacobi/DCF and Toeplitz-spectral preconditioners for the normal
operator `A = R^T R = NUFFT_type1(NUFFT_type2(x))`, usable inside PCG and
amortized across a split-Bregman/ADMM outer loop.

**Companion reference:** `nufft_preconditioners.pdf` / `.typ` (math derivations,
break-even analysis, R2C and 2,3-smooth padding rationale). This plan is the
*engineering* counterpart — it assumes that document's math and focuses on
where code goes and in what order.

---

## 0. Assumptions to verify against the actual codebase before starting

These are inferred from the conversation and are **not** confirmed against
your real headers — Stage 1, Step 0 is to confirm/correct them.

- [ ] `finufft_plan.h` exposes a type-1 (nonuniform → uniform) call that can be
      driven with an all-ones input vector (needed for the PSF) and a type-2
      (uniform → nonuniform) call (needed for `A x`, already used elsewhere).
- [ ] `fftw_plan.h` wraps real-to-complex / complex-to-real transforms
      (`fftw_plan_dft_r2c_3d` / `c2r`) in addition to (or instead of)
      complex-to-complex — confirm R2C is available; if not, this plan adds it.
- [ ] `toeplitz.h` already computes a PSF via `NUFFT_type1(ones)` and its
      spectrum `t_hat`, used to *replace* `A`'s application (fast forward
      operator, "route (a)" in the reference doc). **Preconditioning is
      "route (b)": build `1/t_hat`, apply it to the CG residual, while `A`
      itself keeps being applied exactly via the NUFFT pair.** Confirm
      `toeplitz.h` exposes `t_hat` (or the padded PSF) so `precond.h` can
      reuse it rather than recomputing a second PSF via a second NUFFT call.
- [ ] Existing CG/PCG solver location (file not specified) — find it and
      confirm it accepts an injectable preconditioner (`z = P(r)` hook). If
      it currently hardcodes `P = I`, that's a small edit needed in Stage 1
      too (see §4).
- [ ] Confirm the non-uniform sampling trajectory is k-space symmetric
      (`{k_j} = {-k_j}`) — true if each tilt's projection is a real-valued
      image whose full 2D FFT plane is used directly as the non-uniform
      points for that tilt. This licenses the R2C/C2R optimization
      (§2.3). If the trajectory is *not* symmetric (e.g. randomized
      sampling), R2C must be disabled (fallback flag, see §2.3).

---

## 1. Stage 1 scope (CPU only) — target file: `include/precond.h`

Stage 1 delivers a CPU-only, backend-agnostic preconditioner module that:

1. Defines a minimal `IPrecond` interface so the PCG solver can swap
   preconditioners (including `P = I`) without other code changes.
2. Implements `DensityCompPrecond` (density-compensation diagonal).
3. Implements `ToeplitzPrecond` (FFT-diagonal spectral inverse),
   built on top of `fftw_plan.h` and (ideally) reusing `toeplitz.h`'s
   existing PSF/spectrum machinery rather than duplicating it.
4. Supports the ADMM/split-Bregman amortization pattern: build once outside
   the outer loop, cheap `update_rho()` when `ρ` changes (§2.4 of the
   reference doc).
5. Ships with CPU unit tests and a PCG-iteration-count benchmark script,
   run and validated by the user before Stage 2 begins.

Stage 1 explicitly does **not** touch `cufft_plan.h` or `gpu/toeplitz.h`.
No GPU code, no `.cu` files, in this stage.

---

## 2. Interface design

### 2.1 `IPreconditioner` (new, in `include/precond.h`)

```cpp
// include/precond.h
#pragma once
#include <vector>
#include <cstddef>

// Backend-agnostic preconditioner interface. Stage 1 implementations are
// CPU-only; Stage 2 adds GPU implementations behind the same interface so
// the PCG solver never needs to know which backend is active.
class IPrecond {
public:
    virtual ~IPrecond() = default;

    // z = P * r. r and z are real-valued, length N (volume voxel count),
    // in the same voxel ordering as the rest of the solver.
    virtual void apply(const double* r, double* z, std::size_t N) const = 0;

    // Called when rho changes in the outer ADMM/split-Bregman loop.
    // Default no-op: only ToeplitzPreconditioner overrides this
    // meaningfully (see reference doc Sec. 5.4). JacobiPreconditioner may
    // also override if a cheap update is worthwhile; otherwise it can
    // require a full rebuild (rare — rho changes are infrequent).
    virtual void update_rho(double rho) { (void)rho; }
};

// P = I. Baseline for comparison; also useful as a safe default while
// wiring up the interface before real preconditioners are validated.
class IdentityPrecond : public IPrecond {
public:
    void apply(const double* r, double* z, std::size_t N) const override {
        std::copy(r, r + N, z);
    }
};
```

### 2.2 `DensityCompPrecond`

```cpp
class DensityCompPrecond : public IPrecond {
public:
    // nu_pts: non-uniform k-space sample coordinates (reuse whatever type
    // finufft_plan.h already uses for this — do not introduce a new point
    // type). vol_shape: (Nx, Ny, Nz). n_dcf_iter: Pipe-Menon iterations.
    DensityCompPrecond(const FinufftPlan& plan,
                          const std::array<int,3>& vol_shape,
                          int n_dcf_iter = 10,
                          double floor_frac = 1e-3);

    void apply(const double* r, double* z, std::size_t N) const override;

private:
    std::vector<double> diag_;   // d ~= diag(A), length N, precomputed once
    void compute_dcf_weights(const FinufftPlan& plan, int n_iter);
    void build_diag(const FinufftPlan& plan, const std::array<int,3>& shape,
                     double floor_frac);
};
```

`apply()` is `z[i] = r[i] / diag_[i]` — O(N), no FFT, no NUFFT. This matches
§3.2 of the reference doc.

### 2.3 `ToeplitzPrecond`

```cpp
class ToeplitzPrecond : public IPrecond {
public:
    // Preferred constructor: reuse an existing ToeplitzOperator's spectrum
    // instead of recomputing the PSF via a second NUFFT call.
    explicit ToeplitzPrecond(const ToeplitzOperator& fwd_op,
                                     double reg = 1e-2,
                                     bool assume_symmetric_traj = true);

    // Fallback constructor: build the spectrum independently (only if
    // ToeplitzOperator cannot be reused, e.g. different padding policy).
    ToeplitzPrecond(const FinufftPlan& plan,
                            const std::array<int,3>& vol_shape,
                            double reg = 1e-2,
                            bool assume_symmetric_traj = true);

    void apply(const double* r, double* z, std::size_t N) const override;
    void update_rho(double rho) override;   // t_hat_rho = t_hat + rho * d_hat

private:
    std::array<int,3> pad_shape_;      // 2N-1 per axis, rounded to 2^a*3^b
    std::array<int,3> vol_shape_;
    std::vector<double> t_hat_;        // base spectrum (rho = 0), R2C layout
    std::vector<double> t_hat_inv_;    // current 1/(t_hat + rho*d_hat + eps)
    std::vector<double> d_hat_;        // regularizer symbol (TV/Tikhonov), optional
    FftwPlan fft_r2c_, fft_c2r_;       // from fftw_plan.h
    double reg_;
    bool symmetric_traj_;

    void validate_psf_is_real(const std::vector<std::complex<double>>& psf) const;
    void recompute_inverse(double rho);
};
```

`apply()`: zero-pad `r` into `pad_shape_`, `fft_r2c_`, multiply by
`t_hat_inv_`, `fft_c2r_`, crop back to `vol_shape_`. Matches §4.1–4.3.

---

## 3. Implementation steps (Stage 1, CPU)

Suggested order — each step should compile and (where noted) pass a test
before moving to the next.

1. **Confirm §0 assumptions.** Read `finufft_plan.h`, `fftw_plan.h`,
   `toeplitz.h` and correct this plan's interface sketch (§2) to match real
   signatures/types before writing code against them.
2. **Add `include/precond.h` with `IPreconditioner` + `IdentityPreconditioner`
   only.** Wire this into the existing PCG solver as the active
   preconditioner (`P = I`), confirming the solver's convergence is
   unchanged versus its current unpreconditioned behavior. This isolates
   "does the injection point work" from "is the preconditioner correct."
3. **Implement `JacobiPreconditioner`.**
   - `compute_dcf_weights`: Pipe–Menon iteration using `finufft_plan.h`'s
     type-1/type-2 calls (reference doc §3.1, item 2).
   - `build_diag`: single adjoint NUFFT of the weights → real part → floor.
   - Unit test: on a small synthetic volume (e.g. 32³) with a known,
     simple trajectory (e.g. uniform radial), compare `diag_` against a
     brute-force dense computation of `diag(A)` (feasible only at this
     small size — see §5).
4. **Implement `ToeplitzPreconditioner`, symmetric-trajectory path first.**
   - Padding: `pad_shape = 2*vol_shape - 1` per axis, rounded up to nearest
     `2^a * 3^b` (reference doc's `next_2_3_smooth`; port directly, it's
     ~15 lines and backend-independent — see §3.6 below).
   - PSF: reuse `toeplitz.h`'s existing PSF/spectrum if it already computes
     one at a compatible padded size; otherwise compute independently via
     `finufft_plan.h` type-1 of an all-ones vector. **Prefer reuse** — flag
     in code review if `toeplitz.h`'s padding convention differs (e.g. if
     it uses `2N` instead of `2N-1`/`2^a3^b`) and decide whether to
     harmonize both to the same padded size (recommended, avoids two PSF
     computations existing side by side).
   - `validate_psf_is_real`: check `max(|Im(psf)|) < 1e-3 * max(|Re(psf)|)`;
     throw/log a clear error if violated, telling the caller to pass
     `assume_symmetric_traj = false`.
   - R2C build: `fftw_plan.h`'s r2c/c2r wrappers on the (real) PSF and on
     `r` during `apply()`.
   - Non-symmetric fallback: full complex `fftw_plan.h` c2c transforms,
     `assume_symmetric_traj = false`, no `validate_psf_is_real` call.
5. **`update_rho()`.** Precompute `d_hat_` once (closed-form finite-difference
   regularizer symbol, reference doc §5.4) if a regularizer term is in play;
   `recompute_inverse(rho)` is then one array-add + one reciprocal, no FFTs.
   If there's no regularizer yet in the existing solver, implement this but
   leave it untriggered (`d_hat_` all-zero / unused) until ADMM lands.
6. **Port `next_2_3_smooth` as a small free function** (in `precond.h` or a
   shared `fft_utils.h` if one exists) — self-contained, no external
   dependency, already validated against brute force in the reference
   conversation (correct for `n` up to at least 4095; the algorithm is
   general, not size-limited).
7. **Wire `JacobiPreconditioner` and `ToeplitzPreconditioner` into the PCG
   solver as selectable options** (e.g. an enum/config flag
   `PreconditionerKind::{Identity, Jacobi, Toeplitz}`), so all three can be
   benchmarked back-to-back on the same problem without recompiling.

---

## 4. PCG solver integration

Find the existing CG/PCG implementation and confirm it has (or add) an
injection point of this shape:

```cpp
// before:
// r = b - A(x); p = r; ...
// after:
IPreconditioner* P = make_preconditioner(config.precond_kind, ...);
r = b - A(x);
z = P->apply(r);
p = z;
double rz_old = dot(r, z);
for (...) {
    Ap = A(p);
    alpha = rz_old / dot(p, Ap);
    x += alpha * p;
    r -= alpha * Ap;
    if (converged(r)) break;
    z = P->apply(r);              // <-- the only new call inside the loop
    double rz_new = dot(r, z);
    beta = rz_new / rz_old;
    p = z + beta * p;
    rz_old = rz_new;
}
```

If the outer ADMM/split-Bregman loop already exists, confirm `P` is built
**once before the loop**, not once per outer step (reference doc §5.1–5.3):

```cpp
IPreconditioner* P = make_preconditioner(...);   // built once
for (int k = 0; k < n_bregman_steps; ++k) {
    b_k = compute_rhs(x, aux_k, rho);
    x = pcg(b_k, A_rho, P, x /* warm start */, max_inner_iter);
    aux_k = update_bregman_aux(x, aux_k);
    // only if rho changed this step:
    // P->update_rho(rho);
}
```

If no such outer loop exists yet, note this as a follow-up integration
point rather than building it as part of this plan.

---

## 5. Stage 1 testing plan (CPU)

- **Correctness, small scale (≤ 64³):**
  - `JacobiPreconditioner`: compare `diag_` to a dense brute-force
    `diag(A)` computed by applying `A` to each unit basis vector (feasible
    only at small N — this is a one-off correctness check, not a
    perf-relevant path).
  - `ToeplitzPreconditioner`: compare `P->apply(r)` against a dense
    `A^{-1} r` (via e.g. LU/CG-to-convergence on the small dense matrix)
    to confirm it's a *reasonable* approximate inverse, not exact — expect
    moderate residual reduction, not exact match.
  - R2C path: confirm `t_hat_` from R2C matches the redundant half of a
    full complex FFT of the same PSF, bit-for-bit up to floating-point
    tolerance.
  - `next_2_3_smooth`: unit test against brute-force smoothness check for
    a range of `n` (mirrors the validation already done in the reference
    conversation).
- **Iteration-count benchmark, realistic scale:** run PCG with `Identity`,
  `Jacobi`, and `Toeplitz` on the actual (or representative) missing-wedge
  trajectory, log residual-norm-vs-iteration for each, and wall-clock time
  per outer Bregman step if the ADMM loop is wired up. This is the number
  that decides whether either preconditioner is worth keeping — see the
  break-even analysis in the reference doc (§5.2).
- **Regularization floor sensitivity:** sweep `reg` (Toeplitz) and
  `floor_frac` (Jacobi) and confirm reconstruction quality and iteration
  count are not overly sensitive near the missing-wedge directions, where
  `t_hat` is smallest.

**Stage 1 exit criterion:** the user runs the above on real data/geometry
and decides (a) which preconditioner(s) are worth porting to GPU, and
(b) what `reg`/`floor_frac` defaults to carry into Stage 2.

---

## 6. Stage 2 (GPU) — deferred, implemented by the user after Stage 1 validation

Not implemented in this pass. Left here as a scoped follow-up so the CPU
interface doesn't need to change shape later:

- [ ] Mirror `JacobiPreconditioner` / `ToeplitzPreconditioner` as
      `JacobiPreconditionerGPU` / `ToeplitzPreconditionerGPU`, implementing
      the same `IPreconditioner` interface, backed by `cufft_plan.h` and
      `gpu/toeplitz.h` instead of `fftw_plan.h`/CPU `toeplitz.h`.
- [ ] `cuFFT` R2C/C2R: confirm `cufft_plan.h` exposes `CUFFT_R2C`/`CUFFT_C2R`
      plan types (analogous to the FFTW R2C/C2R used in Stage 1).
- [ ] Re-validate `next_2_3_smooth` sizing choice specifically for cuFFT —
      this was the motivating reason for restricting to `2^a * 3^b` instead
      of allowing 5/7/11 factors (cuFFT's fast-path coverage differs from
      FFTW/pocketfft's); confirm against current cuFFT documentation/release
      notes rather than assuming.
- [ ] Device-memory management for `t_hat_inv_` / `diag_`: decide whether
      these persist device-side for the whole run (likely, given they're
      built once and reused ~100× per the amortization argument in the
      reference doc) or are re-uploaded each PCG call.
- [ ] Only start this stage after Stage 1's iteration-count benchmark
      (§5) confirms which preconditioner(s) justify the GPU port.

---

## 7. File-level change summary (Stage 1 only)

| File | Change |
|---|---|
| `include/precond.h` | **New.** `IPreconditioner`, `IdentityPreconditioner`, `JacobiPreconditioner`, `ToeplitzPreconditioner`, `next_2_3_smooth`. |
| `finufft_plan.h` | No change expected; confirm it exposes what §0 assumes. |
| `fftw_plan.h` | Possibly extend if R2C/C2R wrappers don't already exist. |
| `toeplitz.h` | Possibly extend to expose `t_hat`/PSF for reuse by `precond.h` (avoid duplicate NUFFT call); confirm padding convention matches `2N-1`/`2^a3^b` or harmonize. |
| PCG solver (file TBD, see §0) | Add `IPreconditioner*` injection point per §4; build `P` once outside the Bregman loop if one exists. |
| `cufft_plan.h`, `gpu/toeplitz.h` | **Untouched in Stage 1.** |

---

## 8. Checklist

- [ ] §0 assumptions confirmed/corrected against real headers
- [ ] `IPreconditioner` + `IdentityPreconditioner` wired into PCG, baseline unchanged
- [ ] `JacobiPreconditioner` implemented + unit tested
- [ ] `ToeplitzPreconditioner` implemented (symmetric path) + unit tested
- [ ] R2C/C2R validated exact vs. full complex FFT
- [ ] `next_2_3_smooth` ported + unit tested
- [ ] Non-symmetric-trajectory fallback implemented (or explicitly deferred if not needed)
- [ ] `update_rho()` implemented and tested if ADMM/split-Bregman loop exists
- [ ] Iteration-count benchmark run on real geometry, results reviewed
- [ ] Decision made on which preconditioner(s) proceed to Stage 2
- [ ] Stage 2 scoped as a separate task, started only after the above
