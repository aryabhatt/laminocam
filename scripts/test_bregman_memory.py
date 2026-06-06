#!/usr/bin/env python3
"""
test_bregman_memory.py
-----------------------
Read a reconstruction config.toml and produce a full GPU memory estimate
covering every major allocation in the MBIR pipeline:

  1. Input sinogram (from TIFF) and padded copy
  2. PolarGrid coordinate arrays (x, y, z) on the GPU
  3. Backprojected RHS (yT) and initial guess (x0)
  4. split_bregman + cgsolver persistent and transient arrays
  5. A-operator (sysmat) allocations:
       - forward() / backward() working complex arrays
       - cuFFT plan workspace
       - cuFINUFFT internal oversampled grid (dominant hidden cost!)

cuFINUFFT workspace note
------------------------
cuFINUFFT is called with upsampfac=1.25 (see include/gpu/cufinufft_plan.h).
For a 3D transform with N1×N2×N3 output Fourier modes, the internal
oversampled FFT grid is (1.25·N1)×(1.25·N2)×(1.25·N3) complex elements
≈ 1.953·N_pad complex (1.25^3).
At sizeof(complex<T>) = 2·sizeof(T), this is ≈3.906·N_pad T-element equivalents.
Compare to upsampfac=2.0 which would be 8·N_pad complex = 16·N_pad real — ~4× more.

Usage
-----
    python scripts/test_bregman_memory.py dino/tv/config.toml
    python scripts/test_bregman_memory.py dino/tv/config.toml --verbose
    python scripts/test_bregman_memory.py config.toml \\
        --sino-shape 84 326 303   # override if TIFF not accessible
"""

from __future__ import annotations

import argparse
import math
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Re-use the core estimator from estimate_bregman_memory.py
sys.path.insert(0, str(Path(__file__).parent))
from estimate_bregman_memory import (ALLOCATIONS, PHASE_ORDER,
                                     compute_watermarks, _human, _try_tabulate)


# ---------------------------------------------------------------------------
# Config + sinogram helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def extract_params(cfg: dict) -> dict:
    rp = cfg.get("recon_params", {})
    dims = rp.get("recon_dims", None)
    if dims is None or len(dims) != 3:
        raise ValueError("recon_params.recon_dims must be a 3-element list")
    reg = rp.get("regularizer", {})
    method = reg.get("method", "unknown")
    sb = reg.get("split_bregman", {})
    inp = cfg.get("input", [{}])
    if isinstance(inp, list):
        inp = inp[0]
    return {
        "dims": tuple(dims),
        "method": method,
        "max_iters": rp.get("max_iters", "?"),
        "tol": rp.get("tol", "?"),
        "xtol": rp.get("xtol", "?"),
        "inner_iters": sb.get("inner_iters", "?"),
        "lambda_": sb.get("lambda", "?"),
        "mu": sb.get("mu", "?"),
        "tiff_path": inp.get("filename", ""),
        "angles_path": inp.get("angles", ""),
    }


def read_sino_shape(tiff_path: str) -> Optional[Tuple[int, int, int]]:
    """Return (nangles, nrows, ncols) by reading TIFF metadata only."""
    try:
        import tifffile
        with tifffile.TiffFile(tiff_path) as tf:
            s = tf.series[0].shape
        if len(s) == 3:
            return tuple(s)
        if len(s) == 2:
            return (1, s[0], s[1])
    except Exception:
        pass
    return None


def count_angles(angles_path: str) -> Optional[int]:
    try:
        p = Path(angles_path)
        if p.exists():
            return sum(1 for ln in p.read_text().splitlines() if ln.strip())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Padded-dimension helpers (mirrors mbir.cu logic)
# ---------------------------------------------------------------------------

PAD_FACTOR = math.sqrt(2)  # ReconParams::PAD_FACTOR


def padded_size(n: int) -> int:
    """Compute padded dim: n * PAD_FACTOR, forced to odd (matching mbir.cu)."""
    p = int(n * PAD_FACTOR)
    if p % 2 == 0:
        p -= 1
    return p


# ---------------------------------------------------------------------------
# Allocation catalogue for the full MBIR pipeline
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    name: str
    nbytes: float          # actual bytes (not in N units)
    phase: str
    notes: str = ""
    transient: bool = False


def build_full_catalogue(
    sino_shape: Tuple[int, int, int],   # (nangles, nrows, ncols)
    recon_dims: Tuple[int, int, int],   # from config (unpadded)
    itemsize: int,
) -> Tuple[List[Entry], dict]:
    """
    Return (entries, summary) where summary holds key sizes.
    All sizes are computed analytically from the source code.
    """
    nangles, nrows, ncols = sino_shape
    n1, n2, n3 = recon_dims

    M       = nangles * nrows * ncols           # raw sinogram elements
    # Padded sinogram (nrows/ncols are padded by PAD_FACTOR)
    nrows_p = padded_size(nrows)
    ncols_p = padded_size(ncols)
    M_pad   = nangles * nrows_p * ncols_p       # padded sinogram

    # Padded recon (all 3 dims padded, forced odd)
    n1_p    = padded_size(n1)
    n2_p    = padded_size(n2)
    n3_p    = padded_size(n3)
    N_pad   = n1_p * n2_p * n3_p               # padded recon (= N for Bregman)

    cplx    = 2 * itemsize                      # bytes per complex<T> element

    UPSAMP = 1.25                             # cufinufft_plan.h: opts.upsampfac
    nufft_ovs = UPSAMP ** 3                   # ≈ 1.953 for upsampfac=1.25

    entries: List[Entry] = []

    # ── Phase 1: MBIR setup (before split_bregman) ────────────────────────
    entries.append(Entry(
        "projs  (sinogram on GPU, const ref)",
        M * itemsize, "MBIR setup",
        f"({nangles}×{nrows}×{ncols}) real — loaded from TIFF, lives while MBIR runs"))
    entries.append(Entry(
        "y = projs / scale  (normalised copy)",
        M * itemsize, "MBIR setup",
        "freed after pad2d", transient=True))
    entries.append(Entry(
        "y = pad2d(y, √2)  (padded sinogram)",
        M_pad * itemsize, "MBIR setup",
        f"({nangles}×{nrows_p}×{ncols_p}) real — lives during backproj"))
    entries.append(Entry(
        "PolarGrid::x", M_pad * itemsize, "MBIR setup",
        "GPU coordinate array, lives throughout MBIR"))
    entries.append(Entry(
        "PolarGrid::y", M_pad * itemsize, "MBIR setup",
        "GPU coordinate array, lives throughout MBIR"))
    entries.append(Entry(
        "PolarGrid::z", M_pad * itemsize, "MBIR setup",
        "GPU coordinate array, lives throughout MBIR"))
    entries.append(Entry(
        "yT = backproj(y, pg, out_dims)",
        N_pad * itemsize, "MBIR setup",
        f"({n1_p}×{n2_p}×{n3_p}) real — RHS for split_bregman; lives throughout"))
    entries.append(Entry(
        "x0 = DeviceArray(out_dims)  [zero init]",
        N_pad * itemsize, "MBIR setup",
        "initial guess; becomes x inside split_bregman"))

    # ── Phase 2: split_bregman persistent ─────────────────────────────────
    N_bytes = N_pad * itemsize
    for lbl, note in [
        ("x  (solution iterate)",     "x0.clone()"),
        ("x_old (prev iterate)",      "x0.clone()"),
        ("b[0]", "Bregman multiplier"),
        ("b[1]", "Bregman multiplier"),
        ("b[2]", "Bregman multiplier"),
        ("d[0]", "TV auxiliary variable"),
        ("d[1]", "TV auxiliary variable"),
        ("d[2]", "TV auxiliary variable"),
    ]:
        entries.append(Entry(lbl, N_bytes, "split_bregman persistent", note))

    # ── Phase 3: outer-iter workspace ─────────────────────────────────────
    for lbl, note in [
        ("d_b[0]", "d[0] - b[0]"),
        ("d_b[1]", "d[1] - b[1]"),
        ("d_b[2]", "d[2] - b[2]"),
        ("rhs",    "yT.clone() modified in-place"),
    ]:
        entries.append(Entry(lbl, N_bytes, "outer-iter workspace", note))
    for lbl, note in [
        ("neg_div: divergence(d_b)", "transient inside neg_divergence"),
        ("neg_div: div * (-1)",      "transient inside neg_divergence"),
    ]:
        entries.append(Entry(lbl, N_bytes, "outer-iter workspace", note,
                             transient=True))

    # ── Phase 4: cgsolver persistent ──────────────────────────────────────
    for lbl, note in [
        ("ones",        "all-ones for preconditioner"),
        ("pre = A(ones)", "preconditioner diagonal"),
        ("x_cg",        "x0.clone() inside cgsolver"),
        ("r = y - A(x)", "residual"),
        ("z = precond(r)", "preconditioned residual"),
        ("p = z.clone()", "search direction"),
    ]:
        entries.append(Entry(lbl, N_bytes, "cgsolver persistent", note))

    # ── Phase 5: A-operator transients (during Ap = A(p)) ─────────────────
    # forward(p, pg):
    entries.append(Entry(
        "A::forward – Ft = to_complex(p)",
        N_pad * cplx, "A-operator transient",
        "complex volume, 2× real size", transient=True))
    entries.append(Entry(
        "A::forward – C = DeviceArray<complex>(pg.dims())",
        M_pad * cplx, "A-operator transient",
        f"NU output ({nangles}×{nrows_p}×{ncols_p}) complex", transient=True))
    entries.append(Entry(
        "A::forward – cuFINUFFT type-2 workspace  [upsampfac=1.25]",
        N_pad * cplx * nufft_ovs, "A-operator transient",
        f"(1.25·n1_p)×(1.25·n2_p)×(1.25·n3_p) complex oversampled grid ≈ {nufft_ovs:.3f}×N_pad complex",
        transient=True))
    entries.append(Entry(
        "A::forward – fft::ifft2d output copy",
        M_pad * cplx, "A-operator transient",
        "output array allocated inside fft2d/ifft2d", transient=True))
    entries.append(Entry(
        "A::forward – cuFFT C2C workspace",
        M_pad * cplx, "A-operator transient",
        "internal cuFFT scratch ~ 1× data size", transient=True))
    # backward(proj, pg, out_dims):
    entries.append(Entry(
        "A::backward – C = to_complex(proj)",
        M_pad * cplx, "A-operator transient",
        "complex sinogram", transient=True))
    entries.append(Entry(
        "A::backward – fft::fft2d output copy",
        M_pad * cplx, "A-operator transient",
        "output array allocated inside fft2d", transient=True))
    entries.append(Entry(
        "A::backward – cuFFT C2C workspace",
        M_pad * cplx, "A-operator transient",
        "internal cuFFT scratch ~ 1× data size", transient=True))
    entries.append(Entry(
        "A::backward – F = DeviceArray<complex>(out_dims)",
        N_pad * cplx, "A-operator transient",
        "uniform Fourier modes output", transient=True))
    entries.append(Entry(
        "A::backward – cuFINUFFT type-1 workspace  [upsampfac=1.25]",
        N_pad * cplx * nufft_ovs, "A-operator transient",
        f"(1.25·n1_p)×(1.25·n2_p)×(1.25·n3_p) complex oversampled grid ≈ {nufft_ovs:.3f}×N_pad complex",
        transient=True))
    # neg_laplacian(p) inside Ap lambda:
    entries.append(Entry(
        "Ap lambda – neg_laplacian(p): laplacian(p)",
        N_bytes, "A-operator transient",
        "inside Ap = A(p) + mu * neg_laplacian(p)", transient=True))
    entries.append(Entry(
        "Ap lambda – neg_laplacian(p): lap * (-1)",
        N_bytes, "A-operator transient",
        "negation copy; lap freed after", transient=True))

    # ── Phase 6: remaining CG-iter transients ─────────────────────────────
    entries.append(Entry(
        "CG – p * alpha (for dx)",
        N_bytes, "CG-iter other transient",
        "step-size computation", transient=True))
    entries.append(Entry(
        "CG – z_new = precond(r) (replaces z)",
        N_bytes, "CG-iter other transient",
        "new z before old z freed", transient=True))

    summary = dict(
        M=M, M_pad=M_pad, N_pad=N_pad,
        nangles=nangles, nrows=nrows, ncols=ncols,
        nrows_p=nrows_p, ncols_p=ncols_p,
        n1_p=n1_p, n2_p=n2_p, n3_p=n3_p,
        itemsize=itemsize,
    )
    return entries, summary


# ---------------------------------------------------------------------------
# Watermark calculator over phase groups
# ---------------------------------------------------------------------------

# Phases ordered by when they become live
FULL_PHASE_ORDER = [
    "MBIR setup",
    "split_bregman persistent",
    "outer-iter workspace",
    "cgsolver persistent",
    "A-operator transient",
    "CG-iter other transient",
]


def full_watermark(entries: List[Entry]) -> dict:
    """
    Compute cumulative bytes at each phase boundary.
    Persistent entries (transient=False) accumulate; transients add to the peak
    at their phase but are not carried forward.
    """
    persistent: float = 0.0
    phase_peaks = {}
    for phase in FULL_PHASE_ORDER:
        phase_entries = [e for e in entries if e.phase == phase]
        persistent_here = sum(e.nbytes for e in phase_entries if not e.transient)
        transient_here  = sum(e.nbytes for e in phase_entries if e.transient)
        persistent += persistent_here
        phase_peaks[phase] = persistent + transient_here

    return phase_peaks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full MBIR GPU memory estimate from config.toml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", type=Path, help="Path to config.toml")
    parser.add_argument("--dtype", choices=["float32", "float64"],
                        default="float32",
                        help="Element type (default: float32)")
    parser.add_argument("--sino-shape", metavar=("Na", "Nr", "Nc"), nargs=3,
                        type=int, default=None,
                        help="Override sinogram shape if TIFF is not accessible")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every individual allocation")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg    = load_config(args.config)
    params = extract_params(cfg)
    n1, n2, n3 = params["dims"]
    itemsize = 4 if args.dtype == "float32" else 8

    # ── resolve sinogram shape ─────────────────────────────────────────────
    sino_shape = None
    if args.sino_shape:
        sino_shape = tuple(args.sino_shape)
        sino_src = "--sino-shape argument"
    else:
        sino_shape = read_sino_shape(params["tiff_path"])
        if sino_shape:
            sino_src = f"TIFF metadata: {params['tiff_path']}"
        else:
            # Fall back: angles count × recon n2 × n3
            na = count_angles(params["angles_path"]) or "?"
            print(f"Warning: cannot read TIFF '{params['tiff_path']}'.")
            print(f"  Use --sino-shape Na Nr Nc to specify the sinogram dimensions.")
            sys.exit(1)

    nangles, nrows, ncols = sino_shape

    # ── print header ───────────────────────────────────────────────────────
    print(f"\nConfig       : {args.config}")
    print(f"Method       : {params['method']}")
    print(f"  outer iters: {params['max_iters']}   inner iters: {params['inner_iters']}")
    print(f"  lambda={params['lambda_']}  mu={params['mu']}  "
          f"tol={params['tol']}  xtol={params['xtol']}")
    print()
    print(f"Sinogram     : ({nangles}×{nrows}×{ncols})  [{sino_src}]")
    print(f"Recon dims   : ({n1}×{n2}×{n3})  →  padded: "
          f"({padded_size(n1)}×{padded_size(n2)}×{padded_size(n3)})  "
          f"[PAD_FACTOR=√2={PAD_FACTOR:.4f}]")
    print(f"Element type : {args.dtype}  ({itemsize} bytes)")
    print()

    entries, summ = build_full_catalogue(sino_shape, (n1, n2, n3), itemsize)
    N_pad   = summ["N_pad"]
    M_pad   = summ["M_pad"]
    N_bytes = N_pad * itemsize

    if args.verbose:
        rows = []
        for e in entries:
            rows.append([e.name, e.phase, _human(e.nbytes),
                         "transient" if e.transient else "persistent",
                         e.notes])
        print(_try_tabulate(rows,
              ["Name", "Phase", "Bytes", "Kind", "Notes"]))
        print()

    # ── per-phase watermark ────────────────────────────────────────────────
    peaks = full_watermark(entries)

    phase_rows = []
    running_persistent = 0.0
    for phase in FULL_PHASE_ORDER:
        es = [e for e in entries if e.phase == phase]
        p_bytes = sum(e.nbytes for e in es if not e.transient)
        t_bytes = sum(e.nbytes for e in es if e.transient)
        running_persistent += p_bytes
        phase_rows.append([
            phase,
            f"+{_human(p_bytes)} persistent" + (f"  +{_human(t_bytes)} transient" if t_bytes else ""),
            _human(peaks[phase]),
        ])

    print("Per-phase peak watermark (cumulative persistent + phase transients)")
    print(_try_tabulate(phase_rows,
                        ["Phase", "Added this phase", "Peak at phase"]))
    print()

    # ── verification summary ───────────────────────────────────────────────
    overall_peak   = max(peaks.values())
    peak_phase     = max(peaks, key=peaks.get)
    bregman_only   = peaks.get("CG-iter other transient", 0)  # rough
    nufft_overhead = N_pad * 2 * itemsize * (1.25 ** 3) * 2  # 2 cuFINUFFT calls × 1.25^3 N complex

    print("─" * 60)
    print(f"Overall peak              : {_human(overall_peak)}")
    print(f"  at phase                : {peak_phase}")
    print(f"  N_pad (padded recon)    : {N_pad:,}  ({_human(N_bytes)} real)")
    print(f"  M_pad (padded sino)     : {M_pad:,}  ({_human(M_pad * itemsize)} real)")
    print()
    print("Unaccounted items in the original bregman-only estimate:")
    for phase, label in [
        ("MBIR setup",            "MBIR setup (sinogram, PolarGrid, yT, x0)"),
        ("A-operator transient",  "A-operator (forward+backward FFT + cuFINUFFT)"),
    ]:
        es = [e for e in entries if e.phase == phase]
        b = sum(e.nbytes for e in es)
        print(f"  {label:50s}: {_human(b)}")
    print()
    print("  ⚠️  cuFINUFFT oversampled-grid workspace (2 calls in sysmat, upsampfac=1.25):"
          f" {_human(nufft_overhead)}")
    print("     [Was ~10 GiB at upsampfac=2.0; ~4× reduction from dropping to 1.25]")
    print()


if __name__ == "__main__":
    main()
