#!/usr/bin/env python3
"""
estimate_bregman_memory.py
--------------------------
Analytically estimate the peak GPU memory used by the Split-Bregman solver
(`split_bregman` + `cgsolver`) defined in src/gpu/bregman.cu and
src/gpu/conjgrad.cu.

Usage
-----
    python scripts/estimate_bregman_memory.py --shape 256 256 256
    python scripts/estimate_bregman_memory.py --shape 512 512 512 --dtype float64
    python scripts/estimate_bregman_memory.py --shape 128 256 512 --verbose

The script treats the external `A` operator (Radon / NUFFT) as a black box and
does **not** include its internal allocations.

Memory model
------------
All arrays are 3-D volumes of shape (n1, n2, n3), N = n1*n2*n3 elements.
Arrays are grouped by how long they live:

  Persistent (split_bregman, entire function lifetime)
  ├─ x          1 N   solution iterate (clone of x0)
  ├─ x_old      1 N   previous iterate for convergence check
  ├─ b[0..2]    3 N   Bregman multipliers
  └─ d[0..2]    3 N   auxiliary TV variables
                ────
                8 N

  Outer-iteration workspace (allocated at loop top, freed after cgsolver)
  ├─ d_b[0..2]  3 N   d[i] - b[i]
  ├─ rhs        1 N   clone of yT, modified in-place
  └─ neg_divergence transient
     ├─ div     1 N   divergence(d_b)              ← live only during call
     └─ -div    1 N   div * (-1)                   ← replaces div, returned
     peak contribution: +2 N (the two arrays exist simultaneously briefly)
                ────
                +4 N live when cgsolver starts, +2 N transient peak

  cgsolver persistent (allocated once, freed when cgsolver returns)
  ├─ ones       1 N   all-ones vector for preconditioner
  ├─ pre        1 N   preconditioner diagonal A(ones)
  ├─ x_cg       1 N   CG solution
  ├─ r          1 N   residual
  ├─ z          1 N   preconditioned residual
  └─ p          1 N   search direction
                ────
                +6 N

  Per-CG-iteration transients (peak within one CG step)
  ├─ Ap = A(p)              1 N   A applied to search dir
  ├─ neg_laplacian(p) peak  2 N   laplacian(p) and * (-1) coexist briefly
  ├─ p * alpha              1 N   for dx / step-size computation
  └─ z_new (before z freed) 1 N   new z = precond_apply(r) replaces old z
                            ────
                            ~4 N  (some overlap, this is the conservative bound)

  High-water mark (inside CG, worst case):
      8 + 4 + 2 + 6 + 4 = 24 N    (conservative: neg_div + CG transients overlap)
  Typical high-water mark:
      8 + 4 + 6 + 4     = 22 N    (neg_div transient resolved before cgsolver)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Allocation:
    """One DeviceArray allocation and its lifetime context."""
    name: str
    multiplier: float         # in units of N
    phase: str                # group name for the table
    notes: str = ""
    transient: bool = False   # True → only exists briefly, not in steady state


# ---------------------------------------------------------------------------
# Allocation catalogue  (derived from bregman.cu and conjgrad.cu)
# ---------------------------------------------------------------------------

ALLOCATIONS: List[Allocation] = [
    # ── split_bregman persistent ─────────────────────────────────────────
    Allocation("x  (solution iterate)", 1, "split_bregman persistent",
               "x0.clone()"),
    Allocation("x_old (prev iterate)", 1, "split_bregman persistent",
               "x0.clone()"),
    Allocation("b[0]", 1, "split_bregman persistent", "Bregman multiplier"),
    Allocation("b[1]", 1, "split_bregman persistent", "Bregman multiplier"),
    Allocation("b[2]", 1, "split_bregman persistent", "Bregman multiplier"),
    Allocation("d[0]", 1, "split_bregman persistent", "TV auxiliary variable"),
    Allocation("d[1]", 1, "split_bregman persistent", "TV auxiliary variable"),
    Allocation("d[2]", 1, "split_bregman persistent", "TV auxiliary variable"),

    # ── outer-iteration workspace ─────────────────────────────────────────
    Allocation("d_b[0]", 1, "outer-iter workspace", "d[0] - b[0]"),
    Allocation("d_b[1]", 1, "outer-iter workspace", "d[1] - b[1]"),
    Allocation("d_b[2]", 1, "outer-iter workspace", "d[2] - b[2]"),
    Allocation("rhs", 1, "outer-iter workspace", "yT.clone() modified in-place"),
    # neg_divergence: div (N) and div*(-1) (N) coexist briefly
    Allocation("neg_div: divergence(d_b)", 1, "outer-iter workspace",
               "temporary inside neg_divergence", transient=True),
    Allocation("neg_div: div * (-1)",       1, "outer-iter workspace",
               "temporary inside neg_divergence", transient=True),

    # ── cgsolver persistent ───────────────────────────────────────────────
    Allocation("ones", 1, "cgsolver persistent",
               "all-ones vector for preconditioner"),
    Allocation("pre = A(ones)", 1, "cgsolver persistent",
               "preconditioner diagonal"),
    Allocation("x_cg", 1, "cgsolver persistent",
               "x0.clone() inside cgsolver"),
    Allocation("r = y - A(x)", 1, "cgsolver persistent",
               "residual; A(x) is transient"),
    Allocation("z = precond(r)", 1, "cgsolver persistent",
               "preconditioned residual"),
    Allocation("p = z.clone()", 1, "cgsolver persistent",
               "search direction"),

    # ── per-CG-iteration transients ───────────────────────────────────────
    Allocation("Ap = A(p) result", 1, "CG-iter transient",
               "result of operator application", transient=True),
    Allocation("neg_laplacian(p): laplacian(p)", 1, "CG-iter transient",
               "inside Ap lambda: laplacian result", transient=True),
    Allocation("neg_laplacian(p): lap * (-1)", 1, "CG-iter transient",
               "inside Ap lambda: negation; lap freed after", transient=True),
    Allocation("p * alpha (for dx)", 1, "CG-iter transient",
               "step-size computation", transient=True),
    Allocation("z_new = precond(r) (replaces z)", 1, "CG-iter transient",
               "new z before old z freed", transient=True),
]


# ---------------------------------------------------------------------------
# High-watermark calculation
# ---------------------------------------------------------------------------

PHASE_ORDER = [
    "split_bregman persistent",
    "outer-iter workspace",
    "cgsolver persistent",
    "CG-iter transient",
]


def compute_watermarks(allocations: List[Allocation]) -> dict:
    """
    Return a dict mapping phase label → cumulative N-multiplier at that phase.

    For each phase we include:
      - all persistent allocations from earlier phases (still live)
      - all allocations in the current phase (persistent + transient)

    The 'typical' watermark excludes the transient neg_divergence in the
    outer-iter workspace phase (those resolve before cgsolver is called).
    """
    phase_totals: dict[str, float] = {}
    running = 0.0
    for phase in PHASE_ORDER:
        phase_allocs = [a for a in allocations if a.phase == phase]
        phase_total = sum(a.multiplier for a in phase_allocs)
        running += phase_total
        phase_totals[phase] = running

    # Also compute a "typical" watermark: outer-iter transient (neg_div, +2N)
    # resolves before cgsolver entry, so peak is without those 2N in practice.
    typical_peak = phase_totals.get("CG-iter transient", 0) - 2.0  # remove neg_div

    return {
        "phases": phase_totals,
        "conservative_peak": phase_totals.get("CG-iter transient", 0),
        "typical_peak": typical_peak,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _human(nbytes: float) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.2f} GiB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MiB"
    return f"{nbytes / (1 << 10):.1f} KiB"


def _try_tabulate(rows, headers):
    try:
        from tabulate import tabulate  # type: ignore
        return tabulate(rows, headers=headers, tablefmt="simple")
    except ImportError:
        col_widths = [max(len(str(r[i])) for r in [headers] + rows)
                      for i in range(len(headers))]
        sep = "  ".join("-" * w for w in col_widths)
        lines = ["  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))]
        lines.append(sep)
        for row in rows:
            lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate peak GPU memory for split_bregman + cgsolver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--shape", metavar=("N1", "N2", "N3"), nargs=3,
                        type=int, required=True,
                        help="Volume dimensions n1 n2 n3")
    parser.add_argument("--dtype", choices=["float32", "float64"],
                        default="float32",
                        help="Element type (default: float32)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every individual allocation")
    args = parser.parse_args()

    n1, n2, n3 = args.shape
    N = n1 * n2 * n3
    itemsize = 4 if args.dtype == "float32" else 8
    bytes_per_N = N * itemsize

    print(f"\nMemory estimate for split_bregman + cgsolver")
    print(f"  Volume shape : {n1} × {n2} × {n3}  (N = {N:,})")
    print(f"  Element type : {args.dtype}  ({itemsize} bytes)")
    print(f"  1 N          = {_human(bytes_per_N)}\n")

    if args.verbose:
        rows = []
        for a in ALLOCATIONS:
            nbytes = a.multiplier * bytes_per_N
            rows.append([
                a.name,
                a.phase,
                f"{a.multiplier:.1f} N",
                _human(nbytes),
                "transient" if a.transient else "persistent",
                a.notes,
            ])
        headers = ["Name", "Phase", "Size (×N)", "Bytes", "Kind", "Notes"]
        print(_try_tabulate(rows, headers))
        print()

    wm = compute_watermarks(ALLOCATIONS)

    phase_rows = []
    for phase in PHASE_ORDER:
        mult = wm["phases"][phase]
        nbytes = mult * bytes_per_N
        phase_allocs = [a for a in ALLOCATIONS if a.phase == phase]
        phase_mult = sum(a.multiplier for a in phase_allocs)
        phase_rows.append([
            phase,
            f"+{phase_mult:.0f} N",
            f"{mult:.0f} N total",
            _human(nbytes),
        ])

    headers = ["Phase", "Added", "Cumulative", "Bytes (cumulative)"]
    print("Per-phase cumulative watermark")
    print(_try_tabulate(phase_rows, headers))

    print()
    cons = wm["conservative_peak"]
    typ  = wm["typical_peak"]
    peak_rows = [
        ["Typical peak (neg_div resolved before CG)",
         f"{typ:.0f} N", _human(typ * bytes_per_N)],
        ["Conservative peak (neg_div + CG transients overlap)",
         f"{cons:.0f} N", _human(cons * bytes_per_N)],
    ]
    headers2 = ["Scenario", "Multiplier", "Bytes"]
    print("Peak estimates")
    print(_try_tabulate(peak_rows, headers2))
    print()
    print("Note: does NOT include external A-operator (Radon/NUFFT) memory.")
    print("      Input arrays yT and x0 (2 N) are also not counted above.")
    print(f"      Add ~{_human(2 * bytes_per_N)} for those inputs.\n")


if __name__ == "__main__":
    main()
