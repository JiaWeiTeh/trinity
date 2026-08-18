#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B11.A + B11.B — the photon-conserving fixed point, and the boundary/drive inconsistency.

Both are `shell_structure_pure` replays on committed run snapshots with one input
changed, so they share a harness. Gates are pre-registered in PLAN.md
(§Batch 11 → "Pre-registered gates for B11.A–D"); this script only measures them.

B11.A — photon-conserving fixed point
    A cavity Strömgren-filled at `n(x) = sqrt(3 x Qi / (4 pi chi_e alpha_B R2**3))`
    consumes exactly `x*Qi` for ANY x, so the cavity balance is one equation in two
    unknowns. The shipped code closes it with `x = f_abs(Qi)` — the shell's absorbed
    fraction computed from the UNDEPLETED flux, which is the double-spend. The
    photon-conserving closure of the same scheme is the fixed point

        x = f_abs(Qi * (1 - x)) ,   f_abs(Q) = shell_structure_pure with params['Qi'] = Q

    G11.A1 wants the root structure, so `g(x) = f_abs(Qi(1-x)) - x` is evaluated on a
    grid first (that is the deliverable) and bisected only where the grid brackets a
    sign change. G11.A2 wants `P_C3a_fixedpoint / P_C3a_shipped = sqrt(x* / f_abs(Qi))`.

B11.B — boundary/drive inconsistency
    Re-run the shell at the drive's claimed inner pressure (`P_HII`) instead of the
    shipped `params['Pb']`, and report the SPREAD (Δf_abs, ΔdR_ion, Δdust, Δshell_n0,
    ΔP_C3a). Re-scoped by B11.0: `nShell0 ∝ Pb` is the standard closure, so this sizes
    the inconsistency rather than proposing a correction.

    python docs/dev/phii-identity/harness/photon_ledger.py <run_dir> [...] \
        --stride 2 --out docs/dev/phii-identity/data/b11_photon_ledger.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402
from trinity.shell_structure.shell_structure import shell_structure_pure  # noqa: E402

from _stamp import stamp  # noqa: E402

REPLAY_KEYS = ("bubble_mass", "Pb", "R2", "shell_mass", "Qi", "Li", "Ln", "rShell")

# G11.A1 asks for the shape of g, not just a root, so the grid IS a deliverable.
X_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999)

FIELDS = [
    "run", "row_idx", "phase", "t", "status", "R2", "Qi", "Pb", "P_HII",
    # B11.A
    "f_abs_shipped", "g_grid", "n_roots", "x_star", "x_star_kind",
    "ratio_A_fixedpoint_over_shipped",
    # B11.B
    "f_abs_at_PC3a", "d_f_abs", "dR_ion_Pb", "dR_ion_PC3a", "d_dR_ion_frac",
    "dust_Pb", "dust_PC3a", "shell_n0_Pb", "shell_n0_PC3a", "shell_n0_ratio",
    "P_C3a_at_PC3a_boundary", "d_P_C3a_frac",
]


def solve_shell(params, row, qi=None, pb=None):
    """Replay the shipped shell solve on this row's state, optionally overriding Qi / Pb."""
    for key in REPLAY_KEYS:
        params[key].value = row[key]
    if qi is not None:
        params["Qi"].value = qi
    if pb is not None:
        params["Pb"].value = pb
    return shell_structure_pure(params)


def f_abs_of(params, row, qi):
    """f_abs(Q). Q = 0 has no photons to escape, so f_abs is 1 by definition, not by solve."""
    if qi <= 0.0:
        return 1.0
    try:
        return float(getattr(solve_shell(params, row, qi=qi), "shell_fAbsorbedIon", 1.0))
    except Exception:
        return None


def fixed_point(params, row, Qi):
    """G11.A1: grid the residual g(x) = f_abs(Qi(1-x)) - x, then bisect any bracketed root."""
    grid = []
    for x in X_GRID:
        fa = f_abs_of(params, row, Qi * (1.0 - x))
        grid.append(None if fa is None else fa - x)
    if any(v is None for v in grid):
        return grid, None, None, None

    roots = [
        (X_GRID[i], X_GRID[i + 1])
        for i in range(len(X_GRID) - 1)
        if grid[i] == 0.0 or grid[i] * grid[i + 1] < 0.0
    ]
    if not roots:
        # g > 0 across [0, 1) with g(1) = 0 exactly: the cavity takes every photon.
        return grid, 0 if grid[-1] < 0 else 1, 1.0, "endpoint_x=1"

    lo, hi = roots[0]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        gm = f_abs_of(params, row, Qi * (1.0 - mid))
        if gm is None:
            return grid, len(roots), None, "bisect_failed"
        gm -= mid
        glo = f_abs_of(params, row, Qi * (1.0 - lo)) - lo
        if gm == 0.0:
            return grid, len(roots), mid, "interior"
        lo, hi = (mid, hi) if gm * glo > 0 else (lo, mid)
    return grid, len(roots), 0.5 * (lo + hi), "interior"


def replay(run_dir, stride):
    run_dir = Path(run_dir)
    pfile = next(run_dir.glob("*.param"), None)
    if pfile is None:
        sys.exit(f"no .param in {run_dir} — need the run's own materialised config")
    params = read_param(str(pfile))
    chi_e = params["chi_e_shell"].value
    alpha_B = params["caseB_alpha"].value
    pref = (params["mu_convert"].value / params["mu_ion_shell"].value) * params["k_B"].value * params[
        "TShell_ion"
    ].value

    def p_c3a(R2, Qi, f_abs):
        denom = 4.0 * math.pi * chi_e * alpha_B * R2**3
        if not (denom > 0 and Qi * f_abs > 0):
            return 0.0
        return pref * math.sqrt(3.0 * Qi * f_abs / denom)

    out = []
    for k, ln in enumerate(l for l in (run_dir / "dictionary.jsonl").open() if l.strip()):
        if k % stride:
            continue
        try:
            row = json.loads(ln)
        except ValueError:
            continue
        if any(row.get(key) is None for key in REPLAY_KEYS):
            continue

        rec = dict(
            run=run_dir.name, row_idx=k, phase=row.get("current_phase"),
            t=row.get("t_now"), status="ok",
            R2=float(row["R2"]), Qi=float(row["Qi"]),
            Pb=float(row.get("Pb") or 0.0), P_HII=float(row.get("P_HII") or 0.0),
        )
        R2, Qi, P_HII = rec["R2"], rec["Qi"], rec["P_HII"]

        try:
            sp = solve_shell(params, row)
        except Exception as exc:
            rec["status"] = f"replay_failed:{type(exc).__name__}"
            out.append(rec)
            continue
        f0 = float(getattr(sp, "shell_fAbsorbedIon", 1.0))
        rec["f_abs_shipped"] = f0

        # ---- B11.A ----
        grid, n_roots, x_star, kind = fixed_point(params, row, Qi)
        rec["g_grid"] = "|".join("nan" if v is None else f"{v:.6g}" for v in grid)
        rec["n_roots"], rec["x_star"], rec["x_star_kind"] = n_roots, x_star, kind
        if x_star is not None and f0 > 0:
            rec["ratio_A_fixedpoint_over_shipped"] = math.sqrt(x_star / f0)

        # ---- B11.B ---- only meaningful where the drive actually claims a pressure
        if P_HII > 0:
            try:
                sp_b = solve_shell(params, row, pb=P_HII)
            except Exception as exc:
                rec["status"] = f"B_replay_failed:{type(exc).__name__}"
                out.append(rec)
                continue
            f_b = float(getattr(sp_b, "shell_fAbsorbedIon", 1.0))
            r0 = _ion_thickness(sp)
            r1 = _ion_thickness(sp_b)
            rec.update(
                f_abs_at_PC3a=f_b,
                d_f_abs=f_b - f0,
                dR_ion_Pb=r0,
                dR_ion_PC3a=r1,
                d_dR_ion_frac=((r1 - r0) / r0) if (r0 and r0 > 0) else None,
                dust_Pb=getattr(sp, "shell_fIonisedDust", None),
                dust_PC3a=getattr(sp_b, "shell_fIonisedDust", None),
                shell_n0_Pb=getattr(sp, "shell_n0", None),
                shell_n0_PC3a=getattr(sp_b, "shell_n0", None),
                P_C3a_at_PC3a_boundary=p_c3a(R2, Qi, f_b),
                d_P_C3a_frac=(p_c3a(R2, Qi, f_b) - P_HII) / P_HII,
            )
            n0a, n0b = rec["shell_n0_Pb"], rec["shell_n0_PC3a"]
            if n0a:
                rec["shell_n0_ratio"] = n0b / n0a
        out.append(rec)
    return out


def _ion_thickness(sp):
    r = getattr(sp, "shell_r_arr", None)
    i = int(getattr(sp, "shell_ion_idx", -1))
    if r is None or len(r) < 2 or i < 1:
        return None
    i = min(i, len(r) - 1)
    return float(r[i] - r[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [r for run in args.runs for r in replay(run, args.stride)]
    ok = [r for r in rows if r["status"] == "ok"]
    drive = [r for r in ok if r["P_HII"] > 0]
    conf = [r for r in ok if r["P_HII"] == 0]
    if not drive:
        sys.exit("no driving rows replayed — B11.A/B are VOID, not a null")
    print(f"{len(ok)} rows replayed, {len(drive)} driving, {len(conf)} confined\n")

    print("=== B11.A — photon-conserving fixed point ===")
    print("G11.A1 root structure, driving rows:")
    kinds = {}
    for r in drive:
        kinds[r["x_star_kind"]] = kinds.get(r["x_star_kind"], 0) + 1
    for kind, n in sorted(kinds.items()):
        print(f"   {kind:20s} {n:3d} rows")
    interior = [r for r in drive if r["x_star"] is not None and r["x_star"] < 0.999]
    print(f"   interior roots x* < 0.999: {len(interior)}  "
          f"(FALSIFIES the degeneracy reading if > 0)")

    ratios = [r["ratio_A_fixedpoint_over_shipped"] for r in drive
              if r.get("ratio_A_fixedpoint_over_shipped")]
    if ratios:
        below = sum(1 for x in ratios if x < 1.0)
        print(f"\nG11.A2 P_C3a_fixedpoint / P_C3a_shipped: {min(ratios):.4f}..{max(ratios):.4f}")
        print(f"   rows < 1: {below}/{len(ratios)}   "
              f"(§6b seam A predicts < 1; >= 1 throughout strikes that clause)")

    bad_null = [r for r in conf if r.get("x_star") is not None and r["P_HII"] != 0]
    print(f"\nG11.A3 confined-branch null: {len(bad_null)} violations "
          f"over {len(conf)} confined rows (any > 0 FAILS)")

    print("\n=== B11.B — boundary/drive inconsistency ===")
    sat = [r for r in drive if r["f_abs_shipped"] >= 1.0]
    viol1 = [r for r in sat if r.get("d_f_abs") not in (None,) and r["d_f_abs"] != 0.0]
    print(f"G11.B1 Δf_abs on the {len(sat)} saturated rows: {len(viol1)} non-zero "
          f"(any > 0 FALSIFIES B11.0's revision)")
    dp = [r["d_P_C3a_frac"] for r in drive if r.get("d_P_C3a_frac") is not None]
    if dp:
        neg = sum(1 for x in dp if x < 0)
        print(f"G11.B2 ΔP_C3a/P_C3a: {min(dp):+.4f}..{max(dp):+.4f}   "
              f"rows < 0: {neg}/{len(dp)}   (any < 0 FALSIFIES B11.0's revision)")
    print("G11.B3 size of the inconsistency (descriptive):")
    print(f"   {'phase':11s}{'n':>4}{'shell_n0 ratio':>16}{'ΔdR_ion':>11}{'dust Pb→P_C3a':>20}")
    for phase in ("transition", "momentum"):
        v = [r for r in drive if r["phase"] == phase and r.get("shell_n0_ratio")]
        if not v:
            continue
        med = lambda key: sorted(x[key] for x in v if x.get(key) is not None)[len(v) // 2]  # noqa: E731
        d0, d1 = med("dust_Pb"), med("dust_PC3a")
        print(f"   {phase:11s}{len(v):>4d}{med('shell_n0_ratio'):>16.3f}"
              f"{med('d_dR_ion_frac'):>+11.4f}{f'{d0:.3f}→{d1:.3f}':>20}")

    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        print(f"\n{len(bad)} row(s) not replayed: {sorted({r['status'] for r in bad})}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            fh.write(f"# g_grid is g(x)=f_abs(Qi(1-x))-x at x={list(X_GRID)}\n")
            for run in args.runs:
                fh.write(f"# run {run} (stride {args.stride})\n")
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
