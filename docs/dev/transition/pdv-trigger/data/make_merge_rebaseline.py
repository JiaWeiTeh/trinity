#!/usr/bin/env python3
"""Θ₀ re-baseline after the 2026-08-08 `main` merge, plus the Lancaster Eq-10 screen.

Two tables, one pass over the same three runs, because both read the same
`dictionary.jsonl` and re-running the sims to answer them separately is the exact
waste `docs/dev/CLAUDE.md`'s 💾 rule exists to prevent.

  table=THETA0  the CODE BASELINE check (kappa-3way/PROVENANCE.md §1, §4a).
                Merging `main` deleted the `vd = -1e8` early-phase override and made
                phase-1a segments age-scaled, both of which move full-run trajectories.
                Every full-run number in the campaign was measured at 1056c6d, before
                both. This re-runs the three Θ₀ arms at the merge and scores them
                against the committed values at G0's own bar (abs 5e-4 = half the last
                digit the pre-registration quotes).

                Reported twice per arm, because the implicit window itself can move:
                  native   — each run over its own window (what the campaign quotes)
                  matched  — both truncated to the shorter window, endpoint INTERPOLATED
                             rather than row-dropped (a bare `t <= tmax` cut silently
                             deletes one endpoint row and manufactures a disagreement)

  table=EQ10    the f_area Option-3 screen (kappa-3way/F_AREA_PLAN §5a item 3).
                Lancaster+2021a Eq 10 is a CLOSED-FORM, ℓ-free Θ prediction:
                    1 − Θ = ( ½(1+f_turb)·α_p/α_R + S ) · (Ṙ_b/V_w),   S ≈ α_p (Eq 6)
                Every constant in the prefactor is order-unity — α_p ~ 1.2–4 measured
                (F_KAPPA_FUNCTIONAL_FORM.md:139), α_R ~ 1, S ≈ α_p — so Lancaster's C
                sits in roughly [2, 12]. That makes the prefactor a MEASUREMENT here:
                invert Eq 10 on TRINITY's own trajectory and ask whether the implied C
                is order-unity. No truncation scale, no fractal area, no fitted
                constant — which is the whole reason Option 3 outlived Option 2
                (FINDINGS §15c).

                ⚠️ Mapping caveats, stated not buried: Lancaster's R_b is the hot-gas
                bubble radius and TRINITY's R2 is the contact/shell radius; Lancaster's
                Θ = L_int/Ė_in and TRINITY's θ = bubble_Lloss/Lmech_total
                (LANCASTER_REFERENCE.md §7b treats these as the comparable pair). V_w is
                `v_mech_total` = 2·L_mech/ṗ, which is Lancaster's Eq-1 definition.

Usage (from the repo root):
    python docs/dev/transition/pdv-trigger/data/make_merge_rebaseline.py

Reads   outputs/bench5/{bench1_m5e4_r20,bench2_m1e5_r10,bench3_m1e5_r5}__none_diag/
        dictionary.jsonl                                    (re-run at the merge)
        docs/dev/transition/pdv-trigger/runs/data/bench5r_traj/*__none_diag.csv
                                                            (the committed 1056c6d record)
Writes  docs/dev/transition/pdv-trigger/data/merge_rebaseline.csv
        docs/dev/transition/pdv-trigger/merge_rebaseline.png

Regenerate the three inputs (separate processes — trinity leaks module globals):
    for b in bench1_m5e4_r20 bench2_m1e5_r10 bench3_m1e5_r5; do
      python run.py docs/dev/transition/pdv-trigger/runs/params/bench5/${b}__none_diag.param
    done
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _stamp import stamp  # noqa: E402

REPO = HERE.parents[4]
OLD_TRAJ = HERE.parent / "runs" / "data" / "bench5r_traj"
OUT_CSV = HERE / "merge_rebaseline.csv"
OUT_PNG = HERE.parent / "merge_rebaseline.png"

# Committed Θ₀, measured at code 1056c6d, stamped 2026-07-30 (bench5r). These are also the
# G0 pre-registration targets (FINDINGS §1); the bar is G0's own.
REF = {
    "bench1_m5e4_r20": 0.220551,
    "bench2_m1e5_r10": 0.340860,
    "bench3_m1e5_r5": 0.461806,
}
BAR = 5e-4  # absolute, = half the last digit the pre-registration quotes

# Lancaster Eq-10 prefactor bracket from the measured order-unity constants:
#   C = ½(1+f_turb)·α_p/α_R + S,  with α_p ∈ [1.2, 4] (F_KAPPA_FUNCTIONAL_FORM.md:139),
#   α_R ~ 1, S ≈ α_p within 6% (Eq 6), f_turb ∈ [0, 3].
# Low corner: α_p=1.2, f_turb=0 → ½·1·1.2 + 1.2 = 1.8. High: α_p=4, f_turb=3 → 8 + 4 = 12.
C_LO, C_HI = 1.8, 12.0


def _fin(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def new_traj(run_dir):
    """Accepted implicit rows: (t, theta, Lmech, Rdot_over_Vw). Same filter as
    runs/harvest_bench5.py::trajectory, plus the two Eq-10 columns."""
    out = []
    with (run_dir / "dictionary.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("current_phase") != "implicit":
                continue
            t = _fin(d.get("t_now"))
            Lloss = _fin(d.get("bubble_Lloss"))
            if Lloss is None:
                Lloss = _fin(d.get("bubble_LTotal"))
            Lmech = _fin(d.get("Lmech_total"))
            if t is None or Lloss is None or not Lmech:
                continue
            v2, vw = _fin(d.get("v2")), _fin(d.get("v_mech_total"))
            out.append((t, Lloss / Lmech, Lmech, (v2 / vw) if (v2 is not None and vw) else None))
    out.sort(key=lambda r: r[0])
    return out


def old_traj(arm):
    """The committed 1056c6d trajectory. Carries no v2/V_w, so Eq-10 is new-run only."""
    p = OLD_TRAJ / f"{arm}__none_diag.csv"
    lines = [ln for ln in p.open() if not ln.startswith("#")]
    return [(float(r["t_now"]), float(r["theta"]), float(r["Lmech"]), None)
            for r in csv.DictReader(lines)
            if r["t_now"] and r["theta"] and r["Lmech"] and float(r["Lmech"])]


def _clip(rows, tmax):
    """Rows up to tmax, with the final point INTERPOLATED onto tmax exactly.

    A bare `t <= tmax` filter drops the endpoint row whenever the two windows differ by
    less than one step, which fabricates a Θ difference (observed on bench2: native
    6.6e-5 PASS became a spurious 6.8e-3 FAIL). Interpolating keeps the windows genuinely
    matched.
    """
    if tmax is None or not rows:
        return rows
    kept = [r for r in rows if r[0] <= tmax]
    if len(kept) == len(rows):
        return kept
    if not kept:
        return kept
    a, b = kept[-1], rows[len(kept)]
    span = b[0] - a[0]
    if span <= 0:
        return kept
    w = (tmax - a[0]) / span
    lerp = lambda x, y: None if (x is None or y is None) else x + w * (y - x)  # noqa: E731
    return kept + [(tmax, lerp(a[1], b[1]), lerp(a[2], b[2]), lerp(a[3], b[3]))]


def theta_cum(rows, tmax=None):
    """L_mech-weighted trapezoid mean of θ — the Θ_cum metric
    (data/make_bench5_analysis.py::theta_cum_prefire, duplicated in harvest_bench5.py)."""
    rows = _clip(rows, tmax)
    num = den = 0.0
    for (t0, h0, m0, _), (t1, h1, m1, _) in zip(rows, rows[1:]):
        dt = t1 - t0
        num += 0.5 * (h0 * m0 + h1 * m1) * dt
        den += 0.5 * (m0 + m1) * dt
    return (num / den) if den else None


def main():
    rows_out = []
    figdata = {}
    print("=== THETA0 — CODE BASELINE re-check at the merge ===")
    print(f"{'arm':20s} {'window':>8s} {'committed':>10s} {'re-run':>10s} {'abs diff':>10s} {'verdict':>8s}")
    n_pass = n_fail = 0
    for arm, ref in REF.items():
        d = REPO / "outputs" / "bench5" / f"{arm}__none_diag"
        if not (d / "dictionary.jsonl").exists():
            print(f"{arm:20s}  MISSING {d} — run it first (see module docstring)")
            return 2
        new, old = new_traj(d), old_traj(arm)
        te_n, te_o = new[-1][0], old[-1][0]
        tm = min(te_n, te_o)
        for window, tmax in (("native", None), ("matched", tm)):
            t_new = theta_cum(new, tmax)
            t_old = theta_cum(old, tmax)
            # 'native' scores the re-run against the COMMITTED published value; 'matched'
            # scores the two trajectories against each other on a common window.
            got, want = (t_new, ref) if window == "native" else (t_new, t_old)
            diff = abs(got - want)
            ok = diff < BAR
            n_pass, n_fail = (n_pass + ok, n_fail + (not ok))
            print(f"{arm:20s} {window:>8s} {want:10.6f} {got:10.6f} {diff:10.2e} {'PASS' if ok else 'FAIL':>8s}")
            rows_out.append({
                "table": "THETA0", "arm": arm, "window": window,
                "theta_committed": f"{want:.9g}", "theta_rerun": f"{got:.9g}",
                "abs_diff": f"{diff:.6g}", "bar": BAR,
                "t_end_committed": f"{te_o:.6g}", "t_end_rerun": f"{te_n:.6g}",
                "n_rows_committed": len(old), "n_rows_rerun": len(new),
                "verdict": "PASS" if ok else "FAIL",
            })
        figdata[arm] = (new, old)

    print(f"\nTHETA0: {n_pass} PASS / {n_fail} FAIL   (bar = {BAR} absolute, G0's own)")

    print("\n=== EQ10 — Lancaster Eq-10 screen (f_area Option 3) ===")
    print(f"Lancaster prefactor bracket from order-unity constants: C in [{C_LO}, {C_HI}]")
    print(f"{'arm':20s} {'C med':>9s} {'C min':>9s} {'C max':>9s} {'drift':>7s} {'Th_pred(hi C)':>14s} {'Th_meas':>9s} {'verdict':>8s}")
    for arm in REF:
        new = figdata[arm][0]
        pts = [(t, th, rv) for t, th, _, rv in new if rv and 0 < th < 1]
        if len(pts) < 2:
            continue
        Cs = sorted((1 - th) / rv for _, th, rv in pts)
        med = Cs[len(Cs) // 2]
        drift = Cs[-1] / Cs[0] if Cs[0] > 0 else float("nan")
        # What Θ would Eq 10 predict at TRINITY's own Rdot/Vw, at the generous end of
        # Lancaster's bracket? Evaluated at the LAST implicit row (the operating point
        # closest to the transition the trigger actually fires on).
        t_l, th_l, rv_l = pts[-1]
        th_pred = 1 - C_HI * rv_l
        ok = C_LO <= med <= C_HI
        print(f"{arm:20s} {med:9.2f} {Cs[0]:9.2f} {Cs[-1]:9.2f} {drift:7.1f}x {th_pred:14.4f} {th_l:9.4f} "
              f"{'PASS' if ok else 'FAIL':>8s}")
        rows_out.append({
            "table": "EQ10", "arm": arm, "window": "implicit",
            "C_median": f"{med:.6g}", "C_min": f"{Cs[0]:.6g}", "C_max": f"{Cs[-1]:.6g}",
            "C_drift": f"{drift:.4g}", "C_lo_lancaster": C_LO, "C_hi_lancaster": C_HI,
            "t_last": f"{t_l:.6g}", "Rdot_over_Vw_last": f"{rv_l:.6g}",
            "theta_pred_at_C_hi": f"{th_pred:.6g}", "theta_measured_last": f"{th_l:.6g}",
            "verdict": "PASS" if ok else "FAIL",
        })

    cols = sorted({k for r in rows_out for k in r},
                  key=lambda k: (k != "table", k != "arm", k != "window", k))
    with OUT_CSV.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Theta0 re-baseline after the 2026-08-08 main merge (kappa-3way PROVENANCE 4a,"
                 " FINDINGS 14) + the Lancaster Eq-10 screen (F_AREA_PLAN 5a item 3, FINDINGS 15c).\n")
        fh.write("# table=THETA0: window=native scores the re-run against the COMMITTED value;"
                 " window=matched scores the two trajectories on a common window (endpoint interpolated).\n")
        fh.write("# table=EQ10: C = (1-Theta)/(Rdot/Vw), the inverted Eq-10 prefactor. Lancaster's is"
                 " order-unity; PASS means TRINITY's implied C lands inside his bracket.\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {OUT_CSV}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for j, arm in enumerate(REF):
        new, old = figdata[arm]
        ax = axes[0][j]
        ax.plot([r[0] for r in old], [r[1] for r in old], lw=2.4, alpha=.55, label="committed (1056c6d)")
        ax.plot([r[0] for r in new], [r[1] for r in new], lw=1.3, ls="--", label="re-run (merge 3c090b7)")
        ax.set_title(f"{arm}\nθ(t), implicit phase", fontsize=9)
        ax.set_xlabel("t [Myr]"); ax.set_ylabel(r"$\theta$"); ax.set_xscale("log")
        ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax = axes[1][j]
        pts = [(t, th, rv) for t, th, _, rv in new if rv and 0 < th < 1]
        if pts:
            ax.plot([p[0] for p in pts], [(1 - p[1]) / p[2] for p in pts], lw=1.6, color="crimson")
            ax.axhspan(C_LO, C_HI, color="tab:green", alpha=.22,
                       label=f"Lancaster C ∈ [{C_LO}, {C_HI}]")
            ax.set_yscale("log")
        ax.set_title("Eq-10 implied prefactor  C = (1−Θ)/(Ṙ/V_w)", fontsize=9)
        ax.set_xlabel("t [Myr]"); ax.set_ylabel("C"); ax.set_xscale("log")
        ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle("Θ₀ re-baseline at the main merge (top) · Lancaster Eq-10 screen (bottom)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=135)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
