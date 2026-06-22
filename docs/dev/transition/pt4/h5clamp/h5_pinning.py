#!/usr/bin/env python3
"""H5 — per-epoch boundary-pinning of the LEGACY (beta,delta) on the committed
trajectories. Pure read of cleanroom/data/c0_<cfg>_{legacy,h0}.csv. No sims.

This is the rigorous core of the H5 causal argument that needs NO re-run: at every
implicit epoch of the committed LEGACY trajectory, is (beta,delta) sitting on the
box edge [0,1]x[-1,0] (get_betadelta.py:41-44)? The clamp can only hold Lloss high
(and force the 0.05 crossing) at epochs where it BINDS, i.e. where beta/delta is on
the boundary. Where beta is interior, the box is slack and cannot be causing the
crossing. The committed hybr (unbounded) trajectory is the counterfactual: its free
beta and the resulting ratio_min (does it ever cross 0.05?).

Writes:
  h5_pinning_summary.csv   one row per config: legacy crossing, pin fractions,
                           beta at/near the crossing, hybr ratio_min/crosses.
  h5_pinning_<cfg>.csv     per-epoch (t, legacy beta/delta/ratio, on_boundary).

    python h5_pinning.py
"""
from __future__ import annotations
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.normpath(os.path.join(HERE, "..", "..", "cleanroom", "data"))
TRIGGER = 0.05
EPS = 0.02  # within 2% of a box edge = "on the boundary"
BMIN, BMAX, DMIN, DMAX = 0.0, 1.0, -1.0, 0.0
CONFIGS = ["small_dense_highsfe", "simple_cluster", "midrange_pl0",
           "pl2_steep", "be_sphere", "large_diffuse_lowsfe"]


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            t = float(r["t_now"]); lg = float(r["bubble_Lgain"]); ll = float(r["bubble_Lloss"])
            b = float(r["cool_beta"]); d = float(r["cool_delta"])
        except (TypeError, ValueError, KeyError):
            continue
        if t > 0 and lg > 0 and lg == lg and ll == ll:
            out.append((t, b, d, (lg - ll) / lg))
    return out


def on_boundary(b, d):
    return (abs(b - BMIN) <= EPS or abs(b - BMAX) <= EPS
            or abs(d - DMIN) <= EPS or abs(d - DMAX) <= EPS)


def which_edge(b, d):
    e = []
    if abs(b - BMIN) <= EPS:
        e.append("b=0")
    if abs(b - BMAX) <= EPS:
        e.append("b=1")
    if abs(d - DMIN) <= EPS:
        e.append("d=-1")
    if abs(d - DMAX) <= EPS:
        e.append("d=0")
    return "+".join(e) if e else "interior"


def main():
    srows = []
    for cfg in CONFIGS:
        leg = load(os.path.join(CLEAN, f"c0_{cfg}_legacy.csv"))
        hyb = load(os.path.join(CLEAN, f"c0_{cfg}_h0.csv"))
        if not leg:
            continue
        ci = next((i for i, (t, b, d, ra) in enumerate(leg) if ra < TRIGGER), None)
        crosses = ci is not None
        cross_t = leg[ci][0] if crosses else None
        beta_x = leg[ci][1] if crosses else None
        delta_x = leg[ci][2] if crosses else None
        pin_all = sum(on_boundary(b, d) for _, b, d, _ in leg) / len(leg)
        pre = leg[:ci + 1] if crosses else leg
        pin_pre = sum(on_boundary(b, d) for _, b, d, _ in pre) / len(pre)
        hyb_rmin = min((ra for _, _, _, ra in hyb), default=float("nan")) if hyb else float("nan")
        hyb_cross = any(ra < TRIGGER for _, _, _, ra in hyb) if hyb else None
        hyb_bmax = max((b for _, b, _, _ in hyb), default=float("nan")) if hyb else float("nan")
        srows.append({
            "config": cfg, "legacy_crosses": crosses,
            "cross_t": ("" if cross_t is None else f"{cross_t:.5g}"),
            "beta_at_cross": ("" if beta_x is None else f"{beta_x:.3f}"),
            "delta_at_cross": ("" if delta_x is None else f"{delta_x:.3f}"),
            "edge_at_cross": (which_edge(beta_x, delta_x) if crosses else ""),
            "pin_frac_all": f"{pin_all:.3f}", "pin_frac_preX": f"{pin_pre:.3f}",
            "hybr_ratio_min": f"{hyb_rmin:.4f}", "hybr_crosses": hyb_cross,
            "hybr_beta_max": f"{hyb_bmax:.2f}", "n_implicit": len(leg)})
        # per-epoch detail
        with open(os.path.join(HERE, f"h5_pinning_{cfg}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_now", "legacy_beta", "legacy_delta", "legacy_ratio",
                        "on_boundary", "edge"])
            for t, b, d, ra in leg:
                w.writerow([t, b, d, ra, on_boundary(b, d), which_edge(b, d)])

    out = os.path.join(HERE, "h5_pinning_summary.csv")
    cols = ["config", "legacy_crosses", "cross_t", "beta_at_cross", "delta_at_cross",
            "edge_at_cross", "pin_frac_all", "pin_frac_preX", "hybr_ratio_min",
            "hybr_crosses", "hybr_beta_max", "n_implicit"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in srows:
            w.writerow(r)
    print(f"# wrote {out}\n")
    hdr = f"{'config':22}{'crosses':8}{'cross_t':>9}{'b@X':>6}{'d@X':>7}{'edge@X':>10}{'pinAll':>8}{'pinPreX':>9}{'hyb_rmin':>9}{'hyb_X':>7}"
    print(hdr)
    for r in srows:
        print(f"{r['config']:22}{str(r['legacy_crosses']):8}{r['cross_t']:>9}"
              f"{r['beta_at_cross']:>6}{r['delta_at_cross']:>7}{r['edge_at_cross']:>10}"
              f"{r['pin_frac_all']:>8}{r['pin_frac_preX']:>9}{r['hybr_ratio_min']:>9}"
              f"{str(r['hybr_crosses']):>7}")


if __name__ == "__main__":
    main()
