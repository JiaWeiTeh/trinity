#!/usr/bin/env python3
"""H5 clamp-width sweep figures. Pure reads of h5_sweep.csv + the per-cell
trajectory CSVs in data/ + the committed cleanroom c0_*_{legacy,h0}.csv. No sims
re-run. Saves PDF+PNG (h5_ prefix) into figures/.

    python h5_figures.py

Figures:
  h5_crossing_vs_boxwidth - per config, cross_t (and ratio_min) vs box width
                            (W0->W1->W2->W3->hybr). The headline trend family.
  h5_ratio_trajectories   - cooling ratio(t) for a PINNED config (small_dense) and
                            a NON-PINNED config (simple_cluster) across box widths,
                            with the 0.05 line — does widening move the crossing?
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.normpath(os.path.join(HERE, "..", "..", "cleanroom", "data"))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "figures")
# h5clamp -> pt4 -> transition -> dev -> docs -> repo root  (5 ups)
STYLE = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "paper", "_lib", "trinity.mplstyle"))
if os.path.exists(STYLE):
    plt.style.use(STYLE)
plt.rcParams["text.usetex"] = False

WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
TRIGGER = 0.05
CONFIGS = ["small_dense_highsfe", "simple_cluster", "midrange_pl0",
           "pl2_steep", "be_sphere", "large_diffuse_lowsfe"]
WIDTHS = ["W0", "W1", "W2", "W3", "hybr"]
XPOS = {w: i for i, w in enumerate(WIDTHS)}


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


def load_sweep():
    rows = list(csv.DictReader(open(os.path.join(HERE, "h5_sweep.csv"))))
    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[r["box_width"]] = r
    return by


def _f(r, k):
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return None


def crossing_vs_boxwidth(by):
    fig, (axc, axr) = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, cfg in enumerate(CONFIGS):
        cells = by.get(cfg, {})
        xs_c, ys_c, xs_nc = [], [], []
        xs_rm, ys_rm = [], []
        for w in WIDTHS:
            r = cells.get(w)
            if not r:
                continue
            x = XPOS[w]
            crosses = str(r.get("crosses", "")).lower() == "true"
            ct, rm = _f(r, "cross_t"), _f(r, "ratio_min")
            if crosses and ct is not None:
                xs_c.append(x); ys_c.append(ct)
            elif rm is not None:
                xs_nc.append(x)  # no crossing at this width (ratio recovered)
            if rm is not None:
                xs_rm.append(x); ys_rm.append(rm)
        c = WONG[i % len(WONG)]
        if xs_c:
            axc.plot(xs_c, ys_c, "-o", color=c, label=cfg, ms=6)
        # open markers along the x-axis (y=0) where there is NO crossing
        if xs_nc:
            axc.plot(xs_nc, [0.0] * len(xs_nc), "o", mfc="none", mec=c, ms=7)
        if xs_rm:
            axr.plot(xs_rm, ys_rm, "-o", color=c, label=cfg, ms=6)
    axc.set_xticks(list(XPOS.values())); axc.set_xticklabels(WIDTHS)
    axc.set_xlabel("box width (legacy) -> hybr ref"); axc.set_ylabel("0.05 crossing time [Myr]")
    axc.set_title("crossing time vs box width\n(missing point = no crossing)")
    axc.grid(True, alpha=0.3)
    axr.axhline(TRIGGER, color="k", ls="--", lw=1, label="0.05 trigger")
    axr.set_xticks(list(XPOS.values())); axr.set_xticklabels(WIDTHS)
    axr.set_xlabel("box width (legacy) -> hybr ref"); axr.set_ylabel("min cooling ratio")
    axr.set_title("ratio_min vs box width")
    axr.grid(True, alpha=0.3)
    axr.legend(fontsize=7, ncol=1, loc="best")
    fig.suptitle("H5: does widening the legacy (beta,delta) box move/vanish the 0.05 crossing?",
                 y=1.04)
    fig.tight_layout()
    _save(fig, "h5_crossing_vs_boxwidth")


def _load_traj_legacy_box(cfg, w):
    """ratio(t) for a legacy box width. W0 from committed c0_legacy; W1-3 from data/."""
    if w == "W0":
        path = os.path.join(CLEAN, f"c0_{cfg}_legacy.csv")
        ts, rs = [], []
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                try:
                    t = float(r["t_now"]); Lg = float(r["bubble_Lgain"]); Ll = float(r["bubble_Lloss"])
                except (TypeError, ValueError, KeyError):
                    continue
                if t > 0 and Lg > 0 and Lg == Lg and Ll == Ll:
                    ts.append(t); rs.append((Lg - Ll) / Lg)
        return ts, rs
    if w == "hybr":
        path = os.path.join(CLEAN, f"c0_{cfg}_h0.csv")
        ts, rs = [], []
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                try:
                    t = float(r["t_now"]); Lg = float(r["bubble_Lgain"]); Ll = float(r["bubble_Lloss"])
                except (TypeError, ValueError, KeyError):
                    continue
                if t > 0 and Lg > 0 and Lg == Lg and Ll == Ll:
                    ts.append(t); rs.append((Lg - Ll) / Lg)
        return ts, rs
    # W1/W2/W3: capture/replay per-epoch counterfactual ratio for this box width
    # (replay covers the crossing region; capture covers early segments)
    pts = {}
    for fname in (f"h5_capture_{cfg}.csv", f"h5_replay_{cfg}.csv"):
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            if r.get("width") != w:
                continue
            try:
                t = float(r["t_now"]); ra = float(r["ratio"])
            except (TypeError, ValueError, KeyError):
                continue
            if t > 0:
                pts[round(t, 7)] = (t, ra)  # replay (read 2nd) overwrites capture
    seq = sorted(pts.values())
    return [t for t, _ in seq], [ra for _, ra in seq]


def ratio_trajectories(by):
    pair = [("small_dense_highsfe", "PINNED (pin_frac 0.97)"),
            ("simple_cluster", "NON-PINNED (pin_frac 0.00)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    colors = {"W0": WONG[0], "W1": WONG[1], "W2": WONG[2], "W3": WONG[4], "hybr": "0.35"}
    for ax, (cfg, tag) in zip(axes, pair):
        for w in WIDTHS:
            ts, rs = _load_traj_legacy_box(cfg, w)
            if not ts:
                continue
            ls = "--" if w == "hybr" else "-"
            ax.plot(ts, rs, ls, color=colors[w], lw=1.6, label=w, alpha=0.9)
        ax.axhline(TRIGGER, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("t [Myr]")
        ax.set_title(f"{cfg}\n{tag}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, title="box width")
    axes[0].set_ylabel("cooling ratio (Lgain-Lloss)/Lgain")
    fig.suptitle("H5: cooling ratio(t) across legacy box widths (+ hybr ref), 0.05 line",
                 y=1.04)
    fig.tight_layout()
    _save(fig, "h5_ratio_trajectories")


def main():
    by = load_sweep()
    crossing_vs_boxwidth(by)
    ratio_trajectories(by)


if __name__ == "__main__":
    main()
