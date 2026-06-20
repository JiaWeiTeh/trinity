#!/usr/bin/env python3
"""C0.2 certification residuals figure (PLAN.md S6.5 #4): res_beta(t) and
res_T0_struct(t) per config over the implicit phase.

Story: res_beta is high only in EARLY implicit and decays in time (finite-difference
TRUNCATION, not a defect), with localized spikes at the negative-beta re-pressurisation
surges; res_T0_struct (the solver's T-residual) is tight (~0.1%) everywhere.

    python plot_cert.py docs/dev/transition/cleanroom/data/c0_*_st6.csv
"""
from __future__ import annotations

import csv, glob, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[2] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False
WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]


def load(path, key):
    t, y = [], []
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            tn, v = float(r["t_now"]), float(r[key])
        except (ValueError, TypeError, KeyError):
            continue
        if v == v and v > 0:
            t.append(tn); y.append(v)
    return t, y


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_st6.csv")))
    if not paths:
        sys.exit("no CSVs")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    for i, p in enumerate(paths):
        name = Path(p).name.replace("c0_", "").replace("_st6.csv", "")
        c = WONG[i % len(WONG)]
        t, y = load(p, "res_beta")
        if t:
            a1.plot(t, y, color=c, lw=1.0, alpha=0.85, label=name)
        t, y = load(p, "res_T0_struct")
        if t:
            a2.plot(t, y, color=c, lw=1.0, alpha=0.85)
    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.axhline(0.05, ls="--", lw=1.0, color="0.5")  # 5% reference bar
    a1.set_ylabel(r"$res_\beta$  ($\beta\!\leftrightarrow\!dP_b/dt$, genuine)")
    a2.set_ylabel(r"$res_{T0}$  (solver $T$-residual)")
    a2.set_xlabel("time  [Myr]")
    a1.set_title("C0.2 substrate residuals: res_beta decays (truncation); res_T0 stays tight")
    a1.text(0.99, 0.055, "5% bar", transform=a1.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=7, color="0.5")
    a1.legend(fontsize=7, ncol=3, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"cert_residuals.{ext}", dpi=150)
    print(f"wrote {out}/cert_residuals.(pdf,png) from {len(paths)} configs")


if __name__ == "__main__":
    main()
