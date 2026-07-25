#!/usr/bin/env python3
"""SC-0 figures — the three-candidate f_A screen (FA_STATE_COUPLED.md SC-0).

Pure read of committed CSVs (data/fa_state_screen.csv + data/zone_resolution.csv); runs no sims.
Repo figure convention: Agg backend, text.usetex=False, dpi~135, committed alongside the CSV.

    python docs/dev/transition/pdv-trigger/data/make_fa_state_figures.py
Deliverable: fa_state_screen.png (3 panels).
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PDV = HERE.parent
plt.rcParams.update({"text.usetex": False, "font.size": 9})

C1KEY, C2KEY, C3KEY = "C1_ldv3", "C2_d0.7", "C3_fitted"
COL = {"C1": "#1f6feb", "C2": "#c1121f", "C3": "#6a4c93"}


def _read(p):
    with open(p) as fh:
        return list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def main():
    rows = [r for r in _read(HERE / "fa_state_screen.csv") if _f(r.get("target"))]
    if not rows:
        raise SystemExit("no scored arms in fa_state_screen.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # ---- Panel A: predicted vs measured (the money plot) ----------------------------------
    ax = axes[0]
    for key, lab, c in ((C1KEY, "C1 El-Badry", COL["C1"]),
                        (C2KEY, "C2 Lancaster Eq-11", COL["C2"]),
                        (C3KEY, "C3 fitted scalar", COL["C3"])):
        xs = [_f(r["target"]) for r in rows if _f(r.get(key))]
        ys = [_f(r[key]) for r in rows if _f(r.get(key))]
        if xs:
            ax.scatter(xs, ys, s=42, c=c, label=lab, alpha=0.85, edgecolor="k", linewidth=0.4)
    lim = [0.5, 1e10]
    ax.plot(lim, lim, "k--", lw=1, label="perfect (1:1)")
    ax.fill_between(lim, [x / 2 for x in lim], [x * 2 for x in lim], color="k", alpha=0.07,
                    label="within 2x")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.5, 200); ax.set_ylim(0.02, 1e10)
    ax.set_xlabel("MEASURED f_A (band-entry dose / fire threshold)")
    ax.set_ylabel("PREDICTED f_A (window mean)")
    ax.set_title("A. Predicted vs measured\n(C2 is 3-8 dex high: the falsification)")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.25, which="both")

    # ---- Panel B: spread of ratio per candidate (the discriminator) ------------------------
    # Split by target_kind: `band` (Theta_cum-in-band dose) and `fire` (theta_max threshold) are
    # DIFFERENT criteria, so a combined spread double-counts the ~10x offset between them.
    ax = axes[1]
    keys = [k for k in rows[0] if k.startswith(("C1_", "C2_", "C3"))]
    bars = []
    for k in keys:
        per = {}
        for kind in ("band", "fire"):
            rs = [_f(r[k]) / _f(r["target"]) for r in rows
                  if _f(r.get(k)) and r.get("target_kind") == kind]
            if len(rs) >= 3:
                per[kind] = max(rs) / min(rs)
        if per:
            bars.append((max(per.values()), k, per))
    bars.sort()
    ys = list(range(len(bars)))
    for i, (_worst, k, per) in enumerate(bars):
        for j, kind in enumerate(("band", "fire")):
            if kind in per:
                ax.barh(i + (j - 0.5) * 0.36, per[kind], height=0.34, color=COL[k[:2]],
                        alpha=0.85 if kind == "fire" else 0.45, edgecolor="k", linewidth=0.4,
                        hatch="" if kind == "fire" else "//")
    ax.set_yticks(ys); ax.set_yticklabels([b[1] for b in bars], fontsize=7)
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.axvline(2, color="0.4", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("spread of (predicted/measured) WITHIN a target type  [1 = perfect shape]")
    ax.set_title("B. Shape test (calibration-invariant)\nsolid = fire (n=9), hatched = band (n=3)")
    ax.grid(alpha=0.25, axis="x", which="both")

    # ---- Panel C: why C2 dies -- l_cool vs every real scale --------------------------------
    ax = axes[2]
    # l_cool(p) for the representative bench3 row (SC-0 diagnostic, LANCASTER_REFERENCE 7c)
    ps = [0.0, 0.1, 0.3, 0.5, 0.7]
    lcool = [2.912e-07, 4.236e-08, 1.716e-10, 8.482e-15, 7.600e-25]
    ax.plot(ps, lcool, "o-", color=COL["C2"], label=r"$\ell_{cool}$ from Eq 13 (any p)")
    for y, lab, c in ((0.02, "L21b grid $\\Delta x$ (0.02 pc)", "#2a5d8f"),
                      (5e-7, "conduction front (~5e-7 pc)", "#2a9d3f"),
                      (0.116, "$\\ell$ the measured dose DEMANDS", "#e07a00")):
        ax.axhline(y, color=c, ls="--", lw=1.2, label=lab)
    ax.set_yscale("log"); ax.set_ylim(1e-27, 1e2)
    ax.set_xlabel("cascade index p   (v_t $\\propto \\ell^{\\,p}$; p=1/2 is [D]-grade)")
    ax.set_ylabel(r"length scale [pc]")
    ax.set_title("C. Why C2 fails: $\\ell_{cool}$ is unreachable\nbelow every real scale for all p<1")
    ax.legend(fontsize=6.5, loc="lower left"); ax.grid(alpha=0.25, which="both")

    fig.suptitle("SC-0 offline screen: three candidate f_A laws vs measured doses "
                 "(FA_STATE_COUPLED.md) — first-order, unboosted trajectories", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = PDV / "fa_state_screen.png"
    fig.savefig(out, dpi=135)
    print(f"wrote {out}  ({len(rows)} scored arms)")


if __name__ == "__main__":
    main()
