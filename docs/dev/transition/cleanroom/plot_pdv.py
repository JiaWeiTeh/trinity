#!/usr/bin/env python3
"""Should the PdV work term go into the cooling-ratio trigger? (user Q, 2026-06-21)

The current trigger is F0 = (Lgain-Lloss)/Lgain < 0.05 -- it EXCLUDES the expansion-work
term PdV = 4*pi*R2^2*v2*Pb. Adding it gives (Lgain-Lloss-PdV)/Lgain, which is just the
normalised net energy rate dEb/dt -- i.e. the Eb-peak criterion (the harvest oracle).

Two panels (pure read of data/c0_*_h0.csv):
  A: where the input goes (simple_cluster) -- input = cooling (Lloss) + work (PdV) + net
     (Eb growth). PdV is ~45% of Lgain: the bubble spends nearly half the input DRIVING
     the shell. That work is the energy-driving mechanism, not a radiative loss.
  B: F0 (no PdV, solid) vs F0+PdV (dashed) for all six. Subtracting PdV pulls the ratio
     from ~0.5 down to ~0.05-0.15 -- nearly firing -- but it tracks dEb/dt (the Eb-peak),
     which for 5/6 the bubble never reaches (Eb grows monotonically). It would fire when
     the bubble is HEALTHIEST (max work, still hot), not at radiative death.

    python plot_pdv.py
"""
from __future__ import annotations

import csv
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blowout_marker import mark

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"  # parents[3]=repo root
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False
FOURPI = 4 * math.pi
WONG = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            t = float(r["t_now"]); Lg = float(r["bubble_Lgain"]); Ll = float(r["bubble_Lloss"])
            R2 = float(r["R2"]); v2 = float(r["v2"]); Pb = float(r["Pb"])
        except (ValueError, KeyError, TypeError):
            continue
        if Lg <= 0:
            continue
        pdv = FOURPI * R2 * R2 * v2 * Pb
        out.append(dict(t=t, cool=Ll / Lg, work=pdv / Lg, net=(Lg - Ll - pdv) / Lg, f0=(Lg - Ll) / Lg))
    return out


def main():
    paths = sorted(glob.glob(str(HERE / "data" / "c0_*_h0.csv")))
    data = {Path(p).stem.replace("c0_", "").replace("_h0", ""): load(p) for p in paths}
    data = {k: v for k, v in data.items() if v}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    # Panel A: input partition for simple_cluster (cooling + work + net = 1)
    nameA = "simple_cluster" if "simple_cluster" in data else next(iter(data))
    d = data.get("simple_cluster") or next(iter(data.values()))
    t = [x["t"] for x in d]
    cool = [x["cool"] for x in d]; work = [x["work"] for x in d]; net = [x["net"] for x in d]
    axA.stackplot(t, cool, work, [max(n, 0) for n in net],
                  labels=[r"$L_{\rm loss}/L_{\rm gain}$ (radiated, ~20%)",
                          r"PdV$/L_{\rm gain}$ (work on shell, ~45%)",
                          r"net $\dot E_b/L_{\rm gain}$ (Eb growth, ~35%)"],
                  colors=["#D55E00", "#0072B2", "#009E73"], alpha=0.85)
    # blowout for this single config (star on the radiated-fraction curve)
    mark(axA, nameA, t=t, y=cool, color="0.2", label=True)
    axA.set_xscale("log"); axA.set_ylim(0, 1.05); axA.set_xlabel("t  [Myr]")
    axA.set_ylabel(r"fraction of $L_{\rm gain}$")
    axA.set_title("Where the input goes (simple_cluster):\nnearly half is expansion WORK, not radiated",
                  fontsize=10)
    axA.legend(loc="lower center", fontsize=8, framealpha=0.9)

    # Panel B: F0 (no PdV) vs F0+PdV (=net) for all configs
    for i, (name, d) in enumerate(sorted(data.items())):
        t = [x["t"] for x in d]
        f0 = [x["f0"] for x in d]
        axB.plot(t, f0, color=WONG[i % 6], lw=1.4, ls="-")
        axB.plot(t, [x["net"] for x in d], color=WONG[i % 6], lw=1.4, ls="--", alpha=0.9)
        # each config's blowout: star on its F0 (solid, primary) curve, own colour
        mark(axB, name, t=t, y=f0, color=WONG[i % 6], label=False)
    axB.axhline(0.05, color="k", ls=":", lw=1.1)
    axB.text(axB.get_xlim()[0] if False else 4e-3, 0.075, "trigger 0.05", fontsize=8)
    axB.axhline(0, color="0.6", lw=0.7)
    axB.set_xscale("log"); axB.set_xlabel("t  [Myr]"); axB.set_ylabel("trigger ratio")
    axB.set_ylim(-0.1, 1.0)
    axB.plot([], [], color="0.3", ls="-", label="F0 (current, no PdV)")
    axB.plot([], [], color="0.3", ls="--", label=r"F0+PdV $=\dot E_b/L_{\rm gain}$ (Eb-peak)")
    axB.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axB.set_title("Adding PdV pulls ~0.5 → ~0.05-0.15 (nearly fires),\nbut it measures the Eb-peak, "
                  "not cooling — still never < 0.05 (5/6)", fontsize=10)

    fig.suptitle("PdV is the energy-DRIVING work (~45% of input), not a loss — "
                 "putting it in the trigger asks 'has Eb peaked?', not 'is cooling winning?'",
                 fontsize=11.5)
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"pdv_trigger.{ext}", dpi=150)
    print(f"wrote {out}/pdv_trigger.(pdf,png) from {len(data)} configs")


if __name__ == "__main__":
    main()
