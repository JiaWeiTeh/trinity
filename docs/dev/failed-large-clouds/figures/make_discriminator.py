"""Discriminator: were the failing clouds ever genuinely energy-driven?

For each config, measures how much the hot-bubble thermal reservoir Eb actually grew
(Eb_peak/Eb_init -- a pure state variable, fully reliable) and how PdV/Lmech evolves.
Writes a committed CSV + a 2-panel figure so the finding reproduces without the (ephemeral)
run dictionaries.

Finding: all clouds start at the SAME self-similar handoff (PdV/Lmech ~ 0.5). Healthy clouds
decelerate -> PdV/Lmech falls -> the reservoir builds 10^4x (true Weaver energy-driven). Failing
clouds never decelerate -> PdV/Lmech rises through 1 within ~10% of the phase -> the reservoir
grows only ~1% before Eb collapses. The failing clouds are "stillborn" energy-driven bubbles.

Reproduce:
  python docs/dev/failed-large-clouds/figures/make_discriminator.py
"""
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CSV = os.path.join(DATA, "discriminator.csv")

# (label, ephemeral dict, kind, color)
RUNS = [
    ("fail_repro", "/tmp/flc_fix3/fail_repro/dictionary.jsonl", "fail", "#d95f02"),
    ("fail_pism6", "/tmp/flc_fix3/fail_pism6/dictionary.jsonl", "fail", "#e08214"),
    ("fail_helix", "/tmp/flc_fix3/fail_helix/dictionary.jsonl", "fail", "#b35806"),
    ("small_1e6", "/tmp/ver/small_1e6/dictionary.jsonl", "healthy", "#1b9e77"),
    ("small_1e5", "/tmp/ver/small_1e5/dictionary.jsonl", "healthy", "#41ae76"),
]


def series(path):
    """Per-snapshot phase-fraction f, Eb/Eb_init, PdV/Lmech (from snap 1; snap 0 = IC)."""
    R = [json.loads(l) for l in open(path)]
    Eb = np.array([r["Eb"] for r in R])
    Lm = np.array([r["Lmech_total"] for r in R])
    pdv = np.array([4 * math.pi * r["R2"] ** 2 * r["Pb"] * r["v2"] for r in R])
    n = len(R)
    f = (np.arange(n) - 1) / (n - 1)  # phase fraction, snap1 -> 0
    return f[1:], (Eb / Eb[0])[1:], (pdv / Lm)[1:]


def metrics():
    rows = []
    for label, path, kind, color in RUNS:
        if os.path.exists(path):
            f, ebn, ratio = series(path)
            cross = next((i for i in range(len(f)) if ratio[i] > 1), None)
            rows.append({
                "config": label, "kind": kind,
                "pdv_over_lmech_step1": round(float(ratio[0]), 3),
                "eb_growth_factor": float(f"{ebn.max():.4g}"),
                "pdv_gt1_first_phasefrac": round(float(f[cross]), 3) if cross is not None else "",
                "frac_phase_pdv_gt1": round(float((ratio > 1).mean()), 3),
            })
    if rows:
        with open(CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def figure():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    growth = {}
    for label, path, kind, color in RUNS:
        if not os.path.exists(path):
            continue
        f, ebn, ratio = series(path)
        ls = "-" if kind == "fail" else "--"
        axA.plot(f, ebn, ls, color=color, lw=2.2, label=label)
        axB.plot(f, ratio, ls, color=color, lw=2.2, label=label)
        growth[label] = (ebn.max(), color)

    axA.set_yscale("log")
    axA.set_ylim(1e-2, 1e5)
    axA.set_xlabel("fraction of the energy phase")
    axA.set_ylabel("$E_b / E_{b,\\mathrm{init}}$   (reservoir growth)")
    axA.set_title("Did the thermal reservoir ever build?")
    axA.axhline(1.0, color="k", lw=0.8, ls=":")
    axA.text(0.30, 1.7, "reservoir never builds (failing: $\\times$1.01)", fontsize=9, color="#b35806")
    axA.text(0.04, 6e3, "healthy: $\\times10^{4}$", fontsize=10, color="#1b9e77")
    axA.text(0.62, 3e-2, "failing: $E_b\\to0$ (collapse)", fontsize=8.5, color="#b35806")
    axA.legend(fontsize=8, loc="lower left")
    axA.grid(alpha=0.25, which="both")

    axB.axhline(1.0, color="k", lw=1.0, ls="--")
    axB.set_ylim(0.3, 1.8)
    axB.set_xlabel("fraction of the energy phase")
    axB.set_ylabel("PdV / $L_{mech}$")
    axB.set_title("...or did PdV run away?")
    axB.text(0.97, 1.04, "break-even", fontsize=8.5, ha="right")
    axB.text(0.30, 1.52, "failing: rises through 1", fontsize=9.5, color="#b35806")
    axB.text(0.30, 0.46, "healthy: falls, stays $<1$", fontsize=9.5, color="#1b9e77")
    axB.legend(fontsize=8, loc="upper left")
    axB.grid(alpha=0.25)

    fig.suptitle("Were the failing clouds ever genuinely energy-driven?  (all start at the same ~0.5 handoff)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(HERE, "fig4_energy_driven_discriminator.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


if __name__ == "__main__":
    rows = metrics()
    for r in rows:
        print(r)
    print("wrote", CSV)
    print("wrote", figure())
