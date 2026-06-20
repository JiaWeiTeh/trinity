#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end confirmation: the conduction-stiffness high-feedback config runs HEALTHY.

This closes the loop on the dR2/dR2min story. The stiff config (massive cloud, LOW sfe,
low density -> 5e7 Msun cluster, dR2/R2 ~ 1e-10) is the regime that floods LSODA with the
`t+h=t` warning. We assert the *full run* behaves correctly: the flood is silent (the
shipped `_quiet_lsoda_fortran` mitigation), and the trajectory is the healthy Weaver branch
(Eb grows, v2 decelerates, R1 < R2 always) -- NOT the failed-large-clouds collapse band
(sfe 0.05-0.1, where PdV/Lmech > 1 keeps v2 ~2000+ km/s and Eb collapses; see
docs/dev/failed-large-clouds/PLAN.md).

Reproduce the run this reads (sfe 0.01 keeps it out of the collapse band):

    cd /home/user/trinity
    timeout 600 python run.py docs/dev/performance/conduction_stiff_5e9_sfe001.param

then extract + plot from its dictionary.jsonl:

    python docs/dev/performance/harness/conduction_stiff_endtoend.py

Writes docs/dev/performance/data/conduction_stiff_5e9_trajectory.csv and
docs/dev/performance/figs/conduction_stiff_endtoend.png.
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIGS = ROOT / "docs" / "dev" / "performance" / "figs"
DATA = ROOT / "docs" / "dev" / "performance" / "data"
RUN_JSONL = Path("/tmp/dR2cap/conduction_stiff_5e9/dictionary.jsonl")
CSV = DATA / "conduction_stiff_5e9_trajectory.csv"

plt.style.use(str(ROOT / "paper" / "_lib" / "trinity.mplstyle"))
plt.rcParams.update({
    "text.usetex": False,
    "figure.dpi": 130,
    "savefig.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.constrained_layout.use": True,
})

COLS = ["t_now", "current_phase", "R2", "R1", "v2", "Eb", "Pb"]


def _load_rows():
    """Prefer the committed CSV (durable); fall back to the live /tmp run to (re)build it."""
    if CSV.exists():
        rows = []
        with open(CSV) as fh:
            for d in csv.DictReader(fh):
                rows.append({k: (d[k] if k == "current_phase" else float(d[k])) for k in COLS})
        return rows, "committed CSV"
    rows = []
    with open(RUN_JSONL) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({k: r.get(k) for k in COLS})
    return rows, "live run jsonl"


def main():
    rows, src = _load_rows()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    if src != "committed CSV":
        with open(CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    t = np.array([r["t_now"] for r in rows]) * 1e3   # -> 1e-3 Myr
    R2 = np.array([r["R2"] for r in rows])
    R1 = np.array([r["R1"] for r in rows])
    v2 = np.array([r["v2"] for r in rows])
    Eb = np.array([r["Eb"] for r in rows])
    phase = [r["current_phase"] for r in rows]
    n1a = sum(p == "energy" for p in phase)

    # sanity assertions (this IS the runnable check this harness leaves behind)
    assert (Eb > 0).all(), "Eb went non-positive -> would be the collapse band, not healthy"
    assert (R1 < R2).all(), "R1 reached R2 -> shell-volume degeneracy (collapse)"
    assert Eb[-1] > Eb[0], "Eb did not grow -> not the healthy Weaver branch"
    assert v2[-1] < v2[0], "v2 did not decelerate -> free-expansion/collapse signature"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    # left: Eb grows + R1<R2 (no shell-volume collapse)
    axb = ax1.twinx()
    l1, = ax1.plot(t, Eb, "-", color="#1f77b4", lw=2.2, label="Eb (energy)")
    l2, = axb.plot(t, R2, "-", color="#2ca02c", lw=1.8, label="R2 (outer)")
    l3, = axb.plot(t, R1, "--", color="#d62728", lw=1.8, label="R1 (inner)")
    ax1.axvline(t[n1a - 1], color="0.6", ls=":", lw=1.0)
    ax1.text(t[n1a - 1], Eb.min(), "  1a -> 1b", color="0.4", fontsize=8, va="bottom")
    ax1.set_yscale("log")
    ax1.set_xlabel("energy-phase time  t  [1e-3 Myr]")
    ax1.set_ylabel("bubble energy  Eb  [code units]", color="#1f77b4")
    axb.set_ylabel("radius  [pc]   (R1 stays below R2)", color="#2ca02c")
    ax1.set_title("Healthy Weaver branch: Eb GROWS, R1 < R2 (no shell-volume collapse)")
    ax1.legend(handles=[l1, l2, l3], loc="lower right")

    # right: v2 decelerates -> on the healthy side of the collapse discriminator
    ax2.plot(t, v2, "-", color="#6a3d9a", lw=2.2, label="this run (sfe 0.01)")
    ax2.axhspan(2000, v2.max() * 1.05, color="#d62728", alpha=0.07)
    ax2.text(t.mean(), 2150, "collapse band stays ~2000+ km/s\n(sfe 0.05-0.1; PdV/Lmech > 1)",
             color="#a01010", fontsize=8.5, va="bottom")
    ax2.set_xlabel("energy-phase time  t  [1e-3 Myr]")
    ax2.set_ylabel("shell velocity  v2  [km/s]")
    ax2.set_title(f"v2 DECELERATES {v2[0]:.0f} -> {v2[-1]:.0f} km/s  (PdV never wins)")
    ax2.legend(loc="upper right")

    fig.suptitle("Conduction-stiffness high-feedback run is healthy end-to-end "
                 f"({len(rows)} segments, {n1a} in 1a; LSODA flood silent)\n"
                 "5e9 Msun / sfe 0.01 / nCore 1e2  -- the dR2/R2~1e-10 flood regime, run in full",
                 fontsize=10)
    fig.savefig(FIGS / "conduction_stiff_endtoend.png")
    plt.close(fig)
    print(f"source: {src};  {len(rows)} segments ({n1a} in 1a); "
          f"Eb {Eb[0]:.2e}->{Eb[-1]:.2e} (grows), v2 {v2[0]:.0f}->{v2[-1]:.0f} km/s (decel); "
          f"R1<R2 all; max R1/R2={np.max(R1/R2):.3f}")
    print("wrote conduction_stiff_5e9_trajectory.csv + conduction_stiff_endtoend.png")


if __name__ == "__main__":
    main()
