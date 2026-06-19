"""Diagnostic figures for the massive-cloud energy-phase collapse (failed-large-clouds).

Tells the three-part story behind ENERGY_COLLAPSED (code 51):
  fig1_dEbdt_budget.png      -- the FINDING: dEb/dt is dominated by PdV work, not cooling
  fig2_healthy_vs_failing.png-- the COMPARISON: why this regime dies and a healthy cloud does not
  fig3_bug_and_fix.png       -- the BUG (Eb->0 collapses R1->R2 -> 1/0 -> NaN) and the FIX

Faithful to the code's energy ODE (trinity/phase1_energy/energy_phase_ODEs.py:280):
    Ed = (Lmech_total - L_bubble) - (4*pi*R2**2 * press_bubble) * v2 - L_leak
  gain   = Lmech_total                 (mechanical luminosity in)
  cool   = L_bubble  = bubble_LTotal   (radiative cooling)
  PdV    = 4*pi*R2**2 * Pb * v2        (expansion work on the shell; press_bubble ~ snapshot Pb)
  leak   = L_leak    = bubble_Leak

Durable inputs: extracted once from the run dictionaries into committed CSVs under
data/, so the figures reproduce WITHOUT re-running the (hours-long) sims.

Reproduce:
  # sources (ephemeral runs, harness variant V0 = production code):
  #   fail_repro : docs/dev/failed-large-clouds/harness/run_variant.py --variant V0 \
  #                  --param params/fail_repro.param --out /tmp/flc_fix3/fail_repro
  #   small_1e6  : ... --param params/small_1e6.param --out /tmp/ver/small_1e6
  python docs/dev/failed-large-clouds/figures/make_energy_budget_figs.py
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
FIG = HERE
PC_PER_MYR_TO_KMS = 0.977813  # 1 pc/Myr in km/s

# run dictionaries (ephemeral); fall back to the committed CSV if absent
SOURCES = {
    "fail_repro": "/tmp/flc_fix3/fail_repro/dictionary.jsonl",
    "small_1e6": "/tmp/ver/small_1e6/dictionary.jsonl",
}
COLS = ["t", "Eb", "R1", "R2", "v2", "Pb", "Lmech", "Lcool", "Lleak"]


def load(name):
    """Return dict of np arrays. Extract from the run dictionary into a committed
    CSV the first time; thereafter read the CSV so the figure is reproducible."""
    csv_path = os.path.join(DATA, f"budget_{name}.csv")
    src = SOURCES[name]
    if os.path.exists(src):
        rows = [json.loads(line) for line in open(src)]
        out = {c: [] for c in COLS}
        for r in rows:
            out["t"].append(r.get("t_now", np.nan))
            out["Eb"].append(r.get("Eb", np.nan))
            out["R1"].append(r.get("R1", np.nan))
            out["R2"].append(r.get("R2", np.nan))
            out["v2"].append(r.get("v2", np.nan))
            out["Pb"].append(r.get("Pb", np.nan))
            out["Lcool"].append(r.get("bubble_LTotal") or np.nan)
            out["Lmech"].append(r.get("Lmech_total", np.nan))
            out["Lleak"].append(r.get("bubble_Leak") or 0.0)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for i in range(len(rows)):
                w.writerow([out[c][i] for c in COLS])
    rd = {c: [] for c in COLS}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for c in COLS:
                rd[c].append(float(row[c]))
    d = {c: np.asarray(rd[c]) for c in COLS}
    # derived terms (code units, Msun pc^2 / Myr^3)
    d["PdV"] = 4 * np.pi * d["R2"] ** 2 * d["Pb"] * d["v2"]
    d["Ed"] = d["Lmech"] - d["Lcool"] - d["PdV"] - d["Lleak"]
    return d


def physical_mask(d):
    """Snapshots before the ODE overshoots into the non-physical tail (Eb<=0/R2<=0)."""
    return (d["Eb"] > 0) & (d["R2"] > 0) & np.isfinite(d["Pb"])


# ----------------------------------------------------------------------------- fig 1
def fig1_budget(d):
    m = physical_mask(d)
    t = d["t"][m] * 1e3  # 10^-3 Myr
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True)

    ax1.plot(t, d["Lmech"][m], "-", color="#1b9e77", lw=2.2, label="gain: $L_{mech}$ (wind+SN in)")
    ax1.plot(t, d["PdV"][m], "-", color="#d95f02", lw=2.6, label=r"loss: PdV work $4\pi R_2^2 P_b\,v_2$")
    ax1.plot(t, d["Lcool"][m], "-", color="#7570b3", lw=2.2, label="loss: radiative cooling $L_{cool}$")
    ax1.plot(t, np.abs(d["Lleak"][m]), ":", color="#999999", lw=1.6, label="loss: conduction leak")
    ax1.set_yscale("log")
    ax1.set_ylabel("luminosity  [code units]")
    ax1.set_title("fail_repro  —  energy budget of $dE_b/dt$  (phase 1a/1b)")
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax1.grid(alpha=0.25, which="both")
    ax1.text(0.97, 0.93, "PdV work $>L_{mech}$  →  $E_b$ drained\ncooling is only ~1% of input",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#fff3e0", ec="#d95f02"))

    ax2.axhline(1.0, color="k", lw=1.0, ls="--")
    ax2.plot(t, d["PdV"][m] / d["Lmech"][m], "-", color="#d95f02", lw=2.6, label="PdV / $L_{mech}$")
    ax2.plot(t, d["Lcool"][m] / d["Lmech"][m], "-", color="#7570b3", lw=2.2, label="$L_{cool}$ / $L_{mech}$")
    ax2.set_yscale("log")
    ax2.set_ylabel("loss term / $L_{mech}$")
    ax2.set_xlabel(r"time  [$10^{-3}$ Myr]")
    ax2.legend(loc="center left", fontsize=9, framealpha=0.95)
    ax2.grid(alpha=0.25, which="both")
    ax2.text(0.97, 0.55, "PdV/$L_{mech}\\approx$1.3–2.7  (net loss)\n$L_{cool}/L_{mech}\\approx$0.01",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#ede7f6", ec="#7570b3"))

    fig.tight_layout()
    p = os.path.join(FIG, "fig1_dEbdt_budget.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------- fig 2
def fig2_compare(df, dh):
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    for d, name, color in [(df, "fail_repro (massive, dies)", "#d95f02"),
                           (dh, "small_1e6 (healthy)", "#1b9e77")]:
        m = physical_mask(d)
        tt = d["t"][m]
        tn = (tt - tt.min()) / (tt.max() - tt.min())  # normalized 0..1 over the phase span
        axA.plot(tn, d["PdV"][m] / d["Lmech"][m], "-", color=color, lw=2.4, label=name)
        axB.plot(tn, d["v2"][m] * PC_PER_MYR_TO_KMS, "-", color=color, lw=2.4, label=name)
        axC.plot(tn, d["Eb"][m] / d["Eb"][m].max(), "-", color=color, lw=2.4, label=name)

    axA.axhline(1.0, color="k", lw=1.0, ls="--")
    axA.text(0.02, 1.05, "PdV = $L_{mech}$  (energy-driven break-even)", fontsize=8.5, va="bottom")
    axA.set_yscale("log")
    axA.set_ylabel("PdV / $L_{mech}$")
    axA.set_title("Why the massive cloud's bubble dies and a healthy one does not")
    axA.legend(loc="center right", fontsize=9)
    axA.grid(alpha=0.25, which="both")

    axB.set_ylabel("shell velocity $v_2$  [km/s]")
    axB.grid(alpha=0.25)
    axB.text(0.97, 0.9, "massive: ~2000+ km/s, near free-expansion\nhealthy: decelerates (Weaver-like)",
             transform=axB.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#888"))

    axC.set_ylabel("$E_b / E_{b,\\max}$")
    axC.set_xlabel("fraction of evolution  (t / $t_{end}$)")
    axC.axhline(0.0, color="k", lw=0.8)
    axC.grid(alpha=0.25)
    axC.text(0.5, 0.1, "massive: $E_b\\to0$ (collapses)", color="#d95f02", fontsize=9.5,
             transform=axC.transAxes, ha="center")

    fig.tight_layout()
    p = os.path.join(FIG, "fig2_healthy_vs_failing.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------- fig 3
def fig3_bug_fix(d):
    t = d["t"] * 1e3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.6), sharex=True)

    # collapse index: first non-physical Eb
    coll = np.argmax(d["Eb"] <= 0) if np.any(d["Eb"] <= 0) else len(d["Eb"]) - 1
    m = np.arange(len(t)) <= coll

    ax1.axhline(0, color="k", lw=0.8)
    ax1.plot(t[m], d["Eb"][m], "-o", color="#333", ms=3, lw=1.8, label="$E_b(t)$")
    ax1.plot(t[coll], d["Eb"][coll], "X", color="#d62728", ms=15, mec="k",
             label="OLD: $E_b\\leq 0$ → solve_R1 / E2P NaN → ValueError crash")
    ax1.plot(t[coll - 1], d["Eb"][coll - 1], "P", color="#2ca02c", ms=14, mec="k",
             label="NEW: detect $E_b\\leq 0$ → ENERGY_COLLAPSED (51), clean stop")
    ax1.set_ylabel("bubble energy  $E_b$  [code units]")
    ax1.set_title("fail_repro  —  the crash and the fix")
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.96)
    ax1.grid(alpha=0.25)

    # geometric trigger: R1 -> R2, shell volume -> 0
    ax2.plot(t[m], d["R2"][m], "-", color="#1f77b4", lw=2.2, label="$R_2$ (outer / shell)")
    ax2.plot(t[m], d["R1"][m], "-", color="#ff7f0e", lw=2.2, label="$R_1$ (wind shock)")
    ax2.fill_between(t[m], d["R1"][m], d["R2"][m], color="#1f77b4", alpha=0.15)
    ax2.set_ylabel("radius  [pc]")
    ax2.set_xlabel(r"time  [$10^{-3}$ Myr]")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.25)
    gap = d["R2"] - d["R1"]
    ax2.annotate(f"shell volume $R_2^3-R_1^3\\to0$\n(gap {gap[coll-1]:.2g} pc)  →  $P_b=(\\gamma-1)E_b/V\\to$ 1/0",
                 xy=(t[coll - 1], d["R2"][coll - 1]), xytext=(0.30, 0.30),
                 textcoords="axes fraction", fontsize=9,
                 bbox=dict(boxstyle="round", fc="#ffebee", ec="#d62728"),
                 arrowprops=dict(arrowstyle="->", color="#d62728"))

    fig.tight_layout()
    p = os.path.join(FIG, "fig3_bug_and_fix.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


if __name__ == "__main__":
    df = load("fail_repro")
    dh = load("small_1e6")
    print("wrote", fig1_budget(df))
    print("wrote", fig2_compare(df, dh))
    print("wrote", fig3_bug_fix(df))
    print("data CSVs in", DATA)
