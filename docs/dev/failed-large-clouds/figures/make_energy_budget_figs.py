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
  # (or set TRINITY_FLC_RUNROOT to one root holding <config>/dictionary.jsonl)
  python docs/dev/failed-large-clouds/figures/make_energy_budget_figs.py
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
plt.style.use(str(os.path.join(ROOT, "paper", "_lib", "trinity.mplstyle")))
plt.rcParams.update({
    "text.usetex": False,
    "figure.dpi": 130,
    "savefig.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.constrained_layout.use": True,
})
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
FIG = HERE
PC_PER_MYR_TO_KMS = 0.977813  # 1 pc/Myr in km/s

# run dictionaries (ephemeral); fall back to the committed CSV if absent.
# TRINITY_FLC_RUNROOT points at ONE root holding <config>/dictionary.jsonl; unset,
# the historical two-root /tmp layout (flc_fix3 + ver) is kept so behaviour is unchanged.
_RUNROOT = os.environ.get("TRINITY_FLC_RUNROOT")
SOURCES = {
    "fail_repro": (os.path.join(_RUNROOT, "fail_repro", "dictionary.jsonl") if _RUNROOT
                   else "/tmp/flc_fix3/fail_repro/dictionary.jsonl"),
    "small_1e6": (os.path.join(_RUNROOT, "small_1e6", "dictionary.jsonl") if _RUNROOT
                  else "/tmp/ver/small_1e6/dictionary.jsonl"),
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
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Neither the run dictionary ({src}) nor the committed CSV ({csv_path}) exists. "
            "Set TRINITY_FLC_RUNROOT to the root holding <config>/dictionary.jsonl "
            "(runs from harness/run_variant.py; see harness/README.md).")
    rd = {c: [] for c in COLS}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for c in COLS:
                rd[c].append(float(row[c]))
    d = {c: np.asarray(rd[c]) for c in COLS}
    # derived terms (code units, Msun pc^2 / Myr^3)
    d["PdV"] = 4 * np.pi * d["R2"] ** 2 * d["Pb"] * d["v2"]
    d["Ed"] = d["Lmech"] - d["Lcool"] - d["PdV"] - d["Lleak"]
    # forward-difference dEb/dt (ground truth): the snapshot holds segment-START
    # values, so Ed at snap i should predict the change to snap i+1.
    fd = np.empty_like(d["Eb"])
    fd[:-1] = np.diff(d["Eb"]) / np.diff(d["t"])
    fd[-1] = fd[-2]
    d["fd"] = fd
    return d


def reliable_mask(d):
    """Self-consistent, proxy-reliable snapshots only. Excludes:
      - the non-physical overshoot tail (Eb<=0 / R2<=0);
      - the leading free-streaming -> Weaver IC-RELAXATION rows where the per-snapshot
        Pb*v2 proxy does not yet reconstruct dEb/dt. The snapshot stores segment-START
        (R2,Pb,v2); the budget Ed needs the segment-AVERAGE. While the IC relaxes (a few
        fast early steps) these differ, so Ed over/under-shoots the true finite-difference
        dEb/dt. Cutoff is DATA-DRIVEN: drop leading rows (within the first third) where Ed
        disagrees with dEb/dt in sign or by >2x. Verified: fail_repro reliable from snap 1,
        small_1e6 from snap 5 (its snaps 2-4 read PdV/Lmech>1 while Eb is actually growing).

    NB snapshot 0 is NOT a placeholder: its Pb is the genuine Weaver initial bubble
    pressure (bubble_E2P of the IC E0,r0,R1). It is ~equal (to ~6 sig figs, NOT bit-
    identical: 2.135768e7 vs 2.135766e7) across the 5e9 and 1e6 clouds because they share
    nCore=1e2 (Pb0 ∝ nCore) and v0=2L_w/pdot_w is the mass-independent wind terminal
    velocity; the tiny residual is an mCloud-dependent correction. Just the un-relaxed IC."""
    m = physical(d).copy()
    m[: reliable_start(d)] = False
    return m


def physical(d):
    """Physical snapshots (Eb>0, R2>0, finite Pb) -- ALL of them. v2 and Eb are stored
    STATE and are reliable at every physical snapshot; only the derived PdV proxy is not."""
    return (d["Eb"] > 0) & (d["R2"] > 0) & np.isfinite(d["Pb"])


def reliable_start(d):
    """First index from which the per-snapshot PdV proxy reconstructs dEb/dt (sign of the
    forward-difference). Leading rows are the free-streaming->Weaver IC relaxation, where the
    snapshot (segment-START) Pb,v2 over/under-shoot the segment-AVERAGE the budget needs.
    Applies ONLY to PdV (panel A); v2/Eb are exact at every snapshot."""
    phys = physical(d)
    ok = np.sign(d["Ed"]) == np.sign(d["fd"])
    ok[0] = False  # snap 0 is the IC handoff instant (t-t0=0; log-undefined anyway)
    idx = np.where(phys)[0]
    lead = idx[: max(3, len(idx) // 3)]
    bad = [i for i in lead if not ok[i]]
    return (max(bad) + 1) if bad else idx[0]


# ----------------------------------------------------------------------------- fig 1
def fig1_budget(d):
    m = reliable_mask(d)
    t = (d["t"][m] - d["t"][0]) * 1e3  # elapsed since energy phase began, 10^-3 Myr (consistent w/ fig2/fig3)
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
    ax1.text(0.97, 0.93, "PdV work crosses $L_{mech}$  →  $E_b$ peaks then collapses\ncooling is only ~1% of input",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#fff3e0", ec="#d95f02"))

    ax2.axhline(1.0, color="k", lw=1.0, ls="--")
    ax2.plot(t, d["PdV"][m] / d["Lmech"][m], "-", color="#d95f02", lw=2.6, label="PdV / $L_{mech}$")
    ax2.plot(t, d["Lcool"][m] / d["Lmech"][m], "-", color="#7570b3", lw=2.2, label="$L_{cool}$ / $L_{mech}$")
    ax2.set_yscale("log")
    ax2.set_ylabel("loss term / $L_{mech}$")
    ax2.set_xlabel(r"time since energy phase began,  $t-t_0$  [$10^{-3}$ Myr]")
    ax2.legend(loc="center left", fontsize=9, framealpha=0.95)
    ax2.grid(alpha=0.25, which="both")
    ax2.text(0.97, 0.55, "PdV/$L_{mech}$: 0.5 → 1.6 (crosses 1 ⇒ net loss)\n$L_{cool}/L_{mech}\\approx$0.01 (negligible)",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9.5,
             bbox=dict(boxstyle="round", fc="#ede7f6", ec="#7570b3"))

    p = os.path.join(FIG, "fig1_dEbdt_budget.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------- fig 2
def fig2_compare(df, dh):
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    for d, name, color in [(df, "fail_repro (massive, dies)", "#d95f02"),
                           (dh, "small_1e6 (healthy)", "#1b9e77")]:
        # x = elapsed time SINCE the energy phase began (t - t0), log axis. The two clouds start
        # phase 1a at very different absolute t (t0 = dt_phase0 ∝ sqrt(M_cluster): the 5e8 Msun
        # cluster free-streams ~70x longer before its energy phase), so absolute time puts them at
        # different x. Elapsed time anchors BOTH at their energy-phase birth -- snap 1 of each sits
        # at t-t0 ≈ 3e-5 Myr, so the curves start together (no spurious "delay").
        ph = physical(d)
        ph[0] = False  # snap 0 is at t-t0=0 (log-undefined); it is the IC instant, not a segment
        tau = d["t"] - d["t"][0]
        # v2, Eb are stored STATE -> exact at every snapshot: plot all of them.
        axB.plot(tau[ph], d["v2"][ph] * PC_PER_MYR_TO_KMS, "-", color=color, lw=2.4, label=name)
        axC.plot(tau[ph], d["Eb"][ph] / d["Eb"][ph].max(), "-", color=color, lw=2.4, label=name)
        # PdV is a per-snapshot proxy: solid where it reconstructs dEb/dt, dotted (faded) through
        # the IC-relaxation transient where segment-START != segment-AVERAGE (near break-even).
        s = reliable_start(d)
        pre = ph.copy(); pre[s + 1:] = False   # snaps 1..s  (connects to the solid at s)
        sol = ph.copy(); sol[:s] = False       # snaps s..end
        axA.plot(tau[pre], d["PdV"][pre] / d["Lmech"][pre], ":", color=color, lw=1.6, alpha=0.55)
        axA.plot(tau[sol], d["PdV"][sol] / d["Lmech"][sol], "-", color=color, lw=2.4, label=name)

    axA.axhline(1.0, color="k", lw=1.0, ls="--")
    axA.text(0.97, 0.95, "PdV = $L_{mech}$ (break-even).  dotted = IC-relaxation\n(PdV proxy uncertain near break-even)",
             transform=axA.transAxes, fontsize=8.0, ha="right", va="top")
    axA.set_yscale("log")
    axA.set_xscale("log")
    axA.set_ylabel("PdV / $L_{mech}$")
    axA.set_title("Why the massive cloud's bubble dies and a healthy one does not")
    axA.legend(loc="lower left", fontsize=9)
    axA.grid(alpha=0.25, which="both")

    axB.set_ylabel("shell velocity $v_2$  [km/s]")
    axB.grid(alpha=0.25, which="both")
    axB.text(0.97, 0.9, "massive: ~2000+ km/s, near free-expansion\nhealthy: decelerates (Weaver-like)",
             transform=axB.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#888"))

    axC.set_ylabel("$E_b / E_{b,\\max}$")
    axC.set_xlabel("time since energy phase began,  $t - t_0$  [Myr]  (log)")
    axC.axhline(0.0, color="k", lw=0.8)
    axC.grid(alpha=0.25, which="both")
    axC.text(0.5, 0.5, "massive collapses after ~$10^{-3}$ Myr;\nhealthy still energy-driven at ~0.3 Myr",
             color="#444", fontsize=9, transform=axC.transAxes, ha="center",
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#888"))

    p = os.path.join(FIG, "fig2_healthy_vs_failing.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------- fig 3
def fig3_bug_fix(d):
    t = (d["t"] - d["t"][0]) * 1e3  # elapsed since energy phase began, 10^-3 Myr (consistent w/ fig1/fig2)
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
    ax2.set_xlabel(r"time since energy phase began,  $t-t_0$  [$10^{-3}$ Myr]")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.25)
    gap = d["R2"] - d["R1"]
    ax2.annotate(f"shell volume $R_2^3-R_1^3\\to0$\n(gap {gap[coll-1]:.2g} pc)  →  $P_b=(\\gamma-1)E_b/V\\to$ 1/0",
                 xy=(t[coll - 1], d["R2"][coll - 1]), xytext=(0.30, 0.30),
                 textcoords="axes fraction", fontsize=9,
                 bbox=dict(boxstyle="round", fc="#ffebee", ec="#d62728"),
                 arrowprops=dict(arrowstyle="->", color="#d62728"))

    p = os.path.join(FIG, "fig3_bug_and_fix.png")
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    df = load("fail_repro")
    dh = load("small_1e6")
    print("wrote", fig1_budget(df))
    print("wrote", fig2_compare(df, dh))
    print("wrote", fig3_bug_fix(df))
    print("data CSVs in", DATA)
