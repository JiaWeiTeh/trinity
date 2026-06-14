#!/usr/bin/env python3
"""Negative interior-velocity diagnosis (WARPFIELD "Problem 2") from the
committed stalling-phase CSVs.

Source data (canonical, read from analysis/data/):
  stalling_steep_1e6_alpha-2.csv  (sweep_steep, 133 rows)
  stalling_mock_4e3.csv           (sweep_mock,  144 rows)
See analysis/stalling-energy-phase.md for the writeup. Key diagnostics:
  v_struct_min  = most-negative point in the bubble velocity profile [pc/Myr]
  v_struct_nneg = count of negative-v points (of ~100 sample points; these
                  stalling CSVs predate the v_struct_npts column). >=10 = real
                  inflow, 1..9 = a single inner-BC grid point dipping (artifact).

Finding the plots make: the bubble's interior velocity goes negative (inflow)
when beta+delta goes sufficiently negative -- the source term of dv/dr is
(beta+delta)/t -- NOT simply when beta is negative. Here (steep baseline) the
cut sits near -0.5; the 6-config Phase-6 hunt (plot_hunt.py) refines the onset
to ~ -0.4. The driver is feedback luminosity surges (WR wind, then SN onset)
that re-pressurise the bubble (beta<0). Produces:
  - negvel_trigger.png    : (beta,delta) plane + beta+delta strip, inflow vs no
  - negvel_timeline.png   : steep vs mock -- Lmech surge -> beta+delta -> inflow
  - negvel_dmdt_lmech.png : dMdt vs Lmech_total per run (clean fn + transient lag)
  - negvel_feedback.png   : steep -- cooling_ratio + Eb/Pb vs t (stall + repress.)
  - negvel_causal.png     : steep -- the cause->consequence ladder (Lmech spike
                            -> Eb/Pb up (beta<0) -> beta+delta<-0.4 -> inflow)
                            with the beta/delta decomposition

Usage: python scratch/phase2/analyze_negvel.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "analysis" / "data"  # canonical stalling CSVs (was scratch dupes)
BPD_THRESH = -0.5  # beta+delta inflow threshold (doc)
NNEG_REAL = 10  # v_struct_nneg >= this = real inflow (vs inner-BC artifact)
RUNS = [
    ("stalling_steep_1e6_alpha-2.csv", "steep 1e6 α=−2", "o", "#d62728"),
    ("stalling_mock_4e3.csv", "mock 4e3", "^", "#1f77b4"),
]

_STYLE = HERE.parents[1] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams["text.usetex"] = False

NUM = (
    "t_now cool_beta cool_delta beta_plus_delta Pb bubble_dMdt Lmech_total Lmech_W "
    "Lmech_SN bubble_Lgain bubble_Lloss cooling_ratio v_struct_min v_struct_nneg "
    "R2 v2 Eb bubble_Tavg c_sound"
).split()


def load(fn):
    rows = list(csv.DictReader(open(DATA / fn)))
    return {k: np.array([float(r[k]) for r in rows]) for k in NUM}


INFLOW_C = "#b30000"  # solid colour for real-inflow points (size encodes count)
BG_FACE, BG_EDGE = "0.78", "0.5"  # visible gray for the no-inflow bulk
YROW = {"steep": 1.0, "mock": 0.0}  # strip-plot rows in panel B


def plot_trigger(path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    rng = np.random.default_rng(0)  # jitter so overlapping strip points are visible
    for fn, label, mk, _c in RUNS:
        d = load(fn)
        nneg = d["v_struct_nneg"]
        sig = nneg >= NNEG_REAL  # real inflow; rest is no-inflow / inner-BC artifact
        bg = ~sig
        key = "steep" if "steep" in fn else "mock"

        # panel A: (beta, delta) plane
        axA.scatter(
            d["cool_beta"][bg],
            d["cool_delta"][bg],
            s=16,
            marker=mk,
            facecolor=BG_FACE,
            edgecolor=BG_EDGE,
            linewidths=0.3,
            alpha=0.85,
        )
        axA.scatter(
            d["cool_beta"][sig],
            d["cool_delta"][sig],
            s=60 + nneg[sig] * 6,
            marker=mk,
            facecolor=INFLOW_C,
            edgecolor="k",
            linewidths=1.1,
            zorder=5,
        )

        # panel B: strip plot, y = run (categorical), x = beta+delta
        y = YROW[key] + rng.uniform(-0.13, 0.13, size=nneg.shape)
        axB.scatter(
            d["beta_plus_delta"][bg],
            y[bg],
            s=16,
            marker=mk,
            facecolor=BG_FACE,
            edgecolor=BG_EDGE,
            linewidths=0.3,
            alpha=0.85,
        )
        axB.scatter(
            d["beta_plus_delta"][sig],
            y[sig],
            s=60 + nneg[sig] * 6,
            marker=mk,
            facecolor=INFLOW_C,
            edgecolor="k",
            linewidths=1.1,
            zorder=5,
        )

    # panel A frame + trigger diagonals
    bb = np.array([-3.0, 2.6])
    axA.plot(bb, BPD_THRESH - bb, "r--", lw=1.4, label=r"$\beta+\delta=-0.5$ (trigger)")
    axA.plot(bb, -bb, color="0.5", ls=":", lw=1.0, label=r"$\beta+\delta=0$")
    axA.set_xlim(-2.8, 2.6)
    axA.set_ylim(-1.3, 2.2)
    axA.set_xlabel(r"$\beta$")
    axA.set_ylabel(r"$\delta$")
    axA.set_title(
        "Inflow (red, size ∝ count) sits below β+δ=−0.5,\nnot simply at low β", fontsize=10
    )
    for n in (10, 25, 50):  # size legend for the count encoding
        axA.scatter(
            [], [], s=60 + n * 6, facecolor=INFLOW_C, edgecolor="k", label=f"inflow nneg={n}"
        )
    axA.scatter([], [], s=16, facecolor=BG_FACE, edgecolor=BG_EDGE, label="no inflow")
    axA.legend(loc="upper right", fontsize=7, framealpha=0.92)

    # panel B frame
    axB.axvline(BPD_THRESH, color="r", ls="--", lw=1.4)
    axB.axvspan(-2.0, BPD_THRESH, color="r", alpha=0.06)
    axB.set_yticks([0, 1])
    axB.set_yticklabels(["mock\n4e3", "steep\n1e6 α−2"])
    axB.set_ylim(-0.6, 1.6)
    axB.set_xlim(-1.6, 3.5)
    axB.set_xlabel(r"$\beta+\delta$")
    axB.set_title(
        "β+δ per run — inflow only left of −0.5;\nmock never enters the zone", fontsize=10
    )
    axB.text(BPD_THRESH - 0.05, 1.5, "inflow zone", color="r", ha="right", va="top", fontsize=8)

    fig.suptitle(
        "Negative interior velocity (WARPFIELD Problem 2) is triggered by β+δ < −0.5, not by β alone",
        fontsize=12,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _timeline_col(axcol, d, title):
    """One run's 3-row column: Lmech surge / beta & beta+delta / inflow count."""
    a1, a2, a3 = axcol
    t = d["t_now"]
    lmax = d["Lmech_total"].max()  # normalise so the tiny mock cluster is comparable
    a1.plot(t, d["Lmech_W"] / lmax, color="#1f77b4", label=r"$L_{\rm mech,W}$ (wind)")
    a1.plot(t, d["Lmech_SN"] / lmax, color="#9467bd", label=r"$L_{\rm mech,SN}$")
    a1.plot(t, d["Lmech_total"] / lmax, color="k", lw=1.0, ls="--", label=r"$L_{\rm mech,total}$")
    a1.set_title(title, fontsize=11)

    # colourblind-safe (Wong): beta = blue, beta+delta = vermillion
    a2.plot(t, d["cool_beta"], color="#0072B2", label=r"$\beta$")
    a2.plot(t, d["beta_plus_delta"], color="#D55E00", lw=1.8, label=r"$\beta+\delta$")
    a2.axhline(BPD_THRESH, color="k", ls="--", lw=1.0)
    a2.axhline(0.0, color="0.6", ls=":", lw=0.8)

    a3.plot(t, d["v_struct_nneg"], color="#ff7f0e", marker=".", ms=4)
    a3.axhline(NNEG_REAL, color="0.4", ls=":", lw=1.0)

    real = d["v_struct_nneg"] >= NNEG_REAL
    if real.any():
        tlo, thi = t[real].min() - 0.04, t[real].max() + 0.04
        for ax in axcol:
            ax.axvspan(tlo, thi, color="orange", alpha=0.18, zorder=0)
        a3.text(
            0.97,
            0.9,
            f"inflow: v_min={d['v_struct_min'][real].min():.2f} pc/Myr",
            transform=a3.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#b30000",
        )
    else:
        a3.text(
            0.97,
            0.9,
            "no real inflow (nneg ≤ 1)",
            transform=a3.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.3",
        )


def plot_timeline(path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for r in range(3):  # share y per row so the two runs are directly comparable
        axes[r, 1].sharey(axes[r, 0])
    _timeline_col(axes[:, 0], load(RUNS[0][0]), "steep 1e6, α=−2  (inflow)")
    _timeline_col(axes[:, 1], load(RUNS[1][0]), "mock 4e3  (no inflow)")

    axes[0, 0].set_ylabel(r"$L_{\rm mech}$ / peak")
    axes[1, 0].set_ylabel(r"$\beta$,  $\beta+\delta$")
    axes[2, 0].set_ylabel("v_struct_nneg")
    axes[0, 0].legend(fontsize=7.5, loc="upper left")
    axes[1, 0].legend(fontsize=8, loc="upper left")
    axes[1, 0].text(2.55, BPD_THRESH, " β+δ=−0.5", color="k", va="bottom", fontsize=7.5)
    axes[2, 0].set_xlabel("t  [Myr]")
    axes[2, 1].set_xlabel("t  [Myr]")
    axes[0, 0].set_xlim(2.5, 4.0)  # the WR/SN epoch; t<2.5 quiescent for both
    fig.suptitle(
        "Same WR/SN surges in both — only steep drives β+δ below −0.5, so only steep gets inflow",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_dmdt_lmech(path):
    """dMdt vs Lmech_total per run; colour = time so a lag shows as a loop."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (fn, label, _mk, _c) in zip(axes, RUNS):
        d = load(fn)
        o = np.argsort(d["t_now"])
        L, M, tt = d["Lmech_total"][o], d["bubble_dMdt"][o], d["t_now"][o]
        ax.plot(L, M, color="0.8", lw=0.8, zorder=1)  # time-ordered path
        sc = ax.scatter(L, M, c=tt, cmap="viridis", s=22, zorder=2, edgecolor="k", linewidths=0.2)
        r = np.corrcoef(L, M)[0, 1]
        ax.set_title(f"{label}\nPearson r(dMdt, Lmech) = {r:.3f}", fontsize=10)
        ax.set_xlabel(r"$L_{\rm mech,total}$  [code units]")
        ax.set_ylabel(r"bubble $\dot M$  [M$_\odot$/Myr]")
        fig.colorbar(sc, ax=ax, label="t [Myr]")
    fig.suptitle(
        r"Conductive evaporation $\dot M$ vs injected power $L_{\rm mech,total}$"
        "  (colour = time; loops = lag)",
        fontsize=12,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_feedback(path):
    """Steep run: cooling_ratio + Lmech (top), Eb + Pb (bottom) vs t."""
    d = load(RUNS[0][0])
    t = d["t_now"]
    real = d["v_struct_nneg"] >= NNEG_REAL
    win = (t[real].min() - 0.04, t[real].max() + 0.04) if real.any() else None
    frac = d["Lmech_SN"] / np.where(d["Lmech_total"] > 0, d["Lmech_total"], np.nan)
    sn = np.where((t > 3.0) & (frac > 0.1))[0]
    t_sn = t[sn[0]] if len(sn) else None
    m = (t >= 2.5) & (t <= 4.0)  # window the data so axes scale to the shown range
    tw = t[m]

    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7.5), constrained_layout=True)
    a1.plot(tw, d["cooling_ratio"][m], color="#1f77b4", lw=1.8)
    a1.axhline(0.05, color="k", ls="--", lw=1.0)
    a1.text(2.52, 0.08, "transition threshold 0.05", fontsize=8)
    a1.set_ylabel(r"cooling ratio $(L_g-L_l)/L_g$", color="#1f77b4")
    a1.tick_params(axis="y", labelcolor="#1f77b4")
    a1.set_ylim(0, 1.0)
    a1b = a1.twinx()
    a1b.plot(tw, d["Lmech_total"][m] / 1e8, color="0.55", lw=1.0, ls=":")
    a1b.set_ylabel(r"$L_{\rm mech,total}$ [$10^8$ code units]", color="0.55")
    a1b.tick_params(axis="y", labelcolor="0.55")

    # colourblind-safe (Wong): Eb = bluish-green, Pb = vermillion
    a2.plot(tw, d["Eb"][m] / 1e8, color="#009E73", lw=1.8)
    a2.set_ylabel(r"$E_b$ [$10^8$ code units]", color="#009E73")
    a2.tick_params(axis="y", labelcolor="#009E73")
    a2b = a2.twinx()
    a2b.plot(tw, d["Pb"][m], color="#D55E00", lw=1.8)
    a2b.set_ylabel(r"$P_b$ [code units]", color="#D55E00")
    a2b.tick_params(axis="y", labelcolor="#D55E00")
    a2.set_xlabel("t  [Myr]")

    for ax in (a1, a2):
        if win:
            ax.axvspan(*win, color="orange", alpha=0.18, zorder=0)
        if t_sn is not None:
            ax.axvline(t_sn, color="purple", ls="--", lw=1.0)
    if t_sn is not None:
        a1.text(t_sn + 0.01, 0.92, "SN onset", color="purple", fontsize=8)
    a1.set_xlim(2.5, 4.0)
    fig.suptitle(
        "Feedback response (steep 1e6, α=−2): the WR surge drives Eb up + a local Pb rise (β<0)\n"
        "and resets the cooling ratio upward — it never reaches the 0.05 transition (stall)",
        fontsize=11,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_causal_ladder(path):
    """Steep run: the measured drivers vs the (conjectural) inflow, 4 panels.

    beta = -(t/Pb) dPb/dt (beta<0 = Pb rising); delta = (t/T) dT/dt (delta>0 =
    T rising). The dv/dr source is (beta+delta)/t = -d ln(n)/dt, so beta+delta<0
    is FORMALLY a compression term. Panels 1-3 (Lmech, Eb/Pb, beta/delta) are
    measured and solid; the inflow in panel 4 is the quasi-steady-ansatz output
    -- subsonic (Mach~2e-3), energetically negligible (~1e-6 of thermal), and its
    physical reality (real transient vs ansatz artefact) is OPEN. See
    analysis/stalling-energy-phase.md "Is the inflow physical?".
    """
    d = load(RUNS[0][0])  # steep
    t = d["t_now"]
    m = (t >= 2.85) & (t <= 3.5)  # the WR-surge -> inflow window
    tw = t[m]
    real = d["v_struct_nneg"] >= NNEG_REAL
    win = (t[real].min() - 0.02, t[real].max() + 0.02) if real.any() else None

    fig, (a1, a2, a3, a4) = plt.subplots(4, 1, sharex=True, figsize=(9.5, 11))

    a1.plot(tw, d["Lmech_total"][m] / 1e8, "k-", lw=2, label=r"$L_{\rm mech,total}$")
    a1.plot(
        tw, d["Lmech_W"][m] / 1e8, color="#0072B2", lw=1.3, ls="--", label=r"$L_{\rm mech,W}$ (WR)"
    )
    a1.set_ylabel(r"$L_{\rm mech}$ [$10^8$]")
    a1.set_title("① CAUSE: feedback power surge (WR wind ramp)", fontsize=10, loc="left")
    a1.legend(fontsize=8, loc="upper left")

    a2.plot(tw, d["Eb"][m] / 1e8, color="#009E73", lw=2)
    a2.set_ylabel(r"$E_b$ [$10^8$]", color="#009E73")
    a2.tick_params(axis="y", labelcolor="#009E73")
    a2b = a2.twinx()
    a2b.plot(tw, d["Pb"][m], color="#D55E00", lw=2)
    a2b.set_ylabel(r"$P_b$", color="#D55E00")
    a2b.tick_params(axis="y", labelcolor="#D55E00")
    a2.set_title(
        "② MECHANISM: bubble re-pressurises — Eb climbs, Pb bumps up", fontsize=10, loc="left"
    )

    a3.plot(tw, d["cool_beta"][m], color="#0072B2", lw=1.6, label=r"$\beta$ (<0 = Pb rising)")
    a3.plot(tw, d["cool_delta"][m], color="#009E73", lw=1.6, label=r"$\delta$ (>0 = T rising)")
    a3.plot(
        tw,
        d["beta_plus_delta"][m],
        color="#D55E00",
        lw=2.6,
        label=r"$\beta+\delta = -t\,d\ln n/dt$",
    )
    a3.axhline(-0.4, color="k", ls="--", lw=1.0)
    a3.axhline(0.0, color="0.6", ls=":", lw=0.8)
    a3.text(2.86, -0.45, "inflow trigger ~−0.4", color="k", fontsize=8, va="top")
    a3.set_ylabel(r"$\beta$,  $\delta$,  $\beta+\delta$")
    a3.set_title(
        "③ TRIGGER: β dive (Pb-rate) outweighs δ (T-rate) → β+δ<−0.4 (formal compression term)",
        fontsize=9.5,
        loc="left",
    )
    a3.legend(fontsize=8, loc="lower left")

    a4.plot(tw, d["v_struct_min"][m], color="#b30000", lw=2, marker=".", ms=5)
    a4.axhline(0.0, color="k", lw=0.8)
    a4.set_ylabel("v_struct_min\n[pc/Myr]")
    a4.set_title(
        "④ CONSEQUENCE (ansatz output): inner inflow — subsonic, ~1e-6 of thermal; "
        "likely artefact, real-vs-artefact OPEN",
        fontsize=9,
        loc="left",
    )
    a4.set_xlabel("t  [Myr]")

    for ax in (a1, a2, a3, a4):
        if win:
            ax.axvspan(*win, color="orange", alpha=0.15, zorder=0)
    a1.set_xlim(2.85, 3.5)
    fig.suptitle(
        "Negative-velocity chain (steep 1e6, α=−2): the measured drivers "
        "(①–③ Lmech↑→Eb/Pb↑→β+δ↓) are solid;\nthe inflow (④) is a likely-artefact, "
        "energetically negligible structure-ansatz output",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_trigger(HERE / "negvel_trigger.png")
    plot_timeline(HERE / "negvel_timeline.png")
    plot_dmdt_lmech(HERE / "negvel_dmdt_lmech.png")
    plot_feedback(HERE / "negvel_feedback.png")
    plot_causal_ladder(HERE / "negvel_causal.png")
    print(
        "wrote negvel_trigger.png, negvel_timeline.png, negvel_dmdt_lmech.png, "
        "negvel_feedback.png, negvel_causal.png"
    )


if __name__ == "__main__":
    main()
