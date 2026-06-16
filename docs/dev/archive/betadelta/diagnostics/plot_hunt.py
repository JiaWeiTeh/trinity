#!/usr/bin/env python3
"""Phase 6.0 velocity-contamination hunt — visuals over the 6 hunt configs.

Reads the canonical hunt CSVs from docs/dev/data/hunt_*.csv. The G6 classification
itself is in docs/dev/archive/betadelta/velstruct/analyze_hunt.py; this script makes the plots the
writeup (docs/dev/archive/betadelta/stalling-energy-phase.md, Phase 6.0) calls for.

  ⚠️ Use v_neg_frac_thick (band / bubble thickness), NOT raw v_struct_nneg:
  the hunt harness reads the full ~6e4-point bubble_v_arr, so nneg is a huge raw
  count, not the old "of 100". Real inflow = frac>0.02 or v_min<-0.01 (matching
  analyze_hunt's REAL_FRAC / V_FLOOR); below that is the inner-BC artifact.

Produces:
  - hunt_trigger.png   : v_neg_frac_thick vs beta+delta, all 6 configs
  - hunt_massdep.png   : min(beta+delta) & max band-frac vs cluster mass
                         (finding 1 -- weakest feedback gives the DEEPEST inflow)
  - hunt_dmdt_leads.png: h1 surge walk -- the dMdt jump leads the inflow

Usage: python docs/dev/archive/betadelta/diagnostics/plot_hunt.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "analysis" / "data"  # canonical hunt CSVs (was scratch dupes)
REAL_FRAC, V_FLOOR = 0.02, 0.01  # real inflow vs inner-BC artifact (analyze_hunt)
BPD_OLD = -0.5  # the old steep-baseline threshold

# file, label, cluster mass [Msun], profile, marker, colour, note
CONFIGS = [
    ("hunt_h1_steep_base.csv", "h1 base sfe.01", 1e4, "steep", "o", "#d62728", ""),
    ("hunt_h5_steep_long.csv", "h5 long sfe.03", 3e4, "steep", "o", "#ff7f0e", ""),
    ("hunt_h2_steep_sfe10.csv", "h2 sfe.10", 1e5, "steep", "o", "#2ca02c", ""),
    ("hunt_h4_steep_dense.csv", "h4 dense sfe.10", 1e5, "dense", "X", "#9467bd", "handoff"),
    ("hunt_h3_steep_sfe30.csv", "h3 sfe.30", 3e5, "steep", "o", "#1f77b4", ""),
    ("hunt_h6_flat_sfe30.csv", "h6 flat sfe.30", 3e5, "flat", "s", "#17becf", "flat ctrl"),
]
COLS = (
    "t_now beta_plus_delta cool_beta v_struct_min v_neg_frac_thick "
    "bubble_dMdt Lmech_W cooling_ratio"
).split()

_STYLE = HERE.parents[1] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams["text.usetex"] = False


def load(fn):
    rows = list(csv.DictReader(open(DATA / fn)))

    def col(k):
        out = []
        for r in rows:
            try:
                out.append(float(r[k]))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out)

    return {k: col(k) for k in COLS}


def real_mask(d):
    frac = np.nan_to_num(d["v_neg_frac_thick"])
    vmin = np.nan_to_num(d["v_struct_min"])
    return (frac > REAL_FRAC) | (vmin < -V_FLOOR)


def plot_trigger(path):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for fn, label, _m, _prof, mk, c, note in CONFIGS:
        d = load(fn)
        frac = np.nan_to_num(d["v_neg_frac_thick"])
        bpd = d["beta_plus_delta"]
        real = real_mask(d)
        ax.scatter(bpd[~real], frac[~real], s=6, color="0.86", zorder=1)  # no real inflow
        lab = label + (f" ({note})" if note else "")
        ax.scatter(
            bpd[real],
            frac[real],
            s=45,
            marker=mk,
            facecolor=c,
            edgecolor="k",
            linewidths=0.4,
            zorder=3,
            label=lab,
        )
    ax.axvline(BPD_OLD, ls="--", color="r", lw=1.2)
    ax.text(BPD_OLD - 0.03, 0.7, "old −0.5\nthreshold", color="r", ha="right", fontsize=8)
    ax.axvspan(-0.45, -0.35, color="orange", alpha=0.10)
    ax.text(
        -0.40,
        0.02,
        "inflow onset ~−0.4\n(excl. h4 handoff)",
        color="#cc6600",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.set_xlabel(r"$\beta+\delta$")
    ax.set_ylabel("v_neg_frac_thick   (inflow band / bubble thickness)")
    ax.set_title(
        "Inflow band vs β+δ across 6 hunt configs — onset near β+δ ≈ −0.4 (softer than −0.5),\n"
        "and grows as β+δ goes more negative (deepest in the weakest-feedback h1)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left", title="real inflow (frac>0.02)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_massdep(path):
    stats = []
    for fn, label, mass, prof, mk, c, note in CONFIGS:
        d = load(fn)
        real = real_mask(d)
        frac = np.nan_to_num(d["v_neg_frac_thick"])
        stats.append(
            dict(
                label=label,
                mass=mass,
                prof=prof,
                mk=mk,
                c=c,
                note=note,
                bpd_min=np.nanmin(d["beta_plus_delta"]),
                max_frac=(frac[real].max() if real.any() else 0.0),
            )
        )

    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7.5))
    for s in stats:
        a1.scatter(
            s["mass"],
            s["bpd_min"],
            s=110,
            marker=s["mk"],
            facecolor=s["c"],
            edgecolor="k",
            zorder=3,
        )
        a2.scatter(
            s["mass"],
            s["max_frac"],
            s=110,
            marker=s["mk"],
            facecolor=s["c"],
            edgecolor="k",
            zorder=3,
            label=s["label"] + (f" ({s['note']})" if s["note"] else ""),
        )
        a1.annotate(
            s["label"].split()[0],
            (s["mass"], s["bpd_min"]),
            fontsize=7,
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
        )
    a1.axhline(BPD_OLD, ls=":", color="r", lw=1.0)
    a1.axhline(0.0, ls=":", color="0.6", lw=0.8)
    a1.set_xscale("log")
    a1.set_ylabel(r"min $(\beta+\delta)$ reached")
    a1.set_title(
        "Deepest inflow is the WEAKEST-feedback h1 (cluster 1e4) — not the strongest;\n"
        "'stronger surge → worse inflow' is falsified (no monotonic trend; h5 stays positive → zero)",
        fontsize=9.5,
    )
    a2.set_ylabel("max band frac\n(real inflow)")
    a2.set_xlabel(r"cluster mass  $M_\star = $ sfe $\times M_{\rm cloud}$  [M$_\odot$]  (log)")
    a2.legend(fontsize=7.5, loc="upper right", ncol=2)
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_dmdt_leads(path):
    d = load("hunt_h1_steep_base.csv")
    t = d["t_now"]
    m = (t >= 2.92) & (t <= 3.36)
    tw = t[m]
    dm = d["bubble_dMdt"]
    step = np.full_like(dm, np.nan)
    step[1:] = (dm[1:] / dm[:-1] - 1.0) * 100.0
    stepw = step[m]
    bpdw = d["beta_plus_delta"][m]
    vminw = np.nan_to_num(d["v_struct_min"][m])
    onset = tw[vminw < -V_FLOOR].min()  # first real inflow

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(tw, stepw, width=0.035, color="#ff7f0e", alpha=0.85, label="dMdt step per segment [%]")
    ax.axvspan(onset, tw.max(), color="red", alpha=0.08)
    ax.axvline(onset, color="red", ls="--", lw=1.2)
    ax.text(onset + 0.004, 60, "inflow region\n(v<0)", color="#b30000", fontsize=9, va="top")
    ax.set_ylabel("dMdt step per segment  [%]", color="#cc6600")
    ax.tick_params(axis="y", labelcolor="#cc6600")
    ax.set_xlabel("t  [Myr]")
    ax.set_ylim(-10, 70)

    axb = ax.twinx()
    axb.plot(tw, bpdw, "-o", color="#2ca02c", ms=4, label=r"$\beta+\delta$")
    axb.axhline(0.0, color="0.6", ls=":", lw=0.8)
    axb.axhline(BPD_OLD, color="r", ls=":", lw=0.8)
    axb.set_ylabel(r"$\beta+\delta$", color="#2ca02c")
    axb.tick_params(axis="y", labelcolor="#2ca02c")
    axb.annotate(
        "biggest dMdt jumps (+42%, +62%)\nhappen BEFORE inflow (β+δ still > 0)",
        xy=(3.128, 0.41),
        xytext=(2.95, -1.0),
        arrowprops=dict(arrowstyle="->", color="k"),
        fontsize=8.5,
    )
    ax.set_title(
        "h1 WR surge: the dMdt kink LEADS the inflow — it's the feedback surge, not the band",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    plot_trigger(HERE / "hunt_trigger.png")
    plot_massdep(HERE / "hunt_massdep.png")
    plot_dmdt_leads(HERE / "hunt_dmdt_leads.png")
    print("wrote hunt_trigger.png, hunt_massdep.png, hunt_dmdt_leads.png")


if __name__ == "__main__":
    main()
