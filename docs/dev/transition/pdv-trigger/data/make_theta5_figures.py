#!/usr/bin/env python3
"""Publication figures for the theta5 result (FINDINGS §10) + the f_mix candidate scorecard.

Reads ONLY committed CSVs (no sims): runs/data/theta5_{summary,calibration}.csv,
data/fkappa_functional_form.csv (the retired blowout-θ₀ anchors, for the metric-correction
figure), data/summary.csv (the 819-run kappa sweep, for the knob-choice panel). El-Badry's
closed form is computed analytically (Eq 37/38, A_mix=3.5, λδv=3 — ELBADRY_REFERENCE.md).

Figures (workstream root):
  theta5_arms.png              F1 emergent θ_max vs f_mix, all 8 configs, outcome-classed
  theta5_collapse_law.png      F2 the multiplier θ₁-collapse law (+ kappa's for contrast)
  theta5_metric_correction.png F3 blowout-θ₀ → 5 Myr θ_max (the 📏 rule-2 correction)
  theta5_target_vs_emergent.png F4 El-Badry target vs native and f=4 emergent θ(n)
  theta5_knob_choice.png       F5 kappa breakdown windows vs multiplier monotonicity
Also writes runs/data/theta5_fmix_scorecard.csv — the quantitative per-config margin table
for the f_mix pin. f_mix=4 was ADOPTED 2026-07-02 (maintainer ruling: firing into momentum
and then recollapsing is acceptable physics — PLAN ledger); theta5b refines the window.

REPRODUCE:
    python docs/dev/transition/pdv-trigger/data/make_theta5_figures.py
"""

import csv
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PDV)
from _stamp import stamp  # noqa: E402
from _trinity_style import use_trinity_style  # noqa: E402

use_trinity_style()
import matplotlib.pyplot as plt  # noqa: E402

TRIGGER = 0.95
BAND = (0.90, 0.99)  # Lancaster efficiently-cooled band
FS = [1.0, 2.0, 4.0, 8.0]

# outcome classes: color = the class (validated CVD-safe set), marker = secondary encoding
CLS = {
    "survive": dict(color="#0072B2", marker="o", label="fires & survives to 5 Myr"),
    "recollapse": dict(color="#E69F00", marker="^", label="fires, then shell-collapses"),
    "route_a": dict(color="#009E73", marker="s", label="never fires (route-a)"),
    "pdv": dict(color="#CC79A7", marker="D", label="PdV / Eb$\\leq$0 handoff regime"),
}
CONFIG_CLASS = {
    "simple_cluster": "survive",
    "large_diffuse_lowsfe": "survive",
    "pl2_steep": "recollapse",
    "midrange_pl0": "recollapse",
    "be_sphere": "recollapse",
    "small_1e6": "route_a",
    "fail_repro": "pdv",
    "small_dense_highsfe": "pdv",
}
SHORT = {
    "simple_cluster": "simple_cluster",
    "large_diffuse_lowsfe": "large_diffuse",
    "pl2_steep": "pl2_steep",
    "midrange_pl0": "midrange_pl0",
    "be_sphere": "be_sphere",
    "small_1e6": "small_1e6",
    "fail_repro": "fail_repro",
    "small_dense_highsfe": "small_dense",
}


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.lstrip().startswith("#")))


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def theta_eb(n, lamdv=3.0, a_mix=3.5):
    psi = a_mix * np.sqrt(lamdv * np.asarray(n, dtype=float))
    return psi / (11.0 / 5.0 + psi)


def load():
    summary = read_csv(os.path.join(_PDV, "runs", "data", "theta5_summary.csv"))
    calib = read_csv(os.path.join(_PDV, "runs", "data", "theta5_calibration.csv"))
    runs = {}
    for r in summary:
        cfg, mode = r["run_name"].rsplit("__", 1)
        f = {"none": 1.0, "mult2": 2.0, "mult4": 4.0, "mult8": 8.0}[mode]
        runs.setdefault(cfg, {})[f] = r
    cal = {r["config"]: r for r in calib}
    return runs, cal


# --------------------------------------------------------------------------- F1
def fig_arms(runs):
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    ax.axhspan(*BAND, color="0.88", zorder=0)
    ax.axhline(TRIGGER, color="0.35", lw=1.0, ls="--", zorder=1)
    ax.text(
        0.985,
        TRIGGER + 0.012,
        "trigger $\\theta=0.95$",
        va="bottom",
        ha="left",
        fontsize=8.5,
        color="0.35",
    )
    ax.text(
        0.985,
        BAND[0] - 0.012,
        "Lancaster band 0.90–0.99",
        va="top",
        ha="left",
        fontsize=8.5,
        color="0.45",
    )

    # right-margin label ladder for the five band-converging configs (leader lines);
    # the three outliers are labeled in-plot where they have room
    ladder = {
        "pl2_steep": 1.09,
        "simple_cluster": 1.02,
        "be_sphere": 0.95,
        "large_diffuse_lowsfe": 0.88,
        "midrange_pl0": 0.81,
    }
    ends = {}
    for cfg, arms in runs.items():
        c = CLS[CONFIG_CLASS[cfg]]
        xs, ys, fired = [], [], []
        for f in FS:
            r = arms.get(f)
            if r is None:
                continue
            th = num(r["theta_max"])
            if th is None:  # NaN loss rows (dense edge): mark near the bottom
                ax.plot(f, 0.10, marker="x", color=c["color"], ms=7, mew=1.6, zorder=4)
                ax.annotate(
                    "NaN",
                    (f, 0.10),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                    color=c["color"],
                )
                continue
            xs.append(f)
            ys.append(th)
            fired.append(r["fired_cooling_balance"] == "True")
        ax.plot(xs, ys, color=c["color"], lw=1.6, zorder=3, alpha=0.9)
        for x, y, fi in zip(xs, ys, fired):
            ax.plot(
                x,
                y,
                marker=c["marker"],
                color=c["color"],
                ms=7.5,
                mfc=c["color"] if fi else "white",
                mew=1.4,
                zorder=4,
            )
        ends[cfg] = (xs[-1], ys[-1])

    for cfg, y_lab in ladder.items():
        x_end, y_end = ends[cfg]
        ax.plot([x_end * 1.06, 9.3], [y_end, y_lab], color="0.75", lw=0.7, zorder=2)
        ax.text(
            9.6,
            y_lab,
            SHORT[cfg],
            fontsize=8.5,
            color="0.15",
            va="center",
            bbox=dict(fc="white", ec="none", pad=1.2),
            zorder=5,
        )
    ax.annotate(
        SHORT["small_1e6"],
        (4.0, num(runs["small_1e6"][4.0]["theta_max"])),
        textcoords="offset points",
        xytext=(6, -13),
        fontsize=8.5,
        color="0.15",
    )
    ax.annotate(
        SHORT["small_dense_highsfe"],
        ends["small_dense_highsfe"],
        textcoords="offset points",
        xytext=(8, -3),
        fontsize=8.5,
        color="0.15",
    )
    ax.annotate(
        SHORT["fail_repro"],
        (1.35, 0.006),
        textcoords="offset points",
        xytext=(0, 8),
        fontsize=8.5,
        color="0.15",
    )

    ax.axvline(4, color="0.7", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(FS)
    ax.set_xticklabels(["1\n(none)", "2", "4", "8"])
    ax.set_xlim(0.92, 13.5)
    ax.set_ylim(0, 1.13)
    ax.set_xlabel("cooling boost $f_{\\rm mix}$ (multiplier on the resolved $L_{\\rm cool}$)")
    ax.set_ylabel("emergent $\\theta_{\\max} = \\max_t\\, L_{\\rm loss}/L_{\\rm mech}$  (5 Myr)")
    handles = [
        plt.Line2D([], [], color=v["color"], marker=v["marker"], lw=1.6, ms=7, label=v["label"])
        for v in CLS.values()
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            color="0.3",
            marker="o",
            lw=0,
            ms=7,
            mfc="white",
            mew=1.4,
            label="open = did not fire",
        )
    )
    ax.set_title(
        "The theta5 matrix: 8 configs $\\times$ $f_{\\rm mix}$, all to 5 Myr or physics end"
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.115),
        frameon=False,
    )
    fig.savefig(os.path.join(_PDV, "theta5_arms.png"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- F2
def fig_collapse(cal):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    lab = {
        "simple_cluster": dict(xytext=(8, 2), ha="left"),
        "midrange_pl0": dict(xytext=(-9, 8), ha="right"),
        "large_diffuse_lowsfe": dict(xytext=(-14, 16), ha="center"),
        "be_sphere": dict(xytext=(4, -22), ha="center"),
        "pl2_steep": dict(xytext=(13, 5), ha="left"),
    }
    for cfg, r in cal.items():
        th0, ff = num(r["theta0"]), num(r["f_fire"])
        if th0 is None:
            continue
        ratio = TRIGGER / th0
        if ff and ff > 1:
            f_lo = FS[FS.index(ff) - 1]  # bracket: fired at ff, not at the arm below
            ax.plot([ratio, ratio], [f_lo, ff], color="#0072B2", lw=1.2, alpha=0.55, zorder=2)
            ax.plot(ratio, ff, "o", color="#0072B2", ms=8, zorder=4)
            ax.annotate(
                SHORT[cfg],
                (ratio, ff),
                textcoords="offset points",
                fontsize=8.5,
                color="0.15",
                **lab[cfg],
            )
        elif cfg == "small_1e6":  # censored: no fire up to 8
            ax.annotate(
                "",
                xy=(ratio, 13.5),
                xytext=(ratio, 8.0),
                arrowprops=dict(arrowstyle="-|>", color="#009E73", lw=1.6),
            )
            ax.plot(ratio, 8, "s", color="#009E73", ms=7, mfc="white", mew=1.5, zorder=4)
            ax.annotate(
                "small_1e6\n(censored: $>8$)",
                (ratio, 8),
                textcoords="offset points",
                xytext=(8, -2),
                fontsize=8.5,
                color="0.15",
            )

    x = np.logspace(math.log10(1.15), math.log10(4.2), 50)
    ax.plot(
        x,
        10 ** (0.142 + 1.824 * np.log10(x)),
        color="#0072B2",
        lw=1.8,
        label="multiplier: $f_{\\rm fire}=1.4\\,(0.95/\\theta_0)^{1.82}$",
    )
    ax.plot(
        x,
        10 ** (0.041 + 3.755 * np.log10(x)),
        color="0.55",
        lw=1.5,
        ls="--",
        label="kappa knob (FINDINGS §9): slope 3.76",
    )
    ax.axhline(4, color="#E69F00", lw=1.2, ls=":")
    ax.text(4.35, 4.08, "$f_{\\rm mix}=4$", va="bottom", ha="right", fontsize=9.5, color="#B87700")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.1, 4.5)
    ax.set_ylim(1, 16)
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.set_yticklabels(["1", "2", "4", "8", "16"])
    ax.set_xticks([1.2, 1.5, 2, 3, 4])
    ax.set_xticklabels(["1.2", "1.5", "2", "3", "4"])
    ax.set_xlabel(
        "starting deficit $0.95/\\theta_0$   ($\\theta_0$ = native $\\theta_{\\max}$, 5 Myr)"
    )
    ax.set_ylabel("smallest firing boost $f_{\\rm fire}$")
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.98,
        0.03,
        "bars: grid bracket $(f_{\\rm prev}, f_{\\rm fire}]$;  fail_repro at "
        "$0.95/\\theta_0\\approx300$ (off scale — never fires radiatively)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    ax.set_title("The $\\theta_1$-collapse law is knob-specific")
    fig.savefig(os.path.join(_PDV, "theta5_collapse_law.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- F3
def fig_metric(runs):
    old = {
        r["config"]: num(r["theta0"])
        for r in read_csv(os.path.join(_HERE, "fkappa_functional_form.csv"))
    }
    rows = []
    for cfg, arms in runs.items():
        r = arms.get(1.0)
        if cfg in old and old[cfg] is not None and r is not None:
            rows.append(
                (
                    cfg,
                    old[cfg],
                    num(r["theta_max"]),
                    num(r["t_at_theta_max"]),
                    float(r["run_name"] and 0) or 0,
                )
            )
    rows.sort(key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ys = np.arange(len(rows))
    for y, (cfg, th_b, th_m, t_pk, _) in zip(ys, rows):
        ax.plot([th_b, th_m], [y, y], color="0.75", lw=2.2, zorder=2)
        ax.plot(th_b, y, "o", color="0.55", ms=8, mfc="white", mew=1.6, zorder=3)
        ax.plot(th_m, y, "o", color="#0072B2", ms=8.5, zorder=4)
        ax.annotate(
            f"peak at t={t_pk:.2g} Myr",
            (th_m, y),
            textcoords="offset points",
            xytext=(8, -3.5),
            fontsize=8.5,
            color="0.35",
        )
    ax.axvline(TRIGGER, color="0.35", lw=1.0, ls="--")
    ax.text(
        TRIGGER - 0.008,
        -0.42,
        "trigger 0.95",
        rotation=90,
        fontsize=8,
        color="0.35",
        ha="right",
        va="bottom",
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([SHORT[r[0]] for r in rows])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel("native $\\theta$ at $f=1$")
    hs = [
        plt.Line2D(
            [],
            [],
            marker="o",
            lw=0,
            ms=8,
            color="0.55",
            mfc="white",
            mew=1.6,
            label="$\\theta$ at blowout (retired metric)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            lw=0,
            ms=8.5,
            color="#0072B2",
            label="$\\theta_{\\max}$ over 5 Myr (standing rule 2)",
        ),
    ]
    ax.legend(handles=hs, loc="upper left", fontsize=9)
    ax.set_title(
        "Why blowout-$\\theta$ was retired: it under-reads the peak,\nworst where the peak is latest (diffuse: $2.1\\times$ at $t\\approx4.9$ Myr)"
    )
    fig.savefig(os.path.join(_PDV, "theta5_metric_correction.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- F4
def fig_target(runs, cal):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    n = np.logspace(0, 7, 200)
    ax.axhspan(*BAND, color="0.88", zorder=0)
    ax.plot(
        n,
        theta_eb(n),
        color="0.25",
        lw=1.8,
        label="El-Badry $\\theta(n)$, $\\lambda\\delta v=3$ (Eq 37/38)",
    )
    ax.axvspan(10, 1.3e7, color="#E69F00", alpha=0.06, zorder=0)
    ax.text(
        12,
        0.13,
        "El-Badry calibrated at $n=0.1$–$10\\,{\\rm cm^{-3}}$:\nGMC range is extrapolation",
        fontsize=8.5,
        color="#8a5a00",
        ha="left",
        va="bottom",
    )

    seen_n = {}
    for cfg, r in cal.items():
        nc = num(r["nCore"])
        th0 = num(r["theta0"])
        r4 = runs[cfg].get(4.0)
        th4 = num(r4["theta_max"]) if r4 is not None else None
        if nc is None or th0 is None:
            continue
        # de-conflation: multiple configs share nCore -> deterministic x offset
        k = seen_n.get(nc, 0)
        seen_n[nc] = k + 1
        xx = nc * (1.35**k)
        c = CLS[CONFIG_CLASS[cfg]]
        ax.plot(
            xx, th0, marker=c["marker"], color=c["color"], ms=7.5, mfc="white", mew=1.5, zorder=4
        )
        if th4 is not None:
            fired4 = runs[cfg][4.0]["fired_cooling_balance"] == "True"
            ax.annotate(
                "",
                xy=(xx, th4),
                xytext=(xx, th0),
                arrowprops=dict(arrowstyle="-|>", color=c["color"], lw=1.3, shrinkA=4, shrinkB=4),
            )
            ax.plot(
                xx,
                th4,
                marker=c["marker"],
                color=c["color"],
                ms=8,
                mfc=c["color"] if fired4 else "white",
                mew=1.5,
                zorder=4,
            )
        dy = -14 if cfg in ("small_1e6", "pl2_steep", "be_sphere") else 7
        ax.annotate(
            SHORT[cfg],
            (xx, th0),
            textcoords="offset points",
            xytext=(0, dy),
            ha="center",
            fontsize=8,
            color="0.15",
        )

    ax.set_xscale("log")
    ax.set_xlim(1, 1.3e7)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(
        "core density $n_{\\rm core}\\ ({\\rm cm^{-3}})$  (proxy; El-Badry's $n$ is ambient)"
    )
    ax.set_ylabel("$\\theta_{\\max}$ (5 Myr)")
    hs = [
        plt.Line2D([], [], color="0.25", lw=1.8, label="El-Badry target ($\\lambda\\delta v=3$)"),
        plt.Line2D(
            [],
            [],
            marker="o",
            lw=0,
            ms=7.5,
            color="0.3",
            mfc="white",
            mew=1.5,
            label="native ($f=1$)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            lw=0,
            ms=8,
            color="0.3",
            label="boosted ($f_{\\rm mix}=4$; filled = fired)",
        ),
    ]
    ax.legend(handles=hs, loc="lower right", fontsize=8.5)
    ax.set_title(
        "Calibrate, don't enforce: the $f_{\\rm mix}=4$ boost lifts emergent $\\theta$\ninto the band for GMCs; $\\theta_0$ is NOT a function of $n$ alone"
    )
    fig.savefig(os.path.join(_PDV, "theta5_target_vs_emergent.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- F5
def fig_knob(runs):
    cell = [
        r
        for r in read_csv(os.path.join(_HERE, "summary.csv"))
        if abs(num(r["mCloud"]) - 1e5) < 1
        and abs(num(r["sfe"]) - 0.3) < 1e-9
        and abs(num(r["nCore"]) - 1e5) < 1
    ]
    cell.sort(key=lambda r: num(r["cooling_boost_kappa"]))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), sharey=True)

    ax = axes[0]
    n_ann = 0
    for r in cell:
        f, th = num(r["cooling_boost_kappa"]), num(r["theta_max"]) or 0
        fired = r["cooling_fired"] == "True"
        froze = (not fired) and r["phase_final"] == "implicit" and num(r["t_final"]) < 1.9
        th_c = min(th, 1.28)
        if fired:
            ax.plot(f, th_c, "o", color="#0072B2", ms=7.5, zorder=4)
        elif froze:
            ax.plot(f, th_c, "x", color="#D55E00", ms=8, mew=2, zorder=4)
        else:
            ax.plot(f, th_c, "o", color="0.55", ms=7, mfc="white", mew=1.4, zorder=4)
        if th > 1.28:
            ax.annotate(
                f"$\\theta_{{\\max}}\\!=\\!{th:.1f}$",
                (f, 1.34 + 0.09 * (n_ann % 2)),
                fontsize=7.5,
                ha="center" if f < 50 else "right",
                color="0.35",
            )
            n_ann += 1
    ax.axhspan(*BAND, color="0.88", zorder=0)
    ax.axhline(TRIGGER, color="0.35", lw=1.0, ls="--")
    for f0, f1 in [(1.8, 3.4), (7, 13.5), (20, 28)]:
        ax.axvspan(f0, f1, color="#D55E00", alpha=0.08, zorder=0)
    ax.text(2.45, 0.10, "dead\nwindows", fontsize=8.5, color="#a34700", ha="center")
    hs = [
        plt.Line2D([], [], marker="o", lw=0, ms=7.5, color="#0072B2", label="fired"),
        plt.Line2D([], [], marker="x", lw=0, ms=8, mew=2, color="#D55E00", label="froze mid-run"),
        plt.Line2D(
            [], [], marker="o", lw=0, ms=7, color="0.55", mfc="white", mew=1.4, label="ran, no fire"
        ),
    ]
    ax.legend(handles=hs, loc="lower right", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("$f_\\kappa$ (structural, in-ODE)")
    ax.set_ylabel("$\\theta_{\\max}$")
    ax.set_title("kappa knob, cell (1e5, 0.3, 1e5):\nfiring bands interleave with freezes (§9a)")

    ax = axes[1]
    for cfg in ("simple_cluster", "midrange_pl0", "small_1e6"):
        c = CLS[CONFIG_CLASS[cfg]]
        xs, ys, fired = [], [], []
        for f in FS:
            r = runs[cfg].get(f)
            th = num(r["theta_max"]) if r else None
            if th is None:
                continue
            xs.append(f)
            ys.append(th)
            fired.append(r["fired_cooling_balance"] == "True")
        ax.plot(xs, ys, color=c["color"], lw=1.6)
        for x, y, fi in zip(xs, ys, fired):
            ax.plot(
                x,
                y,
                marker=c["marker"],
                color=c["color"],
                ms=7.5,
                mfc=c["color"] if fi else "white",
                mew=1.4,
            )
        ax.annotate(
            SHORT[cfg],
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(6, -3),
            fontsize=8.5,
            color="0.15",
        )
    ax.axhspan(*BAND, color="0.88", zorder=0)
    ax.axhline(TRIGGER, color="0.35", lw=1.0, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(FS)
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlim(0.9, 17)
    ax.set_xlabel("$f_{\\rm mix}$ (multiplier, post-solve)")
    ax.set_title("multiplier knob (theta5):\nmonotonic in $f$, no dead windows")
    axes[0].set_ylim(0, 1.52)
    fig.savefig(os.path.join(_PDV, "theta5_knob_choice.png"))
    plt.close(fig)


# --------------------------------------------------- f_mix candidate margin table
def f4_table(runs, cal):
    out = []
    for cfg, r in sorted(cal.items()):
        th0, ff = num(r["theta0"]), num(r["f_fire"])
        pred = 10 ** (0.142 + 1.824 * math.log10(TRIGGER / th0)) if th0 and th0 < TRIGGER else None
        r4 = runs[cfg].get(4.0)
        th4 = num(r4["theta_max"]) if r4 else None
        fired4 = bool(r4 and r4["fired_cooling_balance"] == "True")
        out.append(
            {
                "config": cfg,
                "theta0_5Myr": th0,
                "f_fire_bracket": (
                    f"({FS[FS.index(ff)-1]:g},{ff:g}]"
                    if ff and ff > 1
                    else ("=1" if ff == 1 else ">8" if cfg == "small_1e6" else "n/a")
                ),
                "f_fire_predicted_by_law": round(pred, 2) if pred else None,
                "theta_max_at_f4": th4,
                "fires_at_f4": fired4,
                "fate_at_f4": r4["outcome"] if r4 else None,
            }
        )
    path = os.path.join(_PDV, "runs", "data", "theta5_fmix_scorecard.csv")
    stamp_line = stamp(__file__)
    with open(path, "w", newline="") as fh:
        fh.write(stamp_line + "\n")
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path}")
    for row in out:
        print(row)


def main():
    runs, cal = load()
    fig_arms(runs)
    fig_collapse(cal)
    fig_metric(runs)
    fig_target(runs, cal)
    fig_knob(runs)
    f4_table(runs, cal)
    print("wrote 5 figures ->", _PDV)


if __name__ == "__main__":
    main()
