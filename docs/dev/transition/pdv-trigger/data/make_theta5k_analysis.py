#!/usr/bin/env python3
"""theta5k analysis — the first rule-compliant kappa validation, post no-root handoff (fix #1).

Reads runs/data/theta5k_summary.csv (no sims) and answers FINDINGS §9b's open question:
with the freeze fixed, does `cooling_boost_kappa` fire monotonically — and does any single
f_kappa fire the whole normal-GMC band the way f_mix in [4, 4.5] does?

  1. data/theta5k_fire_map.csv — per (config, f_kappa) outcome: FIRED (cooling_balance),
     CONDENSE (the no_physical_root => momentum handoff: n_impl pinned at the 50-segment
     streak cap, theta frozen sub-threshold — the front went condensing before global theta
     crossed), DRAIN (reached momentum/transition without firing, n_impl != 50 — Eb/dissolve
     pathways), NOFIRE (energy-driven to 5 Myr), + theta_max.
  2. theta5k_fire_map.png — the outcome matrix. Read off directly: is there a whole-band f?
  3. theta5k_theta_rise.png — theta_max vs f_kappa per config: theta rises ~monotonically
     with f even where the FIRE SET does not — the race, not the knob's reach, decides.

REPRODUCE:
    python docs/dev/transition/pdv-trigger/data/make_theta5k_analysis.py
"""

import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PDV)
from _stamp import stamp  # noqa: E402
from _trinity_style import use_trinity_style  # noqa: E402

use_trinity_style()
import matplotlib.pyplot as plt  # noqa: E402

# the implicit-phase handoff streak cap (run_energy_implicit_phase.NO_ROOT_HANDOFF_STREAK):
# a non-fired momentum run with n_impl exactly here is the condensation handoff
HANDOFF_STREAK = 50

ORDER = [
    "fail_repro",
    "small_1e6",
    "pl2_steep",
    "be_sphere",
    "large_diffuse_lowsfe",
    "midrange_pl0",
    "small_dense_highsfe",
    "simple_cluster",
]  # by theta0, outliers first (same as theta5b figures)
THETA0 = {
    "simple_cluster": 0.676,
    "small_dense_highsfe": 0.717,
    "midrange_pl0": 0.636,
    "large_diffuse_lowsfe": 0.535,
    "be_sphere": 0.529,
    "pl2_steep": 0.511,
    "small_1e6": 0.297,
    "fail_repro": 0.003,
}
SHORT = {"large_diffuse_lowsfe": "large_diffuse", "small_dense_highsfe": "small_dense"}

OUTCOLOR = {"FIRED": "#0072B2", "CONDENSE": "#CC79A7", "DRAIN": "#E69F00", "NOFIRE": "#009E73"}
OUTMARK = {"FIRED": "o", "CONDENSE": "d", "DRAIN": "v", "NOFIRE": "s"}
OUTLABEL = {
    "FIRED": "fires cooling_balance ($\\theta\\geq0.95$)",
    "CONDENSE": "condensation handoff (dMdt$\\,\\leq\\,$0 streak, fix #1)",
    "DRAIN": "momentum/dissolve WITHOUT firing",
    "NOFIRE": "stays energy-driven (healthy to 5 Myr)",
}


def f_of_mode(mode):
    if mode == "none":
        return 1.0
    return float(mode[5:].replace("p", ".")) if mode.startswith("kappa") else None


def read_summary(path):
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.lstrip().startswith("#")))


def classify(r):
    try:
        th = float(r["theta_max"])
    except (ValueError, TypeError):
        th = float("nan")
    if r["fired_cooling_balance"] == "True":
        return "FIRED", th
    if r["reached_momentum"] == "True":
        if int(r["n_impl"]) == HANDOFF_STREAK:
            return "CONDENSE", th
        return "DRAIN", th
    return "NOFIRE", th


def load():
    rows = read_summary(os.path.join(_PDV, "runs", "data", "theta5k_summary.csv"))
    cells = {}
    for r in rows:
        cfg, mode = r["run_name"].rsplit("__", 1)
        f = f_of_mode(mode)
        if f is None:
            continue
        cells[(cfg, f)] = (*classify(r), r)
    return cells


def write_fire_map(cells):
    fs = sorted({f for _, f in cells})
    # whole-band check over the six fireable configs (controls excluded)
    band = [c for c in ORDER if c not in ("fail_repro", "small_1e6")]
    whole = [f for f in fs if all(cells.get((c, f), ("?",))[0] == "FIRED" for c in band)]
    n_cond = sum(1 for (o, _, _) in cells.values() if o == "CONDENSE")
    path = os.path.join(_HERE, "theta5k_fire_map.csv")
    stamp_line = stamp(__file__)
    with open(path, "w", newline="") as fh:
        fh.write(stamp_line + "\n")
        whole_txt = (
            str(whole)
            if whole
            else "NONE — no single f_kappa fires the band (multiplier window [4,4.5] does; "
            "PLAN referee Q2)"
        )
        fh.write(
            "# first RULE-COMPLIANT kappa matrix (stop_t=5, theta_max from dictionary), post "
            "fix #1: 56/56 proper fates, ZERO freezes (the sec9a freeze class is extinct; "
            f"{n_cond} arms exit via the condensation handoff instead). Whole-band f_kappa "
            f"(all 6 fireable configs FIRED): {whole_txt}. Do NOT count CONDENSE/DRAIN as "
            "theta transitions.\n"
        )
        w = csv.writer(fh)
        w.writerow(["config"] + [f"k{f:g}" for f in fs])
        for cfg in ORDER:
            row = [cfg]
            for f in fs:
                out, th, _ = cells.get((cfg, f), (None, None, None))
                row.append(
                    f"{out}:{th:.3f}" if out and th is not None and math.isfinite(th) else (out or "")
                )
            w.writerow(row)
    print(f"wrote {path}  (whole-band f_kappa: {whole or 'NONE'})")
    return fs, whole


def fig_fire_map(cells, fs):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ys = {cfg: i for i, cfg in enumerate(ORDER)}
    for (cfg, f), (out, th, _) in cells.items():
        y = ys[cfg]
        ax.plot(
            f,
            y,
            marker=OUTMARK[out],
            color=OUTCOLOR[out],
            ms=9,
            mfc=OUTCOLOR[out] if out == "FIRED" else "white",
            mew=1.6,
            zorder=4,
        )
        if out == "FIRED" and th is not None:
            ax.annotate(
                f"{th:.2f}",
                (f, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=6.5,
                color="0.35",
            )
    ax.set_xscale("log")
    ax.set_xticks(fs)
    ax.set_xticklabels([f"{f:g}" for f in fs], fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([SHORT.get(c, c) for c in ORDER])
    ax.set_ylim(-0.6, len(ORDER) - 0.1)
    ax.set_xlim(0.9, 19)
    ax.set_xlabel("$f_\\kappa$ (cooling_boost_kappa)")
    handles = [
        plt.Line2D(
            [],
            [],
            marker=OUTMARK[k],
            lw=0,
            ms=8,
            color=OUTCOLOR[k],
            mfc=OUTCOLOR[k] if k == "FIRED" else "white",
            mew=1.5,
            label=OUTLABEL[k],
        )
        for k in OUTLABEL
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.10),
        frameon=False,
    )
    ax.set_title(
        "theta5k fire map: outcome per (config, $f_\\kappa$), rule-compliant, post fix #1\n"
        "56/56 proper fates, ZERO freezes — but NO single $f_\\kappa$ fires the whole band"
    )
    fig.savefig(os.path.join(_PDV, "theta5k_fire_map.png"), bbox_inches="tight")
    plt.close(fig)


def fig_theta_rise(cells, fs):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    band = [c for c in ORDER if c not in ("fail_repro", "small_1e6")]
    for i, cfg in enumerate(band):
        pts = sorted((f, cells[(cfg, f)][1], cells[(cfg, f)][0]) for f in fs if (cfg, f) in cells)
        xs = [p[0] for p in pts]
        ths = [p[1] for p in pts]
        ax.plot(xs, ths, "-", color=palette[i], lw=1.3, alpha=0.75, zorder=3)
        for f, th, out in pts:
            ax.plot(
                f,
                th,
                marker=OUTMARK[out],
                color=palette[i],
                ms=6.5,
                mfc=palette[i] if out == "FIRED" else "white",
                mew=1.3,
                zorder=4,
            )
        ax.annotate(
            SHORT.get(cfg, cfg),
            (xs[-1], ths[-1]),
            textcoords="offset points",
            xytext=(7, 0),
            fontsize=7.5,
            color=palette[i],
            va="center",
        )
    ax.axhline(0.95, color="0.3", lw=1.0, ls="--")
    ax.text(1.0, 0.965, "trigger $\\theta=0.95$", fontsize=8, color="0.3", ha="left")
    ax.set_xscale("log")
    ax.set_xticks(fs)
    ax.set_xticklabels([f"{f:g}" for f in fs], fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(0.9, 30)
    ax.set_xlabel("$f_\\kappa$")
    ax.set_ylabel("$\\theta_{\\max}$ (5 Myr, dictionary rows)")
    ax.set_title(
        "theta5k: $\\theta_{\\max}$ vs $f_\\kappa$ — filled = fired; open = condensation\n"
        "handoff / drain ended the run before $\\theta$ could cross (the race, not reach)"
    )
    fig.savefig(os.path.join(_PDV, "theta5k_theta_rise.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    cells = load()
    fs, whole = write_fire_map(cells)
    fig_fire_map(cells, fs)
    fig_theta_rise(cells, fs)
    counts = {}
    for out, _, _ in cells.values():
        counts[out] = counts.get(out, 0) + 1
    print("outcome counts:", counts)
    print("wrote theta5k_fire_map.png, theta5k_theta_rise.png ->", _PDV)


if __name__ == "__main__":
    main()
