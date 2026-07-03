#!/usr/bin/env python3
"""theta5b analysis — the fine f_mix bracket + long diffuse arms, combined with theta5.

Reads runs/data/theta5_summary.csv + runs/data/theta5b_summary.csv (no sims) and produces the
referee-defense measurements (PLAN "REFEREE DEFENSE"; FINDINGS §11):

  1. data/theta5_fire_map.csv — per (config, f_mix) outcome: FIRED (cooling_balance),
     DRAIN (reached momentum WITHOUT firing — the Eb<=0 handoff won the fire-vs-drain race),
     NOFIRE (healthy to stop_t, stayed energy-driven), NAN (non-finite loss rows), + theta_max.
  2. data/theta5_law_check.csv — the theta5-fit collapse law f_fire = 1.4*(0.95/theta0)^1.82
     predicting the theta5b fine-measured f_fire OUT-OF-SAMPLE, per config + rms.
  3. theta5b_fire_map.png — the sensitivity matrix (config x f_mix, outcome-colored):
     the "why exactly 4" figure. The whole-band window is read off directly.
  4. theta5b_law_check.png — predicted vs measured f_fire with the rms band.

REPRODUCE:
    python docs/dev/transition/pdv-trigger/data/make_theta5b_analysis.py
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
# theta5-fit collapse law (FINDINGS §10) — used OUT-OF-SAMPLE on the theta5b arms
LAW_C, LAW_S = 0.142, 1.824

ORDER = [
    "fail_repro",
    "small_1e6",
    "pl2_steep",
    "be_sphere",
    "large_diffuse_lowsfe",
    "midrange_pl0",
    "small_dense_highsfe",
    "simple_cluster",
    "normal_n1e3",
]  # by theta0, outliers first; normal_n1e3 (theta5n, 2026-07-03) fires natively
THETA0 = {
    "simple_cluster": 0.676,
    "small_dense_highsfe": 0.717,
    "midrange_pl0": 0.636,
    "large_diffuse_lowsfe": 0.535,
    "be_sphere": 0.529,
    "pl2_steep": 0.511,
    "small_1e6": 0.297,
    "fail_repro": 0.003,
    "normal_n1e3": 1.047,
}
SHORT = {"large_diffuse_lowsfe": "large_diffuse", "small_dense_highsfe": "small_dense"}

OUTCOLOR = {"FIRED": "#0072B2", "DRAIN": "#E69F00", "NOFIRE": "#009E73", "NAN": "#CC79A7"}
OUTMARK = {"FIRED": "o", "DRAIN": "v", "NOFIRE": "s", "NAN": "x"}
OUTLABEL = {
    "FIRED": "fires cooling_balance ($\\theta\\geq0.95$)",
    "DRAIN": "momentum WITHOUT firing ($E_b\\leq0$ drain won the race)",
    "NOFIRE": "stays energy-driven (healthy to stop_t)",
    "NAN": "solve never succeeds: $L_{\\rm loss}$ stays at its NaN default (root at domain edge)",
}


def f_of_mode(mode):
    if mode == "none":
        return 1.0
    return float(mode[4:].replace("p", ".")) if mode.startswith("mult") else None


def read_summary(path):
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.lstrip().startswith("#")))


def classify(r):
    try:
        th = float(r["theta_max"])
    except (ValueError, TypeError):
        th = float("nan")
    if not math.isfinite(th):
        return "NAN", None
    if r["fired_cooling_balance"] == "True":
        return "FIRED", th
    if r["reached_momentum"] == "True":
        return "DRAIN", th
    return "NOFIRE", th


def load():
    rows = read_summary(os.path.join(_PDV, "runs", "data", "theta5_summary.csv"))
    rows += read_summary(os.path.join(_PDV, "runs", "data", "theta5b_summary.csv"))
    rows += read_summary(os.path.join(_PDV, "runs", "data", "theta5n_summary.csv"))
    cells, t8 = {}, {}
    for r in rows:
        cfg, mode = r["run_name"].rsplit("__", 1)
        f = f_of_mode(mode)
        if f is None:
            continue
        out, th = classify(r)
        if cfg.endswith("_t8"):
            t8[(cfg[:-3], f)] = (out, th, r)
        else:
            cells[(cfg, f)] = (out, th, r)
    return cells, t8


def write_fire_map(cells, t8):
    fs = sorted({f for _, f in cells})
    path = os.path.join(_HERE, "theta5_fire_map.csv")
    stamp_line = stamp(__file__)
    with open(path, "w", newline="") as fh:
        fh.write(stamp_line + "\n")
        w = csv.writer(fh)
        w.writerow(["config"] + [f"f{f:g}" for f in fs] + ["f8Myr_arms"])
        for cfg in ORDER:
            row = [cfg]
            for f in fs:
                out, th, _ = cells.get((cfg, f), (None, None, None))
                row.append(f"{out}:{th:.3f}" if out and th is not None else (out or ""))
            extra = "; ".join(
                f"f{f:g}@8Myr={out}:{th:.3f}"
                for (c, f), (out, th, _) in sorted(t8.items())
                if c == cfg
            )
            row.append(extra)
            w.writerow(row)
    print(f"wrote {path}")
    return fs


def law_check(cells):
    """Out-of-sample: smallest firing f per config vs the theta5-law prediction."""
    out = []
    for cfg in ORDER:
        th0 = THETA0[cfg]
        fired = sorted(f for (c, f), (o, _, _) in cells.items() if c == cfg and o == "FIRED")
        f_meas = fired[0] if fired else None
        # theta0 > trigger (normal_n1e3 fires natively): the law still predicts, f_fire ~ 1.4*(0.95/th0)^1.82
        pred = 10 ** (LAW_C + LAW_S * math.log10(TRIGGER / th0))
        resid = (
            math.log10(pred / f_meas)
            if pred and f_meas and cfg not in ("fail_repro", "small_1e6")
            else None
        )
        out.append(
            {
                "config": cfg,
                "theta0": th0,
                "f_fire_measured": f_meas,
                "f_fire_predicted": round(pred, 2) if pred else None,
                "resid_dex": round(resid, 4) if resid is not None else None,
            }
        )
    resids = [r["resid_dex"] for r in out if r["resid_dex"] is not None]
    rms = math.sqrt(sum(x * x for x in resids) / len(resids))
    path = os.path.join(_HERE, "theta5_law_check.csv")
    stamp_line = stamp(__file__)
    with open(path, "w", newline="") as fh:
        fh.write(stamp_line + "\n")
        fh.write(
            f"# out-of-sample check of f_fire = 1.4*(0.95/theta0)^1.82 (fit on the coarse "
            f"theta5 grid) against the theta5b fine bracket: rms = {rms:.4f} dex over "
            f"{len(resids)} configs\n"
        )
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path}  (rms {rms:.3f} dex)")
    return out, rms


def fig_fire_map(cells, t8, fs):
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
    # the 8 Myr diffuse arms, drawn just above the diffuse row
    for (cfg, f), (out, th, _) in t8.items():
        y = ys[cfg] + 0.33
        ax.plot(
            f,
            y,
            marker=OUTMARK[out],
            color=OUTCOLOR[out],
            ms=6.5,
            mfc=OUTCOLOR[out] if out == "FIRED" else "white",
            mew=1.3,
            zorder=4,
            alpha=0.85,
        )
    ax.annotate(
        "8 Myr arms",
        (1.07, ys["large_diffuse_lowsfe"] + 0.36),
        fontsize=7,
        color="0.4",
        ha="left",
        va="center",
    )

    # the measured whole-band window: every normal-GMC config fires
    ax.axvspan(3.75, 4.75, color="#0072B2", alpha=0.09, zorder=0)
    ax.text(
        4.2,
        -0.55,
        "whole-band window [4, 4.5]",
        fontsize=8.5,
        color="#0072B2",
        ha="center",
        va="center",
    )
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 8])
    ax.set_xticklabels(["1", "2", "2.5", "3", "3.5", "4", "4.5", "5", "8"], fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([SHORT.get(c, c) for c in ORDER])
    ax.set_ylim(-0.95, len(ORDER) - 0.1)
    ax.set_xlim(0.9, 9.5)
    ax.set_xlabel("$f_{\\rm mix}$")
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
        "theta5 + theta5b fire map: outcome per (config, $f_{\\rm mix}$), 5 Myr\n"
        "numbers = $\\theta_{\\max}$ at fire; the fire set is NOT monotonic in $f$ "
        "(fire-vs-drain race)"
    )
    fig.savefig(os.path.join(_PDV, "theta5b_fire_map.png"), bbox_inches="tight")
    plt.close(fig)


def fig_law_check(law, rms):
    fig, ax = plt.subplots(figsize=(4.9, 4.6))
    lo, hi = 0.85, 6
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color="0.4", lw=1.2)
    ax.fill_between(
        xs, xs * 10**-rms, xs * 10**rms, color="0.85", zorder=0, label=f"rms {rms:.3f} dex"
    )
    for r in law:
        if r["resid_dex"] is None:
            continue
        ax.plot(r["f_fire_measured"], r["f_fire_predicted"], "o", color="#0072B2", ms=8, zorder=4)
        dy = -11 if r["config"] in ("be_sphere", "midrange_pl0") else 6
        ax.annotate(
            SHORT.get(r["config"], r["config"]),
            (r["f_fire_measured"], r["f_fire_predicted"]),
            textcoords="offset points",
            xytext=(6, dy),
            fontsize=8,
            color="0.15",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [1, 1.5, 2, 2.5, 3, 4, 5]
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(plt.FixedLocator(ticks))
        a.set_major_formatter(plt.FixedFormatter([str(t) for t in ticks]))
        a.set_minor_locator(plt.NullLocator())
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured $f_{\\rm fire}$ (theta5b fine bracket)")
    ax.set_ylabel("predicted $f_{\\rm fire} = 1.4\\,(0.95/\\theta_0)^{1.82}$")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        "The collapse law predicts the fine bracket\nout-of-sample (one parameter: $\\theta_0$)"
    )
    fig.savefig(os.path.join(_PDV, "theta5b_law_check.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    cells, t8 = load()
    fs = write_fire_map(cells, t8)
    law, rms = law_check(cells)
    fig_fire_map(cells, t8, fs)
    fig_law_check(law, rms)
    for r in law:
        print(r)
    print("wrote theta5b_fire_map.png, theta5b_law_check.png ->", _PDV)


if __name__ == "__main__":
    main()
