#!/usr/bin/env python3
"""H4 PdV-drain-cap figures. Pure reads of the committed h4 CSVs — no sims re-run.
Saves PDF+PNG (dpi=150) into docs/dev/transition/pt4/figures/ with an h4_ prefix.

    python docs/dev/transition/pt4/h4_figures.py

Figures (per collapse config that has traj CSVs):
  h4_Eb_sweep_<cfg>       - Eb(t): V0 baseline + each t_window, cap-release marked.
  h4_pdvratio_sweep_<cfg> - PdV/Lmech(t): same, =1 line + release marked (stays >1?).
  h4_summary              - survived / self-sustained vs t_window across configs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TRAJ = HERE / "traj"
EVAL = HERE / "h4_eval.csv"
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False

WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

COLLAPSE = ["fail_repro", "fail_helix", "mass_1e9"]
TWINDOWS = ["1e-3", "3e-3", "1e-2", "1e-1"]
TWVAL = {"1e-3": 1e-3, "3e-3": 3e-3, "1e-2": 1e-2, "1e-1": 1e-1}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_traj(tag):
    p = TRAJ / f"h4_traj_{tag}.csv"
    if not p.exists():
        return None
    t, Eb, ratio = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            tv = _f(r["t_now"])
            if tv != tv:
                continue
            t.append(tv)
            Eb.append(_f(r["Eb"]))
            ratio.append(_f(r.get("pdv_over_lmech")))
    return {"t": t, "Eb": Eb, "ratio": ratio} if t else None


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png")


def fig_eb(cfg):
    v0 = load_traj(f"{cfg}_V0")
    if v0 is None:
        print(f"skip Eb {cfg}: no V0 traj")
        return
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(v0["t"], v0["Eb"], color="k", lw=2.2, label="V0 baseline (no cap)", zorder=5)
    for i, tw in enumerate(TWINDOWS):
        d = load_traj(f"{cfg}_PDVCAP_tw{tw}")
        if d is None:
            continue
        c = WONG[i % len(WONG)]
        ax.plot(d["t"], d["Eb"], color=c, lw=1.6, label=f"cap t_win={tw} Myr")
        ax.axvline(TWVAL[tw], color=c, ls=":", lw=1.0, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e6)
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$E_b$ [au]")
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_title(f"H4 PdV-cap: $E_b(t)$ — {cfg} (dotted = cap release)")
    ax.legend(fontsize=8, loc="best")
    save(fig, f"h4_Eb_sweep_{cfg}")


def fig_ratio(cfg):
    v0 = load_traj(f"{cfg}_V0")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    if v0 is not None:
        ax.plot(v0["t"], v0["ratio"], color="k", lw=2.2, label="V0 baseline", zorder=5)
    for i, tw in enumerate(TWINDOWS):
        d = load_traj(f"{cfg}_PDVCAP_tw{tw}")
        if d is None:
            continue
        c = WONG[i % len(WONG)]
        ax.plot(d["t"], d["ratio"], color=c, lw=1.6, label=f"cap t_win={tw} Myr")
        ax.axvline(TWVAL[tw], color=c, ls=":", lw=1.0, alpha=0.7)
    ax.axhline(1.0, color="0.4", lw=1.2, ls="--", label="PdV/Lmech = 1")
    ax.set_xscale("log")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"PdV / $L_{\rm mech}$")
    ax.set_title(f"H4 PdV-cap: PdV/$L_{{\\rm mech}}$ — {cfg} (does it stay >1 after release?)")
    ax.legend(fontsize=8, loc="best")
    save(fig, f"h4_pdvratio_sweep_{cfg}")


def fig_summary():
    if not EVAL.exists():
        print("skip summary: no eval csv")
        return
    rows = [r for r in csv.DictReader(open(EVAL)) if r["variant"] == "PDVCAP"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    markers = {"fail_repro": "o", "fail_helix": "s", "mass_1e9": "^"}
    for i, cfg in enumerate(COLLAPSE):
        xs, surv, ss = [], [], []
        for r in rows:
            if r["config"] != cfg:
                continue
            tw = _f(r["t_window"])
            xs.append(tw)
            surv.append(1 if r.get("survived_past_window") == "True" else 0)
            ss.append(1 if r.get("self_sustained") == "True" else 0)
        if not xs:
            continue
        order = sorted(range(len(xs)), key=lambda k: xs[k])
        xs = [xs[k] for k in order]
        surv = [surv[k] for k in order]
        ss = [ss[k] for k in order]
        c = WONG[i % len(WONG)]
        m = markers.get(cfg, "o")
        ax.plot(
            xs,
            [s + 0.02 * i for s in surv],
            color=c,
            marker=m,
            ls="-",
            label=f"{cfg}: survived window",
        )
        ax.plot(
            xs,
            [s - 0.02 * i for s in ss],
            color=c,
            marker=m,
            ls="--",
            mfc="none",
            label=f"{cfg}: self-sustained",
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"])
    ax.set_xlabel("t_window [Myr]")
    ax.set_ylabel("outcome")
    ax.set_title("H4 PdV-cap: survived-window (solid) / self-sustained (dashed) vs t_window")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "h4_summary")


def main():
    for cfg in COLLAPSE:
        fig_eb(cfg)
        fig_ratio(cfg)
    fig_summary()


if __name__ == "__main__":
    main()
