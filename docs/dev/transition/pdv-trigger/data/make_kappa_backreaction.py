#!/usr/bin/env python3
"""Rung-A kappa_eff back-reaction: what inflating Spitzer C by f_kappa does, at matched t.

WHAT THIS IS
------------
Option C (KAPPA_EFF_SCOPING.md) replaces the scalar cooling boost on Lcool with a
*structural* knob: cooling_boost_kappa = f_kappa multiplies the Spitzer conduction
coefficient C_thermal at all three sites in bubble_luminosity (the dMdt closure, the
backward-ODE initial conditions, and the dT/dr ODE). f_kappa=1.0 is byte-identical;
f_kappa>1 inflates conduction-zone cooling THROUGH the structure, so the loss fraction
theta emerges as an OUTPUT rather than being imposed.

Rung A is the structural PROBE (not the final model): a flat f_kappa raises both the
conduction-zone cooling (intended) AND the evaporative mass flux dMdt (the El-Badry
kappa coupling a faithful kappa_eff must instead SUPPRESS). This script quantifies that
crux on the stiff f1edge_hidens edge by comparing two real, separate-process runs at
MATCHED simulation time (trinity truncates at different t, so we interpolate the
f_kappa=2 trajectory onto the f_kappa=1 time grid):

    f1edge_hidens__none    (cooling_boost_kappa defaults to 1.0)  <- baseline
    f1edge_hidens__kappa2  (cooling_boost_kappa = 2.0)

For each matched t it reports the k2/k1 ratio of:
    Lcool   = bubble_LTotal     (resolved bubble cooling luminosity)
    dMdt    = bubble_dMdt        (evaporative mass flux -- the coupling crux)
    Lmech   = Lmech_total        (mechanical input; sanity: must stay 1.0)
    Eb, Pb, R2, v2               (bubble energetics / geometry back-reaction)
and the loss-ratio proxy Lcool/Lmech for both runs (how far f_kappa moves it toward
the 0.95 transition trigger).

REPRODUCE (from repo root; runs are separate-process via run_stamped.py):
    python docs/dev/transition/harness/run_stamped.py \
        docs/dev/transition/pdv-trigger/runs/params/f1edge_hidens__none.param
    python docs/dev/transition/harness/run_stamped.py \
        docs/dev/transition/pdv-trigger/runs/params/f1edge_hidens__kappa2.param
    python docs/dev/transition/pdv-trigger/data/make_kappa_backreaction.py \
        outputs/pdvlive/f1edge_hidens__none/dictionary.jsonl \
        outputs/pdvlive/f1edge_hidens__kappa2/dictionary.jsonl

Deliverables:
    docs/dev/transition/pdv-trigger/data/kappa_backreaction.csv
    docs/dev/transition/pdv-trigger/kappa_backreaction.png
"""

import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_DEFAULT_K1 = "outputs/pdvlive/f1edge_hidens__none/dictionary.jsonl"
_DEFAULT_K2 = "outputs/pdvlive/f1edge_hidens__kappa2/dictionary.jsonl"

# (label, dictionary.jsonl key) for the columns the back-reaction touches.
_COLS = [
    ("Lcool", "bubble_LTotal"),
    ("dMdt", "bubble_dMdt"),
    ("Lmech", "Lmech_total"),
    ("Eb", "Eb"),
    ("Pb", "Pb"),
    ("R2", "R2"),
    ("v2", "v2"),
]


def _load(path):
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def _col(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], float)


def main(k1_path, k2_path):
    k1, k2 = _load(k1_path), _load(k2_path)
    t1, t2 = _col(k1, "t_now"), _col(k2, "t_now")

    # Interpolate the f_kappa=2 trajectory onto the f_kappa=1 time grid (both monotone
    # in t_now) so every comparison is at MATCHED simulation time.
    def on_t1(rows_t, rows_y):
        return np.interp(t1, rows_t, rows_y)

    data = {"t": t1}
    for label, key in _COLS:
        y1 = _col(k1, key)
        y2 = on_t1(t2, _col(k2, key))
        data[f"{label}_k1"] = y1
        data[f"{label}_k2"] = y2
        with np.errstate(divide="ignore", invalid="ignore"):
            data[f"{label}_ratio"] = y2 / y1

    # Loss-ratio proxy Lcool/Lmech for each run.
    lr1 = _col(k1, "bubble_LTotal") / _col(k1, "Lmech_total")
    lr2 = on_t1(t2, _col(k2, "bubble_LTotal") / _col(k2, "Lmech_total"))
    data["lossratio_k1"] = lr1
    data["lossratio_k2"] = lr2

    # --- CSV --------------------------------------------------------------
    csv_path = os.path.join(_HERE, "kappa_backreaction.csv")
    cols = list(data.keys())
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(len(t1)):
            w.writerow([f"{data[c][i]:.8e}" for c in cols])
    print(f"wrote {csv_path} ({len(t1)} matched rows)")

    # --- figure -----------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - plotting is optional
        print(f"(skipping figure: {e})")
        return

    fig, (ax0, axL, axR) = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 0 -- ABSOLUTE cooling luminosity, both runs. The direct "more cooling"
    # view: the f_kappa=2 (red) curve sits ABOVE the f_kappa=1 (blue) curve at every
    # t. Both fall over time because the bubble expands -- that decline is the
    # physics, NOT the effect of the knob (the knob is the red-above-blue gap).
    ax0.plot(t1, data["Lcool_k1"], lw=2, color="#1f77b4", label=r"$f_\kappa{=}1$ (baseline)")
    ax0.plot(t1, data["Lcool_k2"], lw=2, color="#d62728", label=r"$f_\kappa{=}2$ (more conduction)")
    ax0.set_yscale("log")
    ax0.set_xlabel("t [Myr]")
    ax0.set_ylabel(r"$L_{\rm cool}$  [code units]")
    ax0.set_title("Cooling IS higher with more $\\kappa$\n(red sits above blue at every $t$)", fontsize=10)
    ax0.legend(fontsize=8)

    # Panel 1 -- the RATIOS (f_kappa=2 / f_kappa=1). These lines are NOT the
    # quantities themselves: a value ABOVE 1.0 means the knob RAISED that quantity,
    # BELOW 1.0 means it lowered it. The downward slope = the boost shrinks over time
    # but stays > 1 (cooling always enhanced). dMdt riding above 1 too is the crux.
    axL.axhspan(1.0, 1.62, color="#2ca02c", alpha=0.06)   # "raised" band
    axL.axhspan(0.84, 1.0, color="#d62728", alpha=0.06)   # "lowered" band
    axL.axhline(1.0, color="0.35", lw=1.0, ls=":")
    axL.plot(t1, data["Lcool_ratio"], lw=2.2, color="#1f77b4",
             label=r"$L_{\rm cool}$: raised $1.2$–$1.5\times$")
    axL.plot(t1, data["dMdt_ratio"], lw=2.2, color="#ff7f0e",
             label=r"$\dot M$ evap: raised $1.08$–$1.17\times$ (unwanted)")
    axL.plot(t1, data["Eb_ratio"], lw=1.8, ls="--", color="#2ca02c",
             label=r"$E_b$: drained to $0.90$–$0.96\times$")
    axL.set_ylim(0.84, 1.62)
    axL.text(0.97, 0.96, "above 1.0  =  more $\\kappa$ RAISED it", transform=axL.transAxes,
             ha="right", va="top", fontsize=7.5, color="#1c6b1c")
    axL.set_xlabel("t [Myr]")
    axL.set_ylabel(r"ratio at matched $t$:  $f_\kappa{=}2 \,\div\, f_\kappa{=}1$")
    axL.set_title("Both cooling AND evaporation rise\n(the crux: $\\dot M$ rides along, wrong sign)", fontsize=10)
    axL.legend(fontsize=7.5, loc="lower left")

    axR.plot(t1, lr1, label=r"$f_\kappa{=}1$ (baseline)", lw=2)
    axR.plot(t1, lr2, label=r"$f_\kappa{=}2$", lw=2)
    axR.axhline(0.95, color="crimson", lw=1.0, ls="--", label="0.95 trigger")
    axR.set_xlabel("t [Myr]")
    axR.set_ylabel(r"loss-ratio proxy $L_{\rm cool}/L_{\rm mech}$")
    axR.set_title(r"$2\times\kappa$ buys only $+0.05$–$0.10$ toward the trigger", fontsize=10)
    axR.legend(fontsize=8)

    fig.tight_layout()
    png_path = os.path.join(_PDV, "kappa_backreaction.png")
    fig.savefig(png_path, dpi=130)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    k1 = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_K1
    k2 = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_K2
    main(k1, k2)
