#!/usr/bin/env python3
"""The dMdt dip — trace data + the two storyline figures (KAPPA_FREEZE_MECHANISM §5, FINDINGS §12.7).

Data: the controlled dense pair (mCloud 1e4, nCore 1e6, rCore 0.1, sfe 0.5, PL0; params
committed in runs/params/dmdt_trace/) run locally at cooling_boost_kappa 6 (the theta5k
CONDENSE arm) and 8 (the FIRE arm) with log_level DEBUG. The per-segment beta-delta eigenvalue
— accepted dMdt from `freeze-watch` lines, rejected negative roots from the `no physical`
warnings — was parsed from those runs' trinity.log into data/dmdt_trace_dense.csv (committed;
the raw logs live only in the ephemeral session scratchpad). θ from these runs is mechanism
diagnosis ONLY, never calibration (CONTAMINATION register).

Figures (written to the workstream root):
  dmdt_dip_traces.png — THE PROBLEM: the eigenvalue dips below zero (condensation), the
      dMdt>0 gate refuses it; k8 recovers through zero and fires, k6 second-dives and is
      handed off. Smooth arcs = solver exonerated.
  dmdt_tackle_flow.png — THE RESOLUTION: symptom → diagnosis → physics identity → the three
      literature treatments → what TRINITY adopted (no-root streak ⇒ momentum handoff) →
      validation (theta5k) and the production conclusion.

REPRODUCE:
    python docs/dev/transition/pdv-trigger/data/make_dmdt_dip_figures.py
    (re-parse from scratch: run the two params in runs/params/dmdt_trace/ with a local
     python run.py, then point PARSE_LOGS at the two trinity.log files)
"""

import csv
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PDV)
from _stamp import stamp  # noqa: E402
from _trinity_style import use_trinity_style  # noqa: E402

use_trinity_style()
import matplotlib.pyplot as plt  # noqa: E402

CSV = os.path.join(_HERE, "dmdt_trace_dense.csv")
# set to {"k6": "<path>/trinity.log", "k8": "<path>/trinity.log"} to re-parse from raw logs
PARSE_LOGS = {}

_REJECT = re.compile(
    r"no physical \(dMdt>0\) root at segment (\d+) \(t=([0-9.e+-]+) Myr, streak (\d+)\): "
    r"non-physical dMdt=(-?[0-9.e+-]+)"
)
_ACCEPT = re.compile(r"freeze-watch: segment=(\d+) t=([0-9.e+-]+) dMdt=([0-9.e+-]+)")


def parse(arm, path):
    rows = []
    for line in open(path, errors="replace"):
        m = _ACCEPT.search(line)
        if m:
            rows.append(
                {"arm": arm, "segment": int(m[1]), "t_Myr": float(m[2]),
                 "dMdt": float(m[3]), "kind": "accepted", "streak": 0}
            )
            continue
        m = _REJECT.search(line)
        if m:
            rows.append(
                {"arm": arm, "segment": int(m[1]), "t_Myr": float(m[2]),
                 "dMdt": float(m[4]), "kind": "rejected", "streak": int(m[3])}
            )
    return rows


def load():
    if PARSE_LOGS:
        rows = []
        for arm, path in sorted(PARSE_LOGS.items()):
            rows += parse(arm, path)
        stamp_line = stamp(__file__)
        with open(CSV, "w", newline="") as fh:
            fh.write(stamp_line + "\n")
            fh.write(
                "# per-segment beta-delta dMdt eigenvalue, dense controlled pair (params in "
                "runs/params/dmdt_trace/): accepted = freeze-watch (structure accepted), "
                "rejected = the NEGATIVE root the solver converged to and the dMdt>0 gate "
                "refused (condensation branch). MECHANISM DIAGNOSIS ONLY - no theta quotes.\n"
            )
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {CSV} ({len(rows)} rows)")
    with open(CSV) as fh:
        return [
            {**r, "segment": int(r["segment"]), "t_Myr": float(r["t_Myr"]),
             "dMdt": float(r["dMdt"]), "streak": int(r["streak"])}
            for r in csv.DictReader(l for l in fh if not l.lstrip().startswith("#"))
        ]


ARMCOLOR = {"k6": "#CC79A7", "k8": "#0072B2"}
ARMLABEL = {
    "k6": "$f_\\kappa=6$ — never recovers $\\Rightarrow$ condensation handoff (theta5k: CONDENSE)",
    "k8": "$f_\\kappa=8$ — recovers through zero $\\Rightarrow$ fires (theta5k: FIRED, $n_{\\rm impl}$=28)",
}


def fig_traces(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhspan(-48, 0, color="#CC79A7", alpha=0.08, zorder=0)
    ax.axhline(0, color="0.25", lw=1.0)
    ax.text(
        0.8, -45.5, "shaded: forbidden by the dMdt$\\,>\\,$0 gate (condensation branch: McKee & Cowie 1977)",
        fontsize=7.5, color="#9c5a80", ha="left", va="bottom",
    )
    for arm in ("k6", "k8"):
        pts = sorted((r["segment"], r["dMdt"], r["kind"]) for r in rows if r["arm"] == arm)
        # clip the +320 birth point: draw the curve only from segment 2 on
        xs = [p[0] for p in pts if p[1] < 85]
        ys = [p[1] for p in pts if p[1] < 85]
        ax.plot(xs, ys, "-", color=ARMCOLOR[arm], lw=1.2, alpha=0.7, zorder=3)
        for s, d, kind in pts:
            if d >= 85:
                continue
            ax.plot(
                s, d, marker="o" if kind == "accepted" else "x",
                color=ARMCOLOR[arm], ms=8 if kind == "accepted" else 5,
                mew=1.6, zorder=4,
                mfc=ARMCOLOR[arm] if kind == "accepted" else "none",
            )
    # birth point (both arms ~+320, off scale): clip marker at top edge
    ax.annotate(
        "segment 1: both arms accept an\nevaporative solve at $+3.2\\times10^2$ (off scale)",
        (1, 80), textcoords="offset points", xytext=(10, -14), fontsize=7.5, color="0.25",
        va="top",
        arrowprops=dict(arrowstyle="->", color="0.55", lw=0.8),
    )
    ax.plot(1, 80, marker="^", color="0.4", ms=7, zorder=4)
    ax.annotate(
        "k8 recovers through zero ($+65$)\n$\\Rightarrow$ evaporation resumes $\\Rightarrow$ FIRES",
        (28, 65.3), textcoords="offset points", xytext=(-150, -4), fontsize=7.5,
        color="#0072B2",
        arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.9),
    )
    ax.annotate(
        "k6 nearly recovers ($-4.0$)…",
        (8, -3.986), textcoords="offset points", xytext=(-30, 22), fontsize=7.5, color="#9c5a80",
        arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=0.9),
    )
    ax.annotate(
        "…then second-dives ($-38$) and never recovers\n$\\Rightarrow$ condensation handoff at streak 50",
        (9.4, -39), textcoords="offset points", xytext=(28, 118), fontsize=7.5,
        color="#9c5a80", ha="left",
        arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=0.9,
                        connectionstyle="arc3,rad=-0.25"),
    )
    ax.set_xlim(0, 31)
    ax.set_ylim(-48, 85)
    ax.set_xlabel("implicit-phase segment")
    ax.set_ylabel("$\\dot M_{\\rm b}$ eigenvalue of the $\\beta$–$\\delta$ solve  [$M_\\odot$/Myr]")
    handles = [
        plt.Line2D([], [], color=ARMCOLOR[a], marker="s", lw=1.4, ms=0, label=ARMLABEL[a])
        for a in ("k6", "k8")
    ] + [
        plt.Line2D([], [], color="0.3", marker="o", lw=0, ms=7, mfc="0.3", label="accepted (evaporating, $\\dot M>0$)"),
        plt.Line2D([], [], color="0.3", marker="x", lw=0, ms=5, mew=1.6, label="rejected root (condensing, $\\dot M<0$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title(
        "The dMdt dip: the conduction front's mass-flux eigenvalue crosses into condensation\n"
        "smooth arcs, controlled pair (identical early dt) — the solver is fine; the MODEL has no $\\dot M<0$ branch"
    )
    fig.savefig(os.path.join(_PDV, "dmdt_dip_traces.png"), bbox_inches="tight")
    plt.close(fig)


def fig_flow():
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, title_c="0.1", fs=7.3):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="top",
                fontsize=fs + 1.2, fontweight="bold", color=title_c, zorder=3)
        ax.text(x + w / 2, y + h - 0.95, body, ha="center", va="top", fontsize=fs,
                color="0.15", zorder=3)

    def arrow(x0, y0, x1, y1, c="0.35", lw=1.4):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=lw), zorder=1)

    # row 1: symptom -> diagnosis -> physics
    box(0.1, 6.6, 2.9, 3.3, "SYMPTOM",
        "runs 'freeze' mid-energy-phase\n$\\theta$ stuck (e.g. 0.53), exit 0\n"
        "$\\rightarrow$ read as non-monotonic\n\"dead windows\" in $f_\\kappa$ (§9a)",
        "#fdf3f7", "#CC79A7")
    box(3.55, 6.6, 2.9, 3.3, "DIAGNOSIS",
        "34/38 freezes die at $\\theta\\geq0.8$;\nsolver converges to $\\dot M_b<0$\n"
        "(−85 $M_\\odot$/Myr); the dMdt>0 gate\nrefuses it; runner grinds, frozen",
        "#fff8ec", "#E69F00")
    box(7.0, 6.6, 2.9, 3.3, "PHYSICS IDENTITY",
        "evaporation$\\,\\to\\,$condensation reversal\n(McKee & Cowie 77; Weaver's own\n"
        "front is already 60/40); closure\n$T\\propto\\dot M^{2/5}$ has NO $\\dot M<0$ branch",
        "#eef5fb", "#0072B2")
    arrow(3.0, 8.25, 3.55, 8.25)
    arrow(6.45, 8.25, 7.0, 8.25)

    # row 2: the three literature treatments
    box(0.1, 3.1, 2.9, 2.6, "follow the sign",
        "Lagrangian hydro: conduction as\ndiffusion, flux free to reverse\n"
        "(El-Badry+19, Vieser & Hensler 07)\n= NEW PROFILE FAMILY, research-grade",
        "#f7f7f7", "0.6")
    box(3.55, 3.1, 2.9, 2.6, "cap the flux",
        "saturated conduction\n(Cowie & McKee 77)\ntames high-$f_\\kappa$ onset;\n"
        "does NOT remove the reversal",
        "#f7f7f7", "0.6")
    box(7.0, 3.1, 2.9, 2.6, "switch regimes — ADOPTED",
        "energy$\\,\\to\\,$momentum handoff at\ncatastrophic-cooling onset\n"
        "(Weaver/Mac Low/Silich lineage)\n= TRINITY's own model class",
        "#eaf4ee", "#009E73", title_c="#00583d")
    arrow(8.45, 6.6, 8.45, 5.7, c="#009E73", lw=2.0)
    arrow(4.0, 6.6, 1.8, 5.7)
    arrow(5.0, 6.6, 5.0, 5.7)

    # row 3: what shipped + outcome
    box(2.2, 0.1, 5.6, 2.5, "WHAT SHIPPED (fix #1) — AND THE VERDICT",
        "50-segment no-root streak $\\Rightarrow$ 'no_physical_root_handoff' (a fate, not a trigger)\n"
        "theta5k, 56 rule-compliant runs: ZERO freezes; old dead windows = CONDENSE fates\n"
        "still no whole-band $f_\\kappa$ (best 5/6) $\\Rightarrow$ $f_{\\rm mix}$ multiplier stays the production knob\n"
        "(window [4, 4.5] fires 6/6; $\\theta_1$-collapse law, 0.064 dex out-of-sample)",
        "#eaf4ee", "#009E73", title_c="#00583d")
    arrow(8.45, 3.1, 7.0, 2.3, c="#009E73", lw=2.0)

    ax.set_title(
        "How the dMdt dip was tackled: from silent freeze to a measured, physical phase boundary",
        fontsize=10.5,
    )
    fig.savefig(os.path.join(_PDV, "dmdt_tackle_flow.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load()
    fig_traces(rows)
    fig_flow()
    print("wrote dmdt_dip_traces.png, dmdt_tackle_flow.png ->", _PDV)


if __name__ == "__main__":
    main()
