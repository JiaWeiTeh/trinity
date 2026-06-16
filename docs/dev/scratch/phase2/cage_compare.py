#!/usr/bin/env python3
"""What a caged (legacy/clamped) solve actually predicts vs hybr — static figure.

The rootmap GIF clamps the hybr root to the box as a *geometric* proxy. This
runs the REAL legacy bounded solver (betadelta_solver='legacy') at the same
segment state, so the "caged" root is the actual in-box optimum it finds (which
rides a contaminated residual, not a simple projection) -- and reconstructs its
velocity profile. Per-segment legacy solve is ~60 s (it grids ~25 structure
solves through the f-pole), so this is a few key segments, not an animation.

Two segments (a pre-surge normal + the deepest WR-surge dip). For each:
  top  : the (beta,delta) roots -- hybr (no cage) vs legacy (caged) + the box.
  right: v vs radial fraction -- hybr (with inflow) vs caged (monotone, none).
The point: at the surge the cage is forced to a different in-box root that
predicts NO inflow -- it HIDES the Problem-2 excursion.

REQUIRES the venv (numpy<2, scipy<2):
  PYTHONPATH=<repo> /path/to/venv/bin/python docs/dev/scratch/phase2/cage_compare.py
"""

import csv
import logging
from pathlib import Path

logging.disable(logging.CRITICAL)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from trinity._input import read_param  # noqa: E402
from trinity._input.dictionary import updateDict  # noqa: E402
from trinity.sps.update_feedback import get_current_sps_feedback  # noqa: E402
from trinity.bubble_structure.bubble_luminosity import get_bubbleproperties_pure  # noqa: E402
import trinity.phase1b_energy_implicit.get_betadelta as GB  # noqa: E402
from trinity.cooling.non_CIE import read_cloudy as non_CIE  # noqa: E402
import trinity.main as tmain  # noqa: E402
import trinity.phase1_energy.run_energy_phase as RE  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[1] / "analysis" / "data" / "stalling_steep_1e6_alpha-2.csv"
PARAM = HERE / "probe_cloudPL.param"
BOX_B, BOX_D = (0.0, 1.0), (-1.0, 0.0)

_STYLE = HERE.parents[1] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams["text.usetex"] = False


class _InitDone(Exception):
    pass


def init_params():
    RE.run_energy = lambda params: (_ for _ in ()).throw(_InitDone())
    params = read_param.read_param(str(PARAM))
    try:
        tmain.start_expansion(params)
    except _InitDone:
        pass
    return params


def _set_state(params, row):
    t = float(row["t_now"])
    params["t_now"].value = t
    updateDict(params, get_current_sps_feedback(t, params))
    c, h, n = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = c
    params["cStruc_heating_nonCIE"].value = h
    params["cStruc_net_nonCIE_interpolation"].value = n
    params["current_phase"].value = "implicit"
    R2, v2, Eb = float(row["R2"]), float(row["v2"]), float(row["Eb"])
    for k, val in (("R2", R2), ("v2", v2), ("Eb", Eb), ("cool_alpha", t / R2 * v2)):
        params[k].value = val
    return R2


def _profile(props, R2):
    r = np.asarray(props.bubble_r_arr, dtype=float)
    v = np.asarray(props.bubble_v_arr, dtype=float)
    f = (r - props.R1) / (R2 - props.R1)
    o = np.argsort(f)
    s = max(1, len(f) // 800)
    return f[o][::s], v[o][::s]


def analyse(params, row):
    R2 = _set_state(params, row)
    hb, hd = float(row["cool_beta"]), float(row["cool_delta"])
    # hybr (no cage): the recorded root
    params["betadelta_solver"].value = "hybr"
    for k, val in (("cool_beta", hb), ("cool_delta", hd)):
        params[k].value = val
    fh, vh = _profile(get_bubbleproperties_pure(params), R2)
    # caged (legacy): re-solve, clamped to the box
    params["betadelta_solver"].value = "legacy"
    res = GB.solve_betadelta_pure(0.5, -0.1, params, "grid")
    fc, vc = (
        _profile(res.bubble_properties, R2) if res.bubble_properties is not None else (None, None)
    )
    return dict(
        t=float(row["t_now"]), hb=hb, hd=hd, cb=res.beta, cd=res.delta, fh=fh, vh=vh, fc=fc, vc=vc
    )


def main():
    rows = list(csv.DictReader(open(CSV)))
    normal = min(rows, key=lambda r: abs(float(r["t_now"]) - 2.83))
    deepest = min(rows, key=lambda r: float(r["v_struct_min"]))
    params = init_params()
    segs = [
        ("normal (pre-surge)", analyse(params, normal)),
        ("WR-surge dip (deepest)", analyse(params, deepest)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for r, (title, s) in enumerate(segs):
        aL, aR = axes[r]
        # (beta, delta) roots
        aL.add_patch(Rectangle((BOX_B[0], BOX_D[0]), 1, 1, fill=False, edgecolor="cyan", lw=2.5))
        aL.text(0.5, -0.5, "the cage\n(legacy box)", color="darkcyan", ha="center", fontsize=8)
        out = not (BOX_B[0] <= s["hb"] <= BOX_B[1] and BOX_D[0] <= s["hd"] <= BOX_D[1])
        if out:
            aL.plot([s["cb"], s["hb"]], [s["cd"], s["hd"]], color="crimson", ls="--", lw=1.2)
        aL.plot(
            s["hb"],
            s["hd"],
            "*",
            ms=22,
            mfc="#ffd000",
            mec="k",
            mew=1.2,
            label=f"hybr (no cage)  β+δ={s['hb']+s['hd']:+.2f}",
        )
        aL.plot(
            s["cb"],
            s["cd"],
            "s",
            ms=12,
            mfc="crimson",
            mec="k",
            label=f"legacy (caged)  β+δ={s['cb']+s['cd']:+.2f}",
        )
        aL.set_xlim(-2.75, 2.1)
        aL.set_ylim(-1.3, 2.05)
        aL.set_xlabel(r"$\beta$")
        aL.set_ylabel(r"$\delta$")
        aL.set_title(f"{title}  (t={s['t']:.2f} Myr)", fontsize=10)
        aL.legend(fontsize=8, loc="upper right")
        aL.grid(alpha=0.3)
        # v vs radial fraction
        aR.axhline(0.0, color="k", lw=0.8)
        aR.axhspan(-1, 0, color="r", alpha=0.05)
        aR.plot(
            s["fh"],
            s["vh"],
            color="#0072B2",
            lw=2,
            label=f"hybr (no cage), v_min={np.nanmin(s['vh']):+.2f}",
        )
        if s["fc"] is not None:
            aR.plot(
                s["fc"],
                s["vc"],
                color="0.45",
                lw=2,
                ls="--",
                label=f"legacy (caged), v_min={np.nanmin(s['vc']):+.2f}",
            )
        aR.set_xlim(0, 1)
        aR.set_xlabel("radial fraction  (0 = R1 inner, 1 = R2 outer)")
        aR.set_ylabel("v(r)  [pc/Myr]")
        inflow = np.nanmin(s["vh"]) < -0.01
        aR.set_title(
            "velocity profile — cage " + ("HIDES the inflow" if inflow else "≈ hybr"),
            fontsize=10,
            color="#b30000" if inflow else "k",
        )
        aR.legend(fontsize=8, loc="upper left")
        aR.grid(alpha=0.3)

    fig.suptitle(
        "What the cage actually predicts (real legacy solve, not a clip): at the WR surge it is\n"
        "forced to a different in-box root with NO interior inflow — it hides Problem 2 (steep 1e6, α=−2)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "cage_compare.png", dpi=130)
    plt.close(fig)
    print("wrote cage_compare.png")
    for title, s in segs:
        print(
            f"  {title}: hybr=({s['hb']:+.2f},{s['hd']:+.2f}) caged=({s['cb']:+.2f},{s['cd']:+.2f})"
        )


if __name__ == "__main__":
    main()
