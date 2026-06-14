#!/usr/bin/env python3
"""Animated (beta,delta) root-finding GIF for the steep run: the cage vs no cage.

Three panels, frame = energy-implicit segment (i.e. increasing time):
  LEFT  : the (beta,delta) plane. The legacy clamp box is "the cage". The hybr
          root (no cage) traces out over time; when it leaves the box the cage
          pins it to the nearest edge (clip), with a connector = the clamp error.
  TOP-R : the reconstructed bubble velocity profile v vs density n at that segment
          (re-solved with get_bubbleproperties_pure; inflow = v<0).
  BOT-R : Lmech_W / Lmech_SN / Lmech_total vs t, with a marker at the current t.

Data: analysis/data/stalling_steep_1e6_alpha-2.csv (state + Lmech) + the steep
config probe_cloudPL.param (for the structure re-solve). REQUIRES the pinned
deps (numpy<2, scipy<2) + pillow:
  PYTHONPATH=<repo> /path/to/venv/bin/python scratch/phase2/make_rootmap_gif.py
Writes rootmap_cage.gif.
"""

import csv
import logging
from pathlib import Path

logging.disable(logging.CRITICAL)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from trinity._input import read_param  # noqa: E402
from trinity._input.dictionary import updateDict  # noqa: E402
from trinity.sps.update_feedback import get_current_sps_feedback  # noqa: E402
from trinity.bubble_structure.bubble_luminosity import get_bubbleproperties_pure  # noqa: E402
from trinity.cooling.non_CIE import read_cloudy as non_CIE  # noqa: E402
import trinity.main as tmain  # noqa: E402
import trinity.phase1_energy.run_energy_phase as RE  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[1] / "analysis" / "data" / "stalling_steep_1e6_alpha-2.csv"
PARAM = HERE / "probe_cloudPL.param"
BOX_B, BOX_D = (0.0, 1.0), (-1.0, 0.0)  # the "cage" (legacy clamp)
SUBSAMPLE = 1  # reconstruct every Nth segment (cost: ~2.2 s each)
NPLOT = 400  # thin the ~6e4-point profile for plotting
FPS = 10

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


def reconstruct(params, row):
    """Re-solve the structure for one segment; return thinned (n, v) + scalars."""
    t = float(row["t_now"])
    R2, v2, Eb = float(row["R2"]), float(row["v2"]), float(row["Eb"])
    beta, delta = float(row["cool_beta"]), float(row["cool_delta"])
    params["t_now"].value = t
    updateDict(params, get_current_sps_feedback(t, params))
    c, h, n = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = c
    params["cStruc_heating_nonCIE"].value = h
    params["cStruc_net_nonCIE_interpolation"].value = n
    params["current_phase"].value = "implicit"
    for k, val in (
        ("R2", R2),
        ("v2", v2),
        ("Eb", Eb),
        ("cool_alpha", t / R2 * v2),
        ("cool_beta", beta),
        ("cool_delta", delta),
    ):
        params[k].value = val
    props = get_bubbleproperties_pure(params)
    narr = np.asarray(props.bubble_n_arr, dtype=float)
    varr = np.asarray(props.bubble_v_arr, dtype=float)
    o = np.argsort(narr)
    narr, varr = narr[o], varr[o]
    s = max(1, len(narr) // NPLOT)
    return narr[::s], varr[::s], dict(t=t, beta=beta, delta=delta)


def main():
    rows = list(csv.DictReader(open(CSV)))
    t_all = np.array([float(r["t_now"]) for r in rows])
    lw = np.array([float(r["Lmech_W"]) for r in rows]) / 1e8
    lsn = np.array([float(r["Lmech_SN"]) for r in rows]) / 1e8
    lt = np.array([float(r["Lmech_total"]) for r in rows]) / 1e8
    b_all = np.array([float(r["cool_beta"]) for r in rows])
    d_all = np.array([float(r["cool_delta"]) for r in rows])

    params = init_params()
    frames = []
    sel = rows[::SUBSAMPLE]
    print(f"reconstructing {len(sel)} segments...")
    for k, row in enumerate(sel):
        n, v, meta = reconstruct(params, row)
        frames.append(dict(n=n, v=v, **meta))
        if k % 10 == 0:
            print(f"  {k}/{len(sel)}  t={meta['t']:.2f}")

    bf = np.array([f["beta"] for f in frames])
    df = np.array([f["delta"] for f in frames])
    nmin = min(f["n"].min() for f in frames)
    nmax = max(f["n"].max() for f in frames)
    vmin = min(f["v"].min() for f in frames)
    vmax = max(f["v"].max() for f in frames)

    fig, axd = plt.subplot_mosaic(
        [["A", "B"], ["A", "C"]],
        figsize=(13, 7.5),
        width_ratios=[1.5, 1.0],
        constrained_layout=True,
    )
    aA, aB, aC = axd["A"], axd["B"], axd["C"]
    tnorm = Normalize(t_all.min(), t_all.max())

    def update(i):
        f = frames[i]
        # ---- panel A: (beta, delta) plane, cage vs no cage ----
        aA.clear()
        aA.add_patch(
            Rectangle(
                (BOX_B[0], BOX_D[0]),
                BOX_B[1] - BOX_B[0],
                BOX_D[1] - BOX_D[0],
                fill=False,
                edgecolor="cyan",
                lw=2.5,
                zorder=2,
            )
        )
        aA.text(
            0.5,
            -0.5,
            "the cage\n(legacy box)",
            color="darkcyan",
            ha="center",
            va="center",
            fontsize=8,
            zorder=2,
        )
        aA.plot(b_all, d_all, color="0.85", lw=1.0, zorder=1)  # full path (context)
        aA.scatter(
            bf[: i + 1],
            df[: i + 1],
            c=[fr["t"] for fr in frames[: i + 1]],
            cmap="viridis",
            norm=tnorm,
            s=16,
            zorder=3,
        )
        bt, dt = f["beta"], f["delta"]
        bc, dc = np.clip(bt, *BOX_B), np.clip(dt, *BOX_D)  # the caged (clamped) root
        out = (bt != bc) or (dt != dc)
        if out:
            aA.plot([bc, bt], [dc, dt], color="crimson", lw=1.4, ls="--", zorder=4)
        aA.plot(
            bt,
            dt,
            marker="*",
            ms=22,
            mfc="#ffd000",
            mec="k",
            mew=1.2,
            ls="none",
            zorder=6,
            label="hybr root (no cage)",
        )
        aA.plot(
            bc,
            dc,
            marker="s",
            ms=11,
            mfc="crimson" if out else "none",
            mec="crimson",
            mew=1.6,
            ls="none",
            zorder=5,
            label="caged root (clamped)",
        )
        aA.set_xlim(-2.75, 2.1)
        aA.set_ylim(-1.3, 2.05)
        aA.set_xlabel(r"$\beta$")
        aA.set_ylabel(r"$\delta$")
        tag = "OUTSIDE the cage" if out else "inside the cage"
        aA.set_title(
            f"Root finding with vs without the cage   (t={f['t']:.2f} Myr; {tag})\n"
            f"β={bt:+.2f}, δ={dt:+.2f}, β+δ={bt+dt:+.2f}",
            fontsize=10,
        )
        aA.text(
            0.02,
            0.98,
            "path colour = time →",
            transform=aA.transAxes,
            fontsize=8,
            va="top",
            color="0.45",
        )
        aA.legend(loc="lower right", fontsize=8)
        aA.grid(alpha=0.3)

        # ---- panel B: bubble velocity vs density ----
        aB.clear()
        aB.axhline(0.0, color="k", lw=0.8)
        aB.axhspan(min(vmin, -0.05) * 1.1, 0.0, color="r", alpha=0.06)
        aB.plot(f["n"], f["v"], color="#0072B2", lw=1.8)
        aB.set_xscale("log")
        aB.set_xlim(nmin, nmax)
        aB.set_ylim(vmin * 1.1 - 0.05, vmax * 1.05)
        aB.set_xlabel("bubble density  n  [code units, log]")
        aB.set_ylabel("v(r)  [pc/Myr]")
        inflow = float(np.nanmin(f["v"])) < -0.01
        aB.set_title(
            "bubble velocity vs n" + ("  — INFLOW (v<0)" if inflow else ""),
            fontsize=10,
            color="#b30000" if inflow else "k",
        )
        aB.grid(alpha=0.3)

        # ---- panel C: Lmech vs t ----
        aC.clear()
        aC.plot(t_all, lt, color="k", lw=1.6, ls="--", label=r"$L_{\rm tot}$")
        aC.plot(t_all, lw, color="#0072B2", lw=1.4, label=r"$L_{\rm W}$ (wind)")
        aC.plot(t_all, lsn, color="#9467bd", lw=1.4, label=r"$L_{\rm SN}$")
        aC.axvline(f["t"], color="crimson", lw=1.4)
        aC.set_xlabel("t  [Myr]")
        aC.set_ylabel(r"$L_{\rm mech}$ [$10^8$]")
        aC.set_title("feedback power", fontsize=10)
        aC.legend(fontsize=7, loc="upper left", ncol=3)
        aC.grid(alpha=0.3)
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False)
    out = HERE / "rootmap_cage.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=85)
    plt.close(fig)
    print(f"wrote {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
