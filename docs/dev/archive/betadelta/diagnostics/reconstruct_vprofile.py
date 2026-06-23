#!/usr/bin/env python3
"""Reconstruct the bubble interior velocity profile v(r) for the steep run's
inflow segments (WARPFIELD "Problem 2").

The real bubble_v_arr was ephemeral and committed nowhere -- only the scalar
v_struct_min / v_struct_nneg survive in stalling_steep_1e6_alpha-2.csv. Here we
re-solve trinity's actual bubble structure for each segment: initialise params
via the real run (start_expansion, short-circuited at phase 1a so SPS + cooling
setup runs), set the SB99 feedback at the segment age, load the age's non-CIE
cooling, override the recorded state (t, R2, v2, Eb, beta, delta, alpha=t*v2/R2),
and call get_bubbleproperties_pure. Validation against the CSV is exact
(Pb, dMdt, v_min all match to the digit), so the profile is faithful.

REQUIRES the pinned deps (numpy<2, scipy<2): run with a venv built from
requirements.txt, e.g.  PYTHONPATH=<repo> /path/to/venv/bin/python this.py
(numpy 2.4 in the base env trips the integrator's monotonic guard.)

Produces negvel_profile.png: v vs radial fraction for the 4 inflow segments
(coloured by beta+delta) plus a pre-surge normal segment -- showing the inner
inflow band that the monotone Weaver outflow ansatz cannot represent.
"""

import csv
import logging
from pathlib import Path

logging.disable(logging.CRITICAL)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from trinity._input import read_param  # noqa: E402
from trinity._input.dictionary import updateDict  # noqa: E402
from trinity.sps.update_feedback import get_current_sps_feedback  # noqa: E402
from trinity.bubble_structure.bubble_luminosity import get_bubbleproperties_pure  # noqa: E402
from trinity.cooling.non_CIE import read_cloudy as non_CIE  # noqa: E402
import trinity.main as tmain  # noqa: E402
import trinity.phase1_energy.run_energy_phase as RE  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE.parents[2] / "data" / "stalling_steep_1e6_alpha-2.csv"  # canonical
PARAM = HERE / "probe_cloudPL.param"  # the steep 1e6, alpha=-2 config
NNEG_REAL = 10

_STYLE = HERE.parents[4] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams.update(
    {
        "text.usetex": False,
        "axes.labelsize": 11.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.5,
    }
)


class _InitDone(Exception):
    pass


def init_params():
    """Run the real init (cloud/SPS/CIE) but stop before phase 1a."""
    RE.run_energy = lambda params: (_ for _ in ()).throw(_InitDone())
    params = read_param.read_param(str(PARAM))
    try:
        tmain.start_expansion(params)
    except _InitDone:
        pass
    return params


def _load_noncie(params):
    c, h, n = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = c
    params["cStruc_heating_nonCIE"].value = h
    params["cStruc_net_nonCIE_interpolation"].value = n


def reconstruct(params, row):
    """Re-solve the bubble structure for one CSV segment; return sorted (f, v)."""
    t = float(row["t_now"])
    R2, v2, Eb = float(row["R2"]), float(row["v2"]), float(row["Eb"])
    beta, delta = float(row["cool_beta"]), float(row["cool_delta"])
    params["t_now"].value = t
    updateDict(params, get_current_sps_feedback(t, params))
    _load_noncie(params)
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
    v = np.asarray(props.bubble_v_arr, dtype=float)
    r = np.asarray(props.bubble_r_arr, dtype=float)
    # radial fraction: 0 = R1 (inner BC), 1 = R2 (outer). bubble_r_arr is
    # integrated from r2_prime (~R2) inward toward R1, so f can stop a hair short
    # of 0/1 -- cosmetic only; the v values themselves are exact (validated).
    f = (r - props.R1) / (R2 - props.R1)
    o = np.argsort(f)
    f, v = f[o], v[o]
    s = max(1, len(f) // 1500)  # thin the very fine grid for plotting
    meta = dict(
        t=t,
        bpd=beta + delta,
        beta=beta,
        vmin=float(np.nanmin(v)),
        csv_vmin=float(row["v_struct_min"]),
        pb=props.Pb,
        csv_pb=float(row["Pb"]),
        dmdt=props.bubble_dMdt,
        csv_dmdt=float(row["bubble_dMdt"]),
    )
    return f[::s], v[::s], meta


def main():
    rows = list(csv.DictReader(open(CSV)))
    inflow = sorted(
        (r for r in rows if float(r["v_struct_nneg"]) >= NNEG_REAL),
        key=lambda r: float(r["beta_plus_delta"]),
    )
    normal = min(rows, key=lambda r: abs(float(r["t_now"]) - 2.85))  # pre-surge

    params = init_params()
    norm = Normalize(vmin=-1.2, vmax=-0.4)
    cmap = plt.cm.autumn  # -1.1 -> red ... -0.5 -> yellow
    fig, ax = plt.subplots(figsize=(9, 6))

    f, v, m = reconstruct(params, normal)
    ax.plot(
        f, v, color="0.45", lw=1.8, ls="--", label=f"normal  t={m['t']:.2f}, β+δ={m['bpd']:+.2f}"
    )
    print(f"normal  t={m['t']:.3f} vmin recon={m['vmin']:+.3f} csv={m['csv_vmin']:+.3f}")

    for row in inflow:
        f, v, m = reconstruct(params, row)
        ax.plot(
            f,
            v,
            color=cmap(norm(m["bpd"])),
            lw=1.8,
            label=f"t={m['t']:.2f}, β+δ={m['bpd']:+.2f}, v_min={m['vmin']:+.2f}",
        )
        print(
            f"inflow  t={m['t']:.3f} b+d={m['bpd']:+.2f} | vmin {m['vmin']:+.3f}/{m['csv_vmin']:+.3f}"
            f" Pb {m['pb']:.2e}/{m['csv_pb']:.2e} dMdt {m['dmdt']:.0f}/{m['csv_dmdt']:.0f}"
        )

    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhspan(ax.get_ylim()[0], 0.0, color="r", alpha=0.05)
    ax.text(0.02, -0.4, "inflow (v<0)", color="#b30000", fontsize=9, va="top")

    # how significant is the inflow? (numbers per docs/dev/archive/betadelta/stalling-energy-phase.md)
    deep = min(inflow, key=lambda r: float(r["v_struct_min"]))
    mach = abs(float(deep["v_struct_min"])) / float(deep["c_sound"])
    ax.text(
        0.98,
        0.02,
        f"deepest band: Mach ≈ {mach:.3f} (subsonic), KE/thermal ~ {mach**2:.0e}\n"
        "→ energetically negligible. The profile is a quasi-steady-ansatz output;\n"
        "likely an artefact — real-vs-artefact is OPEN (stalling-energy-phase.md).",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.25",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92),
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("radial fraction across bubble   (0 = R1 inner, 1 = R2 outer shell)")
    ax.set_ylabel("bubble interior velocity  v(r)  [pc/Myr]")
    ax.set_title(
        "Reconstructed bubble velocity profile — inner inflow band at the β+δ dive\n"
        "(steep 1e6, α=−2; validated against CSV Pb/dMdt/v_min)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=r"$\beta+\delta$ (inflow segments)")
    fig.tight_layout()
    fig.savefig(HERE / "negvel_profile.png", dpi=130)
    plt.close(fig)
    print("wrote negvel_profile.png")


if __name__ == "__main__":
    main()
