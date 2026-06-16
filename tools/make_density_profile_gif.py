#!/usr/bin/env python3
"""Density-profile GIF from a TRINITY run's ``dictionary.jsonl``.

Each frame is one timestep. The x-axis is radius (pc), spanning the bubble
interior (R1 -> R2) and the swept-up shell (R2 -> rShell); the y-axis is number
density n (cm^-3). Vertical lines mark the region boundaries R1, R2 and rShell,
and the title shows ``t_now`` in Myr.

Where the per-timestep data lives (each line of dictionary.jsonl is one step):
  - bubble interior : radius = ``bubble_n_arr_r_arr`` (pc, stored descending),
                      density = ``10**log_bubble_n_arr``
  - shell           : radius = ``shell_r_arr`` (pc, ascending),
                      density = ``10**log_shell_n_arr``
Densities are in internal units (pc^-3); divide by ``ndens_cgs2au`` for cm^-3.

In the momentum phase the bubble has collapsed (R1 == R2) and its arrays are
stale leftovers from the last energy-phase solve, so the bubble segment is
clipped to [R1, R2] and naturally disappears once the bubble is gone.

Frames are paced by interpolating the profiles onto a uniform log10(t) grid, so
every decade of time gets equal screen time. (t is the pacing axis because it is
the only monotonic one -- rShell recedes once the shell recollapses.) The grid
is clipped at ``--t-start`` so the microscopic early energy phase, which spans
several decades below 0.01 pc, doesn't eat the whole animation. Pass ``--raw``
for the old one-frame-per-recorded-timestep behaviour.

Usage:
  python tools/make_density_profile_gif.py [run_dir] [-o out.gif]
         [--fps 15] [--frames 150] [--t-start 1e-3] [--raw [--stride N]]

``run_dir`` defaults to the bundled mock run (``outputs/mockOutput/mockFullrun``).
Unless ``-o`` is given, the GIF is
written to ``fig/density_profile/<parent-folder>__<run-name>.gif`` (the run's
parent folder + its own name, so GIFs from different sweeps don't collide).
Requires the pinned deps plus Pillow (matplotlib's PillowWriter).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.disable(logging.CRITICAL)

# Allow `python tools/make_density_profile_gif.py` to find the trinity package
# without requiring PYTHONPATH to be set on the command line.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from trinity._functions.unit_conversions import ndens_cgs2au  # noqa: E402

# Idea borrowed from paper/methods/figures/paper_densityProfile.py: overlay the
# static ambient cloud profile n(r) (core -> power-law -> ISM at the cloud edge)
# so the frames show the shell sweeping up cloud material. (Not assumed correct —
# recomputed here directly from the run's metadata, power-law profiles only.)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_RUN = REPO / "outputs" / "mockOutput" / "mockFullrun"
DEFAULT_FIG_DIR = REPO / "fig" / "density_profile"

# Colours for the two physical regions and the boundary markers.
C_BUBBLE = "#d1495b"  # hot, rarefied shocked-wind interior (R1 -> R2)
C_SHELL = "#2e6f95"  # dense swept-up shell (R2 -> rShell)
BOUNDS = [("R1", "#3a7d44"), ("R2", "#e08e0b"), ("rShell", "#6d6875")]


def load_frames(run_dir):
    """Read dictionary.jsonl and pull out the per-step profile data.

    Returns a list of dicts (one per timestep) with the bubble and shell radius
    and density arrays already converted to pc and cm^-3, plus the scalar
    boundaries and the time/phase used for the frame label.
    """
    jsonl = Path(run_dir) / "dictionary.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"no dictionary.jsonl in {run_dir}")

    frames = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        R1, R2, rShell = d["R1"], d["R2"], d["rShell"]

        # Bubble interior: clip the (possibly stale) arrays to the live [R1, R2]
        # window, then sort ascending in radius for plotting. In the momentum
        # phase R1 == R2 so nothing survives the clip and the segment vanishes.
        br = np.asarray(d["bubble_n_arr_r_arr"], dtype=float)
        bn = np.asarray(d["log_bubble_n_arr"], dtype=float)
        keep = np.isfinite(br) & np.isfinite(bn) & (br >= R1) & (br <= R2)
        br, bn = br[keep], bn[keep]
        order = np.argsort(br)
        bubble_r = br[order]
        bubble_n = np.power(10.0, bn[order]) / ndens_cgs2au

        # Shell: already ascending in radius.
        sr = np.asarray(d["shell_r_arr"], dtype=float)
        sn = np.asarray(d["log_shell_n_arr"], dtype=float)
        keep = np.isfinite(sr) & np.isfinite(sn)
        shell_r = sr[keep]
        shell_n = np.power(10.0, sn[keep]) / ndens_cgs2au

        frames.append(
            {
                "t": d["t_now"],
                "phase": d["current_phase"],
                "bubble_r": bubble_r,
                "bubble_n": bubble_n,
                "shell_r": shell_r,
                "shell_n": shell_n,
                "R1": R1,
                "R2": R2,
                "rShell": rShell,
            }
        )
    return frames


def load_ambient(run_dir, xlim):
    """Static ambient cloud profile n(r) [cm^-3] for a power-law cloud.

    Reads the run's metadata.json (nCore/nISM are stored in internal pc^-3,
    radii in pc) and reconstructs the initial cloud density: flat core inside
    rCore, n ∝ r^alpha out to rCloud, then the ISM floor. Returns
    (r_grid, n_cgs, rCloud) or None for non-power-law clouds / missing metadata.
    """
    meta_path = Path(run_dir) / "metadata.json"
    if not meta_path.exists():
        return None
    md = json.loads(meta_path.read_text())
    if md.get("dens_profile") != "densPL":
        return None  # only the power-law family is reconstructed here
    nCore = md["nCore"] / ndens_cgs2au
    nISM = md["nISM"] / ndens_cgs2au
    rCore = md["rCore"]
    rCloud = md["rCloud"]
    alpha = md.get("densPL_alpha", 0.0)

    r = np.geomspace(xlim[0], xlim[1], 600)
    n = np.full_like(r, nISM)
    core = r < rCore
    cloud = (r >= rCore) & (r < rCloud)
    n[core] = nCore
    n[cloud] = nCore * (r[cloud] / rCore) ** alpha
    return r, n, rCloud


def axis_limits(frames):
    """Fixed log-log limits covering every frame, so evolution is comparable."""
    r_lo, r_hi, n_lo, n_hi = np.inf, -np.inf, np.inf, -np.inf
    for f in frames:
        for r, n in ((f["bubble_r"], f["bubble_n"]), (f["shell_r"], f["shell_n"])):
            if r.size:
                r_lo = min(r_lo, r.min())
                r_hi = max(r_hi, r.max())
                n_lo = min(n_lo, n[n > 0].min() if np.any(n > 0) else n_lo)
                n_hi = max(n_hi, n.max())
        r_lo = min(r_lo, f["R1"])
        r_hi = max(r_hi, f["rShell"])
    # Pad by ~0.1 dex on each side and floor the density so the rarefied bubble
    # interior stays on-scale without dragging the axis to absurd values.
    xlim = (r_lo / 1.3, r_hi * 1.3)
    ylim = (max(n_lo / 2, 1e-2), n_hi * 3)
    return xlim, ylim


def _profile_signature(rec):
    """(has_bubble, has_shell): the profile 'topology' that must match for two
    records to be safely blended, so a bubble or shell never fades in/out of
    existence by interpolating across the energy->momentum collapse."""
    has_b = rec["R2"] > rec["R1"] * (1.0 + 1e-9) and rec["bubble_r"].size >= 2
    has_s = rec["rShell"] > rec["R2"] * (1.0 + 1e-9) and rec["shell_r"].size >= 2
    return has_b, has_s


def _normalized_logn(rec, nb, ns):
    """Resample each region's log10(n) onto a fixed fractional grid in [0, 1]:
    bubble fraction (r-R1)/(R2-R1), shell fraction (r-R2)/(rShell-R2). Putting
    both records' profiles on a common fractional grid is what lets us blend
    them while R1/R2/rShell move independently (keeps the contact discontinuity
    at R2 sharp). Returns (logn_b, logn_s); either is None where absent."""
    has_b, has_s = _profile_signature(rec)
    logn_b = logn_s = None
    if has_b:
        xb = (rec["bubble_r"] - rec["R1"]) / (rec["R2"] - rec["R1"])
        logn = np.log10(np.clip(rec["bubble_n"], 1e-300, None))
        o = np.argsort(xb)
        logn_b = np.interp(np.linspace(0.0, 1.0, nb), xb[o], logn[o])
    if has_s:
        xs = (rec["shell_r"] - rec["R2"]) / (rec["rShell"] - rec["R2"])
        logn = np.log10(np.clip(rec["shell_n"], 1e-300, None))
        o = np.argsort(xs)
        logn_s = np.interp(np.linspace(0.0, 1.0, ns), xs[o], logn[o])
    return logn_b, logn_s


def build_logt_frames(records, t_start, n_frames, nb=120, ns=120):
    """Interpolate the per-record profiles onto a uniform log10(t) grid.

    Pacing by log-t gives every decade of time equal screen time (t is the only
    monotonic axis -- rShell recedes when the shell recollapses, so radius can't
    pace the animation). The grid is clipped to ``t_start`` so the microscopic
    early energy phase, several decades at sub-0.01 pc, doesn't dominate.

    Within a phase, and where the bubble/shell topology matches, the bracketing
    records are blended: radii geometrically (constant visual speed on the log
    axis), log-densities linearly on the fractional grid. Across a phase boundary
    or a region appearing/disappearing the frame snaps to the nearer record. The
    records are adaptive samples of a continuous evolution, so blending within a
    phase reconstructs that evolution rather than inventing solver states; only
    the handful of boundary frames are non-blended snapshots.
    """
    # Strictly increasing t (drop the rare dt<=0 duplicate timestamps).
    recs = [records[0]]
    for r in records[1:]:
        if r["t"] > recs[-1]["t"]:
            recs.append(r)
    t = np.array([r["t"] for r in recs])

    # Cache each record's normalized profiles + topology once.
    for r in recs:
        r["_logn_b"], r["_logn_s"] = _normalized_logn(r, nb, ns)
        r["_sig"] = _profile_signature(r)

    t_start = max(t_start, t[0])
    t_grid = np.logspace(np.log10(t_start), np.log10(t[-1]), n_frames)
    grid_b = np.linspace(0.0, 1.0, nb)
    grid_s = np.linspace(0.0, 1.0, ns)

    frames = []
    for tg in t_grid:
        j = min(max(int(np.searchsorted(t, tg)), 1), len(t) - 1)
        a, b = recs[j - 1], recs[j]
        w = (np.log10(tg) - np.log10(a["t"])) / (np.log10(b["t"]) - np.log10(a["t"]))
        w = float(np.clip(w, 0.0, 1.0))
        # Snap across discontinuities: collapse the bracket to the nearer record.
        if a["phase"] != b["phase"] or a["_sig"] != b["_sig"]:
            a = b = a if w < 0.5 else b
            w = 0.0
        has_b, has_s = a["_sig"]

        # Geometric (log-linear) blend of the moving boundaries.
        R1 = a["R1"] ** (1 - w) * b["R1"] ** w
        R2 = a["R2"] ** (1 - w) * b["R2"] ** w
        rShell = a["rShell"] ** (1 - w) * b["rShell"] ** w

        if has_b:
            logn_b = (1 - w) * a["_logn_b"] + w * b["_logn_b"]
            bubble_r, bubble_n = R1 + grid_b * (R2 - R1), np.power(10.0, logn_b)
        else:
            bubble_r = bubble_n = np.array([])
        if has_s:
            logn_s = (1 - w) * a["_logn_s"] + w * b["_logn_s"]
            shell_r, shell_n = R2 + grid_s * (rShell - R2), np.power(10.0, logn_s)
        else:
            shell_r = shell_n = np.array([])

        frames.append(
            {
                "t": float(tg),
                "phase": (a if w < 0.5 else b)["phase"],
                "bubble_r": bubble_r,
                "bubble_n": bubble_n,
                "shell_r": shell_r,
                "shell_n": shell_n,
                "R1": R1,
                "R2": R2,
                "rShell": rShell,
            }
        )
    return frames


def render_gif(frames, run_dir, out_path, fps, note=None):
    """Render a list of prepared draw-frames (raw or interpolated) to a GIF."""
    xlim, ylim = axis_limits(frames)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("radius  [pc]")
    ax.set_ylabel(r"number density  $n$  [cm$^{-3}$]")

    # Static ambient cloud profile (faint) + the cloud edge, for context.
    ambient = load_ambient(run_dir, xlim)
    if ambient is not None:
        r_amb, n_amb, rCloud = ambient
        ax.plot(r_amb, n_amb, color="0.6", ls=":", lw=1.3, label="ambient cloud")
        ax.axvline(rCloud, color="0.6", ls=":", lw=1.0, alpha=0.7)

    (line_bubble,) = ax.plot([], [], "-", color=C_BUBBLE, lw=2.0, label="bubble (R1→R2)")
    (line_shell,) = ax.plot([], [], "-", color=C_SHELL, lw=2.0, label="shell (R2→rShell)")
    vlines = [ax.axvline(np.nan, color=c, ls="--", lw=1.2, alpha=0.8) for _, c in BOUNDS]
    # Proxy handles so the boundary lines show up in the legend with their names.
    for (name, c) in BOUNDS:
        ax.plot([], [], color=c, ls="--", lw=1.2, label=name)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    if note:
        # Label the pacing so the GIF is honest about interpolated frames.
        ax.text(0.985, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="0.5", style="italic")
    title = ax.set_title(" ", pad=10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.91)  # reserve headroom so the time title isn't clipped

    def update(i):
        f = frames[i]
        line_bubble.set_data(f["bubble_r"], f["bubble_n"])
        line_shell.set_data(f["shell_r"], f["shell_n"])
        for vl, (name, _) in zip(vlines, BOUNDS):
            vl.set_xdata([f[name], f[name]])
        title.set_text(f"t = {f['t']:.3f} Myr   ·   phase: {f['phase']}")
        return [line_bubble, line_shell, *vlines, title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return len(frames)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", default=str(DEFAULT_RUN), help="run output dir containing dictionary.jsonl")
    p.add_argument("-o", "--out", default=None, help="output GIF path (default fig/density_profile/<parent>__<run>.gif)")
    p.add_argument("--fps", type=int, default=15, help="frames per second (default 15)")
    p.add_argument("--frames", type=int, default=150, help="number of interpolated log-t frames (default 150)")
    p.add_argument("--t-start", type=float, default=1e-3, help="clip the log-t grid to start at this time [Myr] (default 1e-3)")
    p.add_argument("--raw", action="store_true", help="one frame per recorded timestep (no log-t interpolation)")
    p.add_argument("--stride", type=int, default=1, help="(raw mode only) use every Nth timestep")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if args.out:
        out_path = Path(args.out)
    else:
        # Name the GIF after the run's parent folder + its own name so outputs
        # from different sweeps land side by side in fig/ without colliding.
        out_path = DEFAULT_FIG_DIR / f"{run_dir.parent.name}__{run_dir.name}.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_frames(run_dir)
    if args.raw:
        frames = records[:: args.stride]
        note = "raw · frame-per-record"
    else:
        frames = build_logt_frames(records, args.t_start, args.frames)
        note = "log-t pacing · interpolated"
    n = render_gif(frames, run_dir, out_path, args.fps, note=note)
    print(f"wrote {out_path}  ({n} frames @ {args.fps} fps, {'raw' if args.raw else 'log-t interp'})")


if __name__ == "__main__":
    main()
