#!/usr/bin/env python3
"""Animated cage-vs-no-cage root-finding GIF for the steep run -- a PURE READ.

All the expensive physics (the real legacy/caged solve and the re-solved interior
profiles) is tabulated once by tabulate_cage.py into two COMMITTED csvs
(rootmap_cage_scalars.csv per segment + rootmap_cage_profiles.csv.gz long-format);
this script only reads those and renders, so it runs in seconds, is cheap to
re-style, and reproduces from git after the container is gone. Run
tabulate_cage.py first (needs the pinned venv); this script needs only
numpy + pandas + matplotlib + pillow.

Frame pacing: the run samples t very unevenly (dt ramps 3e-4 -> capped 5e-2, so
~35% of segments live in the first 3% of physical time). Playing one frame per
segment would dwell on the early phase. Frames are laid on a uniform grid in t
(linear): the late phase is already uniform (dt capped) so the grid is DENSER than
the segments there -- preserving sharp 1-segment features like the surge inflow --
while the over-dense early phase is decimated, fixing the dwell. (log-t was tried
first but under-samples the late phase and washes out the inflow spike.) The moving
star/square and the profile curves are INTERPOLATED onto the grid for smooth
motion; the accumulated (beta,delta) scatter and the residual history stay at the
REAL segment times (dots), so nothing fabricated hides the true samples.

"cage" means two different things across the panels -- labelled on each:
  * A/B/D show the cage as a SHADOW: the legacy/caged root-finder re-solved at each
    state ALONG the hybr trajectory (a counterfactual, "what would the cage return
    here?"). Because the hybr run reaches 4 Myr, the shadow spans 4 Myr too.
  * C and F show the cage as a REAL standalone run: it evolves under those clamped
    roots, expands to R2~2 pc (t~0.26 Myr), then RECOLLAPSES (v2<0, R2 falls back to
    ~1.5 pc, run ends t~0.44 Myr) -- while the hybr run expands to R2~37 pc at 4 Myr.
They are complementary: A/B say the cage gives bad/non-converged roots at every
state; C/F show the real caged run those roots produce turns over and collapses.

Six panels, current time = the frame's t grid point:
  LEFT  A : the (beta,delta) plane. Cyan box = the legacy clamp ("the cage").
            hybr roots (no cage) escape the box; the cage SHADOW roots (squares,
            re-solved along the hybr path) ride the edge. Dots = real segments
            (colour = time); a dashed connector marks the current clamp error.
  LEFT  B : residual g of the two arms vs t (g<1e-4 = converged). hybr converges;
            the cage shadow cannot (structurally forbidden the out-of-box root).
  LEFT  C : expansion trajectory v2 vs R2 (log-R2): the two REAL runs, caged
            (crimson) vs hybr (blue). They track together while decelerating, then
            diverge -- the caged bubble peaks at R2~2 pc and recollapses (v2<0,
            shaded) while hybr expands on to R2~37 pc.
  RIGHT D : interior velocity v(r) vs radial fraction (0=R1, 1=R2), cage SHADOW vs
            no cage (inflow = v<0; the cage's monotone solve hides the surge inflow).
  RIGHT E : Lmech_W / Lmech_SN / Lmech_total vs t, marker at the current t.
  RIGHT F : shell radii vs t (log-log), both REAL runs: R2 (solid, inner edge) and
            rShell (dashed, outer edge), caged (black) vs hybr (blue).

  python docs/dev/archive/betadelta/diagnostics/make_rootmap_gif.py
Writes rootmap_cage.gif.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib import cm as mcm  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_PREFIX = "rootmap_cage"  # steep run; --prefix selects another tabulation
BOX_B, BOX_D = (0.0, 1.0), (-1.0, 0.0)  # the cage (legacy clamp)
THRESH = 1e-4
NFRAMES = 150  # frames on the uniform (linear) t grid
FPS = 12
HYBR_C, CAGE_C = "#0072B2", "0.45"  # no-cage (blue) vs cage (grey)

_STYLE = HERE.parents[1] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams["text.usetex"] = False


def load(scalars, profiles, legacy_traj=None, hybr_traj=None):
    if not scalars.exists() or not profiles.exists():
        raise SystemExit(
            f"missing {scalars.name}/{profiles.name} -- run "
            f"`python {HERE.name}/tabulate_cage.py` first"
        )
    scal = pd.read_csv(scalars)
    prof = pd.read_csv(profiles)
    nseg = len(scal)
    nf = int((prof["segment"] == 0).sum())
    f_grid = prof["f"].to_numpy()[:nf]
    rs = lambda c: prof[c].to_numpy().reshape(nseg, nf)  # noqa: E731
    col = lambda c: scal[c].to_numpy() if c in scal else np.full(nseg, np.nan)  # noqa: E731
    lg = pd.read_csv(legacy_traj) if (legacy_traj and legacy_traj.exists()) else None
    hg = pd.read_csv(hybr_traj) if (hybr_traj and hybr_traj.exists()) else None
    tr = lambda df, c: df[c].to_numpy() if df is not None else None  # noqa: E731  traj column
    return dict(
        t=scal["t"].to_numpy(),
        hb=scal["beta_nocage"].to_numpy(),
        hd=scal["delta_nocage"].to_numpy(),
        cb=scal["beta_cage"].to_numpy(),
        cd=scal["delta_cage"].to_numpy(),
        g_h=scal["g_nocage"].to_numpy(),
        g_c=scal["g_cage"].to_numpy(),
        cage_ok=scal["cage_ok"].to_numpy().astype(bool),
        R2=scal["R2"].to_numpy(),
        v2=scal["v2"].to_numpy(),
        R1_h=scal["R1_nocage"].to_numpy(),
        R1_c=scal["R1_cage"].to_numpy(),
        R_IF=col("R_IF_nocage"),
        # panels C & F: the two REAL driven runs (caged / hybr), each from its own CSV
        leg_t=tr(lg, "t"),
        leg_R2=tr(lg, "R2"),
        leg_v2=tr(lg, "v2"),
        leg_rsh=tr(lg, "rShell"),
        hyb_t=tr(hg, "t"),
        hyb_R2=tr(hg, "R2"),
        hyb_v2=tr(hg, "v2"),
        hyb_rsh=tr(hg, "rShell"),
        lw=scal["Lmech_W"].to_numpy() / 1e8,
        lsn=scal["Lmech_SN"].to_numpy() / 1e8,
        lt=scal["Lmech_total"].to_numpy() / 1e8,
        f_grid=f_grid,
        v_h=rs("v_nocage"),
        v_c=rs("v_cage"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="CSV/gif basename prefix")
    ap.add_argument("--label", default="", help="optional config label shown in the title")
    args = ap.parse_args()
    lbl = f" — {args.label}" if args.label else ""
    d = load(
        HERE / f"{args.prefix}_scalars.csv",
        HERE / f"{args.prefix}_profiles.csv.gz",
        HERE / "steep_legacy_traj.csv",
        HERE / "steep_hybr_traj.csv",
    )
    t = d["t"]
    hb, hd, cb, cd = d["hb"], d["hd"], d["cb"], d["cd"]
    g_h, g_c, cage_ok = d["g_h"], d["g_c"], d["cage_ok"]
    R2, v2 = d["R2"], d["v2"]
    leg_t, leg_R2, leg_v2, leg_rsh = d["leg_t"], d["leg_R2"], d["leg_v2"], d["leg_rsh"]
    hyb_t, hyb_R2, hyb_v2, hyb_rsh = d["hyb_t"], d["hyb_R2"], d["hyb_v2"], d["hyb_rsh"]
    lw, lsn, lt = d["lw"], d["lsn"], d["lt"]
    f_grid, v_h, v_c = d["f_grid"], d["v_h"], d["v_c"]
    # discrete time colour (fixed bins) -> a point's colour never changes AND the GIF
    # palette stays stable frame-to-frame (continuous viridis re-quantizes -> shimmer).
    NTBINS = 10
    bounds = np.linspace(t.min(), t.max(), NTBINS + 1)
    cmap_t = plt.get_cmap("viridis", NTBINS)
    tnorm = BoundaryNorm(bounds, cmap_t.N)

    # ---- frame grid: uniform in t; denser than the (capped-dt) late segments so the
    # surge inflow survives, decimating the over-dense early phase ----
    tg = np.linspace(t[0], t[-1], NFRAMES)
    lin = lambda y: np.interp(tg, t, y)  # noqa: E731  smooth current-marker value
    logi = lambda y: 10 ** np.interp(tg, t, np.log10(np.clip(y, 1e-7, None)))  # noqa: E731
    hb_f, hd_f, cb_f, cd_f = lin(hb), lin(hd), lin(cb), lin(cd)
    R2_f, v2_f = lin(R2), lin(v2)
    gh_f, gc_f = logi(g_h), logi(g_c)
    cage_f = lin(cage_ok.astype(float)) > 0.999  # cage curves valid at this frame?

    def prof_interp(P):  # interpolate each radial-fraction column onto tg
        return np.column_stack([np.interp(tg, t, P[:, j]) for j in range(P.shape[1])])

    vh_f, vc_f = prof_interp(v_h), prof_interp(v_c)

    # ---- static axis limits (from the real data, so nothing jumps) ----
    allb = np.concatenate([hb, cb[cage_ok]])
    alld = np.concatenate([hd, cd[cage_ok]])
    xlo, xhi = min(allb.min(), BOX_B[0]), max(allb.max(), BOX_B[1])
    ylo, yhi = min(alld.min(), BOX_D[0]), max(alld.max(), BOX_D[1])
    xpad, ypad = 0.08 * (xhi - xlo) + 0.05, 0.08 * (yhi - ylo) + 0.05
    XLIM, YLIM = (xlo - xpad, xhi + xpad), (ylo - ypad, yhi + ypad)
    vlo = float(np.nanmin([np.nanmin(v_h), np.nanmin(v_c)]))
    vhi = float(np.nanmax([np.nanmax(v_h), np.nanmax(v_c)]))
    VLIM = (min(vlo, -0.2) - 0.05 * abs(vhi), vhi * 1.05)
    # panels C & F span both real runs (hybr to R2~37; caged peaks ~2 pc then recollapses)
    R2all = np.concatenate([a for a in (hyb_R2, leg_R2) if a is not None])
    v2all = np.concatenate([a for a in (hyb_v2, leg_v2) if a is not None])
    R2MAX = float(np.nanmax(R2all)) * 1.05  # panel C x-axis (R2)
    R2MIN = float(np.nanmin(R2all)) * 0.9
    v2hi = float(np.nanmax(v2all))
    V2LIM = (float(np.nanmin(v2all)) - 0.06 * abs(v2hi), v2hi * 1.05)
    rfall = np.concatenate([a for a in (hyb_R2, hyb_rsh, leg_R2, leg_rsh) if a is not None])
    RFMAX = float(np.nanmax(rfall)) * 1.25  # panel F y-axis (log)
    RFMIN = float(np.nanmin(rfall)) * 0.8

    fig, axd = plt.subplot_mosaic(
        [["A", "D"], ["A", "D"], ["B", "E"], ["C", "F"]],
        figsize=(14, 9.5),
        width_ratios=[1.5, 1.0],
        height_ratios=[1.2, 1.2, 0.9, 0.9],
        constrained_layout=True,
    )
    aA, aB, aC, aD, aE, aF = (axd[k] for k in "ABCDEF")
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.965))  # reserve bottom strip for the note
    # one persistent colourbar for the time-coloured scatter (created once, NOT in
    # update(), so it neither stacks nor rescales -> no colour blinking).
    sm = mcm.ScalarMappable(norm=tnorm, cmap=cmap_t)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=aA, fraction=0.045, pad=0.02)
    cbar.set_label("t  [Myr]", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    # persistent note: the two senses of "cage" in this figure (added once -> no flicker).
    fig.text(
        0.5,
        0.012,
        'cage shown two ways — A/B/D: cage root-finder re-solved along the hybr path '
        "(counterfactual).    C/F: the two real runs — caged expands to ~2 pc then "
        "recollapses (v₂<0), while hybr expands to ~37 pc.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.35",
    )

    def update(i):
        tc = tg[i]
        rm = t <= tc + 1e-12  # real segments revealed so far
        out = not (BOX_B[0] <= hb_f[i] <= BOX_B[1] and BOX_D[0] <= hd_f[i] <= BOX_D[1])

        # ---- A: (beta, delta) plane, cage vs no cage ----
        aA.clear()
        aA.add_patch(
            Rectangle((BOX_B[0], BOX_D[0]), 1, 1, fill=False, edgecolor="cyan", lw=2.5, zorder=2)
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
        aA.plot(hb, hd, color="0.85", lw=1.0, zorder=1)  # full hybr path (context)
        rmc = rm & cage_ok
        aA.scatter(
            cb[rmc],
            cd[rmc],
            c=t[rmc],
            cmap=cmap_t,
            norm=tnorm,
            marker="s",
            s=18,
            edgecolor="0.3",
            lw=0.3,
            zorder=3,
            label="cage shadow (along hybr path)",
        )
        aA.scatter(
            hb[rm],
            hd[rm],
            c=t[rm],
            cmap=cmap_t,
            norm=tnorm,
            s=24,
            edgecolor="k",
            lw=0.3,
            zorder=4,
            label="no cage (hybr)",
        )
        if out and cage_f[i]:
            aA.plot(
                [cb_f[i], hb_f[i]], [cd_f[i], hd_f[i]], color="crimson", lw=1.4, ls="--", zorder=5
            )
        if cage_f[i]:
            aA.plot(
                cb_f[i],
                cd_f[i],
                marker="s",
                ms=12,
                mfc="crimson",
                mec="k",
                mew=1.2,
                ls="none",
                zorder=6,
            )
        aA.plot(
            hb_f[i],
            hd_f[i],
            marker="*",
            ms=22,
            mfc="#ffd000",
            mec="k",
            mew=1.2,
            ls="none",
            zorder=7,
        )
        aA.set_xlim(*XLIM)
        aA.set_ylim(*YLIM)
        aA.set_xlabel(r"$\beta$")
        aA.set_ylabel(r"$\delta$")
        tag = "OUTSIDE the cage" if out else "inside the cage"
        aA.set_title(
            f"Root finding with vs without the cage{lbl}  (t={tc:.3g} Myr; {tag})\n"
            f"hybr  β={hb_f[i]:+.2f}, δ={hd_f[i]:+.2f}   |   cage  β={cb_f[i]:+.2f}, δ={cd_f[i]:+.2f}",
            fontsize=10,
        )
        aA.text(
            0.02,
            0.98,
            "dots = real segments (colour = time)\n"
            "squares = cage re-solved along this path (counterfactual)",
            transform=aA.transAxes,
            fontsize=8,
            va="top",
            color="0.45",
        )
        aA.legend(loc="lower right", fontsize=8)
        aA.grid(alpha=0.3)

        # ---- B: residual g of the two arms vs t (real history + current marker) ----
        aB.clear()
        aB.plot(
            t[rm], np.clip(g_h[rm], 1e-7, None), "-o", ms=2.5, color=HYBR_C, label="no cage (hybr)"
        )
        aB.plot(
            t[rm],
            np.clip(g_c[rm], 1e-7, None),
            "-s",
            ms=2.5,
            color="crimson",
            label="cage shadow (along hybr path)",
        )
        aB.plot(tc, gh_f[i], "o", ms=8, mfc="#ffd000", mec="k", zorder=5)
        aB.plot(tc, gc_f[i], "s", ms=8, mfc="crimson", mec="k", zorder=5)
        aB.axhline(THRESH, color="k", ls="--", lw=1, label=f"converged < {THRESH:g}")
        aB.set_yscale("log")
        aB.set_ylim(4e-7, 1e2)
        aB.set_xlim(t.min(), t.max())
        aB.set_xlabel("t  [Myr]")
        aB.set_ylabel(r"residual $g$")
        aB.set_title("convergence — two arms", fontsize=9)
        aB.legend(fontsize=7, loc="upper left", ncol=1)
        aB.grid(alpha=0.3)

        # ---- C: expansion trajectory v2 vs R2 — the two REAL driven runs ----
        # caged (crimson) and hybr (blue), each from its own run CSV. log-R2 so the
        # caged hook (R2<2) and the full hybr path (R2->37) are both legible. They
        # track together while decelerating, then DIVERGE: the caged bubble peaks at
        # R2~2 pc and recollapses (v2<0, shaded), while hybr keeps expanding.
        aC.clear()
        aC.axhspan(V2LIM[0], 0.0, color="r", alpha=0.06)  # v2<0 = recollapse / infall
        aC.axhline(0.0, color="k", lw=0.8)
        aC.plot(hyb_R2, hyb_v2, color=HYBR_C, lw=1.8, label="hybr (free β,δ)")
        if leg_R2 is not None:
            aC.plot(leg_R2, leg_v2, color="crimson", lw=2.0, label="caged (clamped β,δ)")
            aC.plot(leg_R2[-1], leg_v2[-1], marker="X", ms=11, mfc="crimson", mec="k", ls="none")
            ip = int(np.argmax(leg_R2))  # turnover point
            aC.annotate(
                f"caged peaks R₂={leg_R2[ip]:.1f} pc\nthen recollapses (v₂<0)",
                xy=(leg_R2[ip], leg_v2[ip]),
                xytext=(leg_R2[ip] * 0.16, V2LIM[0] + 0.30 * (V2LIM[1] - V2LIM[0])),
                fontsize=7,
                color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1),
            )
        aC.plot(R2_f[i], v2_f[i], "o", ms=8, mfc="#ffd000", mec="k", zorder=6)
        # animated caged marker: rides the real caged run (t<=0.44 Myr) then parks
        # on the final recollapsed state for the rest of the animation.
        if leg_R2 is not None:
            if tc <= leg_t[-1]:
                cr = float(np.interp(tc, leg_t, leg_R2))
                cv = float(np.interp(tc, leg_t, leg_v2))
            else:
                cr, cv = float(leg_R2[-1]), float(leg_v2[-1])
            aC.plot(cr, cv, marker="s", ms=9, mfc="crimson", mec="k", mew=1.0, ls="none", zorder=7)
        aC.set_xscale("log")
        aC.set_xlim(R2MIN, R2MAX)
        aC.set_ylim(*V2LIM)
        aC.set_xlabel(r"$R_2$  [pc]")
        aC.set_ylabel(r"$v_2$  [pc/Myr]")
        aC.set_title("expansion trajectory: caged vs hybr (real runs)", fontsize=9)
        aC.legend(fontsize=7, loc="upper right")
        aC.grid(alpha=0.3, which="both")

        # ---- D: interior velocity vs radial fraction ----
        aD.clear()
        aD.axhline(0.0, color="k", lw=0.8)
        aD.axhspan(VLIM[0], 0.0, color="r", alpha=0.06)
        aD.plot(
            f_grid,
            vh_f[i],
            color=HYBR_C,
            lw=1.9,
            label=f"no cage  (v_min={np.nanmin(vh_f[i]):+.2f})",
        )
        if cage_f[i]:
            aD.plot(
                f_grid,
                vc_f[i],
                color=CAGE_C,
                lw=1.9,
                ls="--",
                label=f"cage shadow  (v_min={np.nanmin(vc_f[i]):+.2f})",
            )
        aD.set_xlim(0, 1)
        aD.set_ylim(*VLIM)
        aD.set_xlabel("radial fraction  (0 = R1, 1 = R2)")
        aD.set_ylabel("v(r)  [pc/Myr]")
        inflow = float(np.nanmin(vh_f[i])) < -0.01
        aD.set_title(
            f"interior velocity  (R2={R2_f[i]:.2f} pc)" + ("  — INFLOW (v<0)" if inflow else ""),
            fontsize=10,
            color="#b30000" if inflow else "k",
        )
        aD.legend(fontsize=8, loc="upper left")
        aD.grid(alpha=0.3)

        # ---- E: Lmech vs t ----
        aE.clear()
        aE.plot(t, lt, color="k", lw=1.6, ls="--", label=r"$L_{\rm tot}$")
        aE.plot(t, lw, color=HYBR_C, lw=1.4, label=r"$L_{\rm W}$ (wind)")
        aE.plot(t, lsn, color="#9467bd", lw=1.4, label=r"$L_{\rm SN}$")
        aE.axvline(tc, color="crimson", lw=1.4)
        aE.set_xlim(t.min(), t.max())
        aE.set_xlabel("t  [Myr]")
        aE.set_ylabel(r"$L_{\rm mech}$ [$10^8$]")
        aE.set_title("feedback power", fontsize=9)
        aE.legend(fontsize=7, loc="upper left", ncol=3)
        aE.grid(alpha=0.3)

        # ---- F: shell radii vs t — both REAL runs, R2 (solid) & rShell (dashed) ----
        # caged (black) vs hybr (blue). log-log: the two runs differ ~20x in both
        # radius and lifetime, so log axes show the caged recollapse (R2 turns over
        # ~2 pc, ends t~0.44 Myr) alongside the hybr run climbing to ~37 pc at 4 Myr.
        aF.clear()
        aF.plot(hyb_t, hyb_R2, color=HYBR_C, lw=1.7, label=r"hybr $R_2$")
        aF.plot(hyb_t, hyb_rsh, color=HYBR_C, lw=1.4, ls="--", label=r"hybr $r_{\rm shell}$")
        if leg_R2 is not None:
            aF.plot(leg_t, leg_R2, color="k", lw=1.7, label=r"caged $R_2$")
            aF.plot(leg_t, leg_rsh, color="k", lw=1.4, ls="--", label=r"caged $r_{\rm shell}$")
            aF.plot(leg_t[-1], leg_R2[-1], marker="X", ms=9, mfc="k", mec="w", mew=0.8, ls="none")
            aF.annotate(
                "caged ends\n(recollapse)",
                xy=(leg_t[-1], leg_R2[-1]),
                xytext=(leg_t[-1] * 1.3, leg_R2[-1] * 0.28),
                fontsize=7,
                color="0.2",
                arrowprops=dict(arrowstyle="->", color="0.2", lw=1),
            )
        aF.axvline(tc, color="crimson", lw=1.0, alpha=0.6)
        aF.plot(tc, R2_f[i], "o", ms=7, mfc="#ffd000", mec="k", zorder=5)
        aF.set_xscale("log")
        aF.set_yscale("log")
        aF.set_xlim(min(float(leg_t[0]), float(hyb_t[0])) * 0.8, float(hyb_t[-1]) * 1.1)
        aF.set_ylim(RFMIN, RFMAX)
        aF.set_xlabel("t  [Myr]")
        aF.set_ylabel("radius  [pc]")
        aF.set_title("shell radii: caged vs hybr (R₂ solid, r_shell dashed)", fontsize=9)
        aF.legend(fontsize=7, loc="upper left", ncol=2)
        aF.grid(alpha=0.3, which="both")
        return []

    anim = FuncAnimation(fig, update, frames=NFRAMES, interval=1000 / FPS, blit=False)
    out = HERE / f"{args.prefix}.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=85)
    plt.close(fig)
    print(f"wrote {out}  ({NFRAMES} frames, linear-t paced)")


if __name__ == "__main__":
    main()
