#!/usr/bin/env python3
"""Inspect TRINITY_BUBBLE_DIAG capture files.

Usage:
    python inspect_bubble_diag.py <dir-or-glob-or-files>...
    (default: ./bubble_diag/*.npz)

Prints a compact table for all events, then a DETAILED breakdown for every
non-'boundary_transient' event (the ones that decide the fix), localizing each
violation against the geometry (cloud radius, R1/R2, grid stitches, the
over-refined outer band). If matplotlib is available, writes a 2x2 PNG per
detailed event next to the .npz: T profile + reference lines, a zoom on the
violation, grid spacing (dr), and the LSODA step size (when captured).

Reads the enriched fields (rCloud/rCore, info_* LSODA diagnostics) when
present and degrades gracefully on older captures that lack them.
"""
import sys, os, glob
import numpy as np

CIE_SWITCH = 10 ** 5.5      # K  (non-CIE <-> CIE cooling boundary)
COOL_SWITCH = 1e4           # K  (no cooling below this)

args = sys.argv[1:] or ["bubble_diag/*.npz"]
files = []
for a in args:
    if os.path.isdir(a):
        files += sorted(glob.glob(os.path.join(a, "*.npz")))
    elif os.path.isfile(a):
        files.append(a)
    else:
        files += sorted(glob.glob(a))
files = sorted(set(files))
if not files:
    print("no .npz files found for:", args); sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


def _get(d, key, default=None):
    """Read an npz field if present, else default (back-compat with old files)."""
    return d[key] if key in d.files else default


def _idx_at_r(r, value):
    return int(np.argmin(np.abs(r - value)))


def load(f):
    d = np.load(f, allow_pickle=True)
    T = np.asarray(d["T"], float)
    r = np.asarray(d["r"], float)
    n = T.size
    diffs = np.diff(T)
    bad = np.where(diffs < 0)[0]                     # downward steps (increasing profile)
    cmax = np.maximum.accumulate(T)
    dd = (cmax - T) / np.maximum(np.abs(cmax), 1e-300)
    i_dd = int(np.argmax(dd))
    ndips = (1 + int(np.sum(np.diff(bad) > 1))) if bad.size else 0   # contiguous runs

    # grid spacing + structure (orientation-agnostic magnitude)
    dr = np.abs(np.diff(r))
    med_dr = float(np.median(dr)) if dr.size else 0.0
    # "over-dense" = near-duplicate radii: relative spacing far below normal.
    # The over-refined outer band sits at dr/r ~ 1e-9; a normal grid is
    # ~1e-5..1e-3, so a 1e-7 cut cleanly separates them.
    reldr = dr / np.maximum(np.abs(r[:-1]), 1e-300) if dr.size else np.array([])
    overdense = reldr < 1e-7
    if overdense.any():
        od_idx = np.where(overdense)[0]
        od_lo, od_hi = int(od_idx.min()), int(od_idx.max())
    else:
        od_lo = od_hi = -1
    # grid-stitch indices: large jumps in consecutive spacing ratio
    stitches = []
    if dr.size > 2:
        ratio = dr[1:] / np.maximum(dr[:-1], 1e-300)
        logr = np.abs(np.log(np.maximum(ratio, 1e-300)))
        cand = np.where(logr > 0.5)[0] + 1
        # collapse near-adjacent candidates to representative indices
        for c in cand:
            if not stitches or c - stitches[-1] > 50:
                stitches.append(int(c))

    rCloud = _get(d, "rCloud"); rCloud = float(rCloud) if rCloud is not None else None
    rCore = _get(d, "rCore"); rCore = float(rCore) if rCore is not None else None
    info_hu = _get(d, "info_hu")
    info_hu = np.asarray(info_hu, float) if info_hu is not None else None
    info_nst = _get(d, "info_nst")
    info_nst = np.asarray(info_nst, float) if info_nst is not None else None
    info_mused = _get(d, "info_mused")
    info_mused = np.asarray(info_mused, float) if info_mused is not None else None

    tail = T[-max(10, n // 100):]
    return dict(
        f=os.path.basename(f), path=f, mode=str(d["mode"]),
        beta=float(d["beta"]), delta=float(d["delta"]),
        R1=float(_get(d, "R1", np.nan)), R2=float(d["R2"]),
        Eb=float(d["Eb"]), t=float(d["t_now"]),
        dMdt=float(d["bubble_dMdt"]), ier=int(_get(d, "ier", -999)),
        msg=str(_get(d, "message", "")),
        n=n, T=T, r=r, diffs=diffs, bad=bad, dd=dd, i_dd=i_dd,
        maxdd=float(dd[i_dd]), ndips=ndips,
        Tmin=float(T.min()), Tend=float(T[-1]), tailmin=float(tail.min()),
        dr=dr, med_dr=med_dr, od_lo=od_lo, od_hi=od_hi, stitches=stitches,
        rCloud=rCloud, rCore=rCore, info_hu=info_hu,
        info_nst=info_nst, info_mused=info_mused,
    )


def fmt_frac(i, n):
    return f"{i}/{n} ({i/n*100:.2f}%)"


def verdict(e):
    """Where does the worst violation sit, geometrically?"""
    i, n = e["i_dd"], e["n"]
    tags = []
    if e["od_lo"] >= 0 and e["od_lo"] <= i <= e["od_hi"]:
        tags.append("in over-dense outer band")
    if e["rCloud"] is not None:
        ic = _idx_at_r(e["r"], e["rCloud"])
        if abs(i - ic) < 0.005 * n:
            tags.append("near rCloud")
    if e["stitches"] and min(abs(i - s) for s in e["stitches"]) < 0.005 * n:
        tags.append("near a grid stitch")
    if not tags:
        tags.append("mid-profile (not boundary/stitch) — inspect for real inversion")
    return "; ".join(tags)


evs = [load(f) for f in files]

print("=" * 104)
print(f"{'file':28s} {'mode':18s} {'beta':>6s} {'delta':>6s} {'t':>7s} "
      f"{'maxdrawdn':>9s} {'#dips':>5s} {'bad@[lo..hi]frac':>18s} {'Tend':>9s} {'tailmin':>9s}")
print("-" * 104)
for e in evs:
    lo = e["bad"].min() / e["n"] if e["bad"].size else 0
    hi = e["bad"].max() / e["n"] if e["bad"].size else 0
    tag = e["mode"].split("(")[0][:18]
    print(f"{e['f'][:28]:28s} {tag:18s} {e['beta']:6.3f} {e['delta']:6.3f} {e['t']:7.4f} "
          f"{e['maxdd']:9.2e} {e['ndips']:5d} {lo*100:7.3f}..{hi*100:6.3f}% "
          f"{e['Tend']:9.2e} {e['tailmin']:9.2e}")

# DETAILED breakdown for the events that matter
detail = [e for e in evs if not e["mode"].startswith("boundary_transient")]
print("\n" + "=" * 104)
print(f"DETAILED: {len(detail)} non-boundary event(s)")
for e in detail:
    T, r, bad, n = e["T"], e["r"], e["bad"], e["n"]
    print("\n" + "-" * 94)
    print(f"{e['f']}  mode={e['mode']}")
    print(f"  beta={e['beta']:.4f} delta={e['delta']:.4f}  R1={e['R1']:.4e} R2={e['R2']:.4e} "
          f"Eb={e['Eb']:.4e} t={e['t']:.4e}  dMdt={e['dMdt']:.4e}  msg={e['msg']!r}")
    print(f"  n={n}  T: [{T[0]:.4e} .. {T[-1]:.4e}]  min={e['Tmin']:.4e}  tailmin={e['tailmin']:.4e}")
    if bad.size:
        print(f"  wrong-direction steps: {bad.size}  in {e['ndips']} contiguous run(s)")
        print(f"    index span: {fmt_frac(int(bad.min()), n)} .. {fmt_frac(int(bad.max()), n)}")
        print(f"    T over those steps: [{T[bad].min():.4e} .. {T[bad].max():.4e}]   "
              f"(boundary≈3e4; CIE switch≈3.16e5)")
    print(f"  worst cumulative drawdown: {e['maxdd']:.3e} at idx {fmt_frac(e['i_dd'], n)}  "
          f"T_there={T[e['i_dd']]:.4e}  r_there={r[e['i_dd']]:.4e}")
    if e["med_dr"] > 0:
        band = (f"idx {e['od_lo']}..{e['od_hi']} ({e['od_lo']/n*100:.1f}..{e['od_hi']/n*100:.1f}%)"
                if e["od_lo"] >= 0 else "none")
        stitch = f"; stitches~{e['stitches'][:4]}" if e["stitches"] else ""
        print(f"  grid: median dr={e['med_dr']:.3e} pc; over-dense band {band}{stitch}")
    if e["rCloud"] is not None:
        extra = (f"; rCore={e['rCore']:.4f} -> idx {_idx_at_r(r, e['rCore'])}"
                 if e["rCore"] is not None else "")
        print(f"  rCloud={e['rCloud']:.4f} pc -> idx {_idx_at_r(r, e['rCloud'])}{extra}")
    if e["info_nst"] is not None and e["info_nst"].size:
        nsteps = int(np.max(e["info_nst"]))
        method = ("Adams" if (e["info_mused"] is not None and np.all(e["info_mused"] == 1))
                  else "mixed/BDF")
        print(f"  LSODA: {nsteps} internal steps for {n} output points "
              f"(~{n / max(nsteps, 1):.0f}x dense-output interpolation), method={method}")
    print(f"  VERDICT: worst violation is {verdict(e)}")
    lo = max(0, e['i_dd'] - 4); hi = min(n, e['i_dd'] + 6)
    print(f"  T[{lo}:{hi}] = " + np.array2string(T[lo:hi], precision=4, floatmode='maxprec'))

    if HAVE_PLT:
        png = e['path'].replace('.npz', '.png')
        idx = np.arange(n)
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))

        def refs(a, with_h=False):
            if e["od_lo"] >= 0:
                a.axvspan(e["od_lo"], e["od_hi"], color='orange', alpha=0.12,
                          label='over-dense band')
            for s in e["stitches"][:6]:
                a.axvline(s, color='gray', ls=':', lw=0.7)
            if e["rCloud"] is not None:
                a.axvline(_idx_at_r(r, e["rCloud"]), color='g', ls='--', lw=1, label='rCloud')
            a.axvline(e['i_dd'], color='r', ls='--', lw=1, label='worst dip')
            if with_h:
                a.axhline(CIE_SWITCH, color='purple', ls=':', lw=0.8, label='CIE switch 3.16e5')
                a.axhline(COOL_SWITCH, color='brown', ls=':', lw=0.8, label='cool switch 1e4')

        ax[0, 0].semilogy(idx, T, lw=0.8); refs(ax[0, 0], with_h=True)
        ax[0, 0].set_title(f"{e['f']}\nfull T profile"); ax[0, 0].set_xlabel("index (outer→inner)")
        ax[0, 0].set_ylabel("T [K]"); ax[0, 0].legend(fontsize=7, loc='lower right')

        w = slice(max(0, (bad.min() if bad.size else e['i_dd']) - 20),
                  ((bad.max() if bad.size else e['i_dd']) + 40))
        ax[0, 1].plot(idx[w], T[w], '.-', ms=3)
        ax[0, 1].plot(e['i_dd'], T[e['i_dd']], 'ro', ms=6, label='worst')
        ax[0, 1].set_title("zoom on violation"); ax[0, 1].set_xlabel("index")
        ax[0, 1].set_ylabel("T [K]"); ax[0, 1].legend(fontsize=7)

        ax[1, 0].semilogy(idx[:-1], e["dr"], lw=0.8); refs(ax[1, 0])
        ax[1, 0].set_title("grid spacing dr vs index"); ax[1, 0].set_xlabel("index")
        ax[1, 0].set_ylabel("dr [pc]"); ax[1, 0].legend(fontsize=7, loc='lower right')

        if e["info_hu"] is not None and e["info_hu"].size:
            # the integrator runs over decreasing r, so hu < 0; plot the magnitude
            hu = np.abs(e["info_hu"])
            ax[1, 1].semilogy(np.arange(hu.size), np.where(hu > 0, hu, np.nan), lw=0.8)
            if e["od_lo"] >= 0:
                ax[1, 1].axvspan(e["od_lo"], e["od_hi"], color='orange', alpha=0.12)
            ax[1, 1].axvline(e['i_dd'], color='r', ls='--', lw=1)
            ax[1, 1].set_title("LSODA |step size| (|info_hu|) vs index")
            ax[1, 1].set_ylabel("|hu| [pc]")
        else:
            ax[1, 1].plot(idx, e["dd"], lw=0.8)
            ax[1, 1].axvline(e['i_dd'], color='r', ls='--', lw=1)
            ax[1, 1].set_title("cumulative rel. drawdown (no info_hu in file)")
            ax[1, 1].set_ylabel("drawdown")
        ax[1, 1].set_xlabel("index")

        fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
        print(f"  plot -> {png}")
print("\nDone.")
