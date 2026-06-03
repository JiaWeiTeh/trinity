#!/usr/bin/env python3
"""Inspect TRINITY_BUBBLE_DIAG capture files.

Usage:
    python inspect_bubble_diag.py <dir-or-glob-or-files>...
    (default: ./bubble_diag/*.npz)

Prints a compact table for all events, then a DETAILED breakdown for every
non-'boundary_transient' event (the ones that decide the fix). If matplotlib
is available, also writes a PNG per detailed event next to the .npz.
"""
import sys, os, glob
import numpy as np

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


def load(f):
    d = np.load(f, allow_pickle=True)
    T = np.asarray(d["T"], float)
    r = np.asarray(d["r"], float)
    n = T.size
    diffs = np.diff(T)
    bad = np.where(diffs < 0)[0]                     # downward steps (for an increasing profile)
    cmax = np.maximum.accumulate(T)
    dd = (cmax - T) / np.maximum(np.abs(cmax), 1e-300)
    i_dd = int(np.argmax(dd))
    ndips = (1 + int(np.sum(np.diff(bad) > 1))) if bad.size else 0   # contiguous runs of bad steps
    tail = T[-max(10, n // 100):]
    return dict(
        f=os.path.basename(f), mode=str(d["mode"]),
        beta=float(d["beta"]), delta=float(d["delta"]),
        R2=float(d["R2"]), Eb=float(d["Eb"]), t=float(d["t_now"]),
        dMdt=float(d["bubble_dMdt"]), ier=int(d["ier"]),
        msg=str(d["message"]) if "message" in d.files else "",
        n=n, T=T, r=r, diffs=diffs, bad=bad, dd=dd, i_dd=i_dd,
        maxdd=float(dd[i_dd]), ndips=ndips,
        Tmin=float(T.min()), Tend=float(T[-1]), tailmin=float(tail.min()),
    )


def fmt_frac(i, n): return f"{i}/{n} ({i/n*100:.2f}%)"

evs = [load(f) for f in files]

print("="*100)
print(f"{'file':28s} {'mode':18s} {'beta':>6s} {'delta':>6s} {'t':>7s} "
      f"{'maxdrawdn':>9s} {'#dips':>5s} {'bad@[lo..hi]frac':>18s} {'Tend':>9s} {'tailmin':>9s}")
print("-"*100)
for e in evs:
    lo = e["bad"].min()/e["n"] if e["bad"].size else 0
    hi = e["bad"].max()/e["n"] if e["bad"].size else 0
    tag = e["mode"].split("(")[0][:18]
    print(f"{e['f'][:28]:28s} {tag:18s} {e['beta']:6.3f} {e['delta']:6.3f} {e['t']:7.4f} "
          f"{e['maxdd']:9.2e} {e['ndips']:5d} {lo*100:7.3f}..{hi*100:6.3f}% "
          f"{e['Tend']:9.2e} {e['tailmin']:9.2e}")

# DETAILED breakdown for the events that matter
detail = [e for e in evs if not e["mode"].startswith("boundary_transient")]
print("\n" + "="*100)
print(f"DETAILED: {len(detail)} non-boundary event(s)")
for e in detail:
    T, r, bad, n = e["T"], e["r"], e["bad"], e["n"]
    print("\n" + "-"*90)
    print(f"{e['f']}  mode={e['mode']}")
    print(f"  beta={e['beta']:.4f} delta={e['delta']:.4f}  R2={e['R2']:.4e} Eb={e['Eb']:.4e} "
          f"t={e['t']:.4e}  dMdt={e['dMdt']:.4e}  ier={e['ier']} msg={e['msg']!r}")
    print(f"  n={n}  T: [{T[0]:.4e} .. {T[-1]:.4e}]  min={e['Tmin']:.4e}  tailmin={e['tailmin']:.4e}")
    if bad.size:
        print(f"  wrong-direction steps: {bad.size}  in {e['ndips']} contiguous run(s)")
        print(f"    index span: {fmt_frac(int(bad.min()),n)} .. {fmt_frac(int(bad.max()),n)}")
        print(f"    T over those steps: [{T[bad].min():.4e} .. {T[bad].max():.4e}]   "
              f"(boundary≈3e4; CIE switch≈3.16e5)")
    print(f"  worst cumulative drawdown: {e['maxdd']:.3e} at idx {fmt_frac(e['i_dd'],n)}  "
          f"T_there={T[e['i_dd']]:.4e}  r_there={r[e['i_dd']]:.4e}")
    # show the profile around the worst dip
    lo = max(0, e['i_dd']-4); hi = min(n, e['i_dd']+6)
    print(f"  T[{lo}:{hi}] = " + np.array2string(T[lo:hi], precision=4, floatmode='maxprec'))
    if HAVE_PLT:
        png = os.path.join(os.path.dirname(files[0]) or ".", e['f'].replace('.npz', '.png'))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].semilogy(np.arange(n), T); ax[0].set_title(f"{e['f']}\nfull T profile")
        ax[0].set_xlabel("index (outer→inner)"); ax[0].set_ylabel("T [K]")
        w = slice(max(0, bad.min()-20) if bad.size else 0, (bad.max()+40) if bad.size else min(n,400))
        ax[1].plot(np.arange(n)[w], T[w], '.-'); ax[1].set_title("zoom on violation region")
        ax[1].set_xlabel("index"); ax[1].set_ylabel("T [K]")
        fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
        print(f"  plot -> {png}")
print("\nDone.")
