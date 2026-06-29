#!/usr/bin/env python3
r"""Controlled f_kappa(n_H) calibration — fit + plot from the reduced summary.csv (laptop step).

Step 3 of the reduce-then-plot pipeline (REPRODUCE.md Block C):
  1. run the 819-combo sweep on Helix          -> runs/run_fkappa.sbatch        (sync.sh submit)
  2. reduce the jsonl to one small summary.csv  -> data/reduce_fkappa_sweep.py   (ON HPC; sync.sh reduce)
  3. THIS script (the LAPTOP step): read ONLY summary.csv, group by (mCloud, sfe, nCore) cell, fit
     theta = a * f_kappa^p, solve f_kappa_fire (theta -> 0.95 cooling_balance trigger), fit
     f_kappa_fire(nCore) as a power law, and draw the de-conflation figure (do the mCloud/sfe series
     collapse onto one n_H curve, or spread?).

Reads only summary.csv (the multi-GB jsonl stays on the cluster), so you iterate on the figure locally
with no numpy/trinity-on-cluster and no re-reading the sweep. This REPLACES the conflated 3-anchor
estimate (compact/mid/diffuse varied mCloud+sfe+nCore together) with a clean single-variable f_kappa(n_H).

Consumes the summary.csv columns written by reduce_fkappa_sweep.py:
    mCloud, sfe, nCore, cooling_boost_kappa, theta_blowout, cooling_fired

REPRODUCE (after the sweep + reduce -- see REPRODUCE.md Block C):
    ./docs/dev/transition/pdv-trigger/runs/sync.sh down      # pulls summary.csv -> data/summary.csv
    python docs/dev/transition/pdv-trigger/data/make_fkappa_nH_sweep.py            # reads data/summary.csv
    python docs/dev/transition/pdv-trigger/data/make_fkappa_nH_sweep.py PATH/summary.csv   # or an explicit path
Self-test (no data needed):  python .../make_fkappa_nH_sweep.py --selftest
Deliverables:
    docs/dev/transition/pdv-trigger/data/fkappa_nH_sweep.csv   (one row per mCloud,sfe,nCore cell + fit)
    docs/dev/transition/pdv-trigger/fkappa_nH_sweep.png
"""

import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
# Default input is the reduced table that `sync.sh down` drops next to this script. Override by
# passing a summary.csv path as argv[1] (e.g. the cluster's outputs/sweep_fkappa_nH/summary.csv).
_DEFAULT_SUMMARY = os.path.join(_HERE, "summary.csv")
_TRIGGER = 0.95


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fit_fire(fks, thetas):
    """Fit theta = a*f_kappa^p (log-log least squares) on >=2 finite points; return (a, p, f_kappa_fire)."""
    pts = [(f, t) for f, t in zip(fks, thetas)
           if math.isfinite(f) and math.isfinite(t) and f > 0 and t > 0]
    if len(pts) < 2:
        return float("nan"), float("nan"), float("nan")
    xs = [math.log(f) for f, _ in pts]
    ys = [math.log(t) for _, t in pts]
    n = len(pts)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    p = sxy / sxx
    a = math.exp(my - p * mx)
    f_fire = (_TRIGGER / a) ** (1.0 / p) if p > 0 and a > 0 else float("nan")
    return a, p, f_fire


def _selftest():
    """fit_fire recovers a known power law theta = a*f^p and its f_fire (theta -> _TRIGGER)."""
    a, p = 0.2, 0.5
    fks = [1.0, 2.0, 4.0, 8.0, 16.0]
    thetas = [a * f ** p for f in fks]
    got_a, got_p, got_fire = fit_fire(fks, thetas)
    want_fire = (_TRIGGER / a) ** (1.0 / p)
    assert abs(got_a - a) < 1e-9 and abs(got_p - p) < 1e-9, f"fit ({got_a},{got_p}) != ({a},{p})"
    assert abs(got_fire - want_fire) <= 1e-9 * want_fire, f"f_fire {got_fire} != {want_fire}"
    print(f"selftest OK: fit_fire recovers theta={a}*f_k^{p} -> f_k_fire={got_fire:.3f} (== {want_fire:.3f})")


def main():
    if "--selftest" in sys.argv:
        _selftest()
        return
    summary = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_SUMMARY
    if not os.path.exists(summary):
        print(f"No summary.csv at {summary}.\n"
              "Run the sweep, REDUCE it on HPC, then pull the CSV (see REPRODUCE.md Block C):\n"
              "  ./docs/dev/transition/pdv-trigger/runs/sync.sh submit    # run the 819-combo array\n"
              "  ./docs/dev/transition/pdv-trigger/runs/sync.sh reduce    # jsonl -> summary.csv (on HPC)\n"
              "  ./docs/dev/transition/pdv-trigger/runs/sync.sh down      # summary.csv -> data/\n"
              "Or pass a summary.csv path explicitly as the first argument.")
        return

    # group theta_blowout by (mCloud, sfe, nCore) cell from the reduced summary.csv
    by_cell = {}
    for r in csv.DictReader(open(summary)):
        mCloud, sfe, nCore = _f(r.get("mCloud")), _f(r.get("sfe")), _f(r.get("nCore"))
        fk, theta = _f(r.get("cooling_boost_kappa")), _f(r.get("theta_blowout"))
        fired = str(r.get("cooling_fired", "")).strip().lower() in ("true", "1")
        if not (math.isfinite(mCloud) and math.isfinite(nCore) and math.isfinite(fk)):
            continue
        by_cell.setdefault((mCloud, sfe, nCore), []).append((fk, theta, fired))

    if not by_cell:
        print(f"summary.csv at {summary} had no parseable rows.")
        return

    rows = []
    for (mCloud, sfe, nCore) in sorted(by_cell):
        pts = sorted(by_cell[(mCloud, sfe, nCore)])
        fks = [p[0] for p in pts]
        ths = [p[1] for p in pts]
        a, p_exp, f_fire = fit_fire(fks, ths)
        fired_fks = [p[0] for p in pts if p[2]]               # MEASURED: lowest f_kappa that fired
        f_fire_meas = min(fired_fks) if fired_fks else float("nan")
        rows.append(dict(mCloud=mCloud, sfe=sfe, nCore=nCore, n_points=len(pts),
                         theta_fk1=(ths[0] if fks and fks[0] == 1 else float("nan")),
                         fit_a=a, fit_p=p_exp, f_kappa_fire_fit=f_fire,
                         f_kappa_fire_measured=f_fire_meas))
        print(f"mCloud={mCloud:.0e} sfe={sfe:.2f} nCore={nCore:.0e}: theta~f_k^{p_exp:.2f}  "
              f"f_k_fire(fit)={f_fire:.1f}  f_k_fire(meas)={f_fire_meas}")

    csv_path = os.path.join(_HERE, "fkappa_nH_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mCloud", "sfe", "nCore", "n_points", "theta_fk1",
                                           "fit_a", "fit_p", "f_kappa_fire_fit", "f_kappa_fire_measured"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        sys.path.insert(0, _HERE)
        from _trinity_style import use_trinity_style
        use_trinity_style()
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    cells = sorted({(r["mCloud"], r["sfe"]) for r in rows})
    cmap = plt.get_cmap("viridis")
    for i, (mC, sf) in enumerate(cells):
        sub = sorted([r for r in rows if r["mCloud"] == mC and r["sfe"] == sf], key=lambda r: r["nCore"])
        x = np.array([r["nCore"] for r in sub], float)
        y = np.array([r["f_kappa_fire_measured"] for r in sub], float)
        yfit = np.array([r["f_kappa_fire_fit"] for r in sub], float)
        y = np.where(np.isfinite(y), y, yfit)
        col = cmap(i / max(1, len(cells) - 1))
        ax.loglog(x, y, "o-", color=col, lw=1.5, ms=6, label=rf"$M_{{\rm cl}}{{=}}{mC:.0e}$, sfe$={sf:g}$")
    allx = np.array([r["nCore"] for r in rows], float)
    ally = np.array([r["f_kappa_fire_measured"] if np.isfinite(r["f_kappa_fire_measured"])
                     else r["f_kappa_fire_fit"] for r in rows], float)
    good = np.isfinite(allx) & np.isfinite(ally) & (ally > 0)
    if good.sum() >= 2:
        q, lnA = np.polyfit(np.log(allx[good]), np.log(ally[good]), 1)
        xx = np.logspace(np.log10(allx[good].min()), np.log10(allx[good].max()), 50)
        ax.loglog(xx, np.exp(lnA) * xx ** q, "--", color="k", lw=1.6,
                  label=rf"all-cell fit: $f_\kappa^{{\rm fire}}\propto n_{{\rm core}}^{{{q:.2f}}}$")
    ax.set_xlabel(r"$n_{\rm core}$  [cm$^{-3}$]")
    ax.set_ylabel(r"$f_\kappa$ to reach $\theta=0.95$ (cooling fires)")
    ax.set_title(r"$f_\kappa(n_{\rm H})$ calibration — do the $M_{\rm cl}$/sfe series collapse onto one curve?",
                 fontsize=11.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    png = os.path.join(_PDV, "fkappa_nH_sweep.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
