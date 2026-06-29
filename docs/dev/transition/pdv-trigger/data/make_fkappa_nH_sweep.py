#!/usr/bin/env python3
r"""Controlled f_kappa(n_H) calibration — harvest + fit the density sweep (HPC outputs).

Closes the loop on the controlled sweep `runs/params/sweep_fkappa_nH.param` (sweeps nCore x
cooling_boost_kappa x mCloud x sfe). After the HPC grid runs (819 combos -> <out>/sweep_fkappa_nH/<run>/), this:
  1. parses (nCore, f_kappa) from each run-name (e.g. `1e6_sfe010_n1e3_PL0_coolingBoostKappa8p0`),
  2. harvests the developed theta_blowout per run (REUSES the proven harvest() from
     make_kappa_blowout_calibration.py -- same definition, theta at first R2>rCloud),
  3. per density fits theta = a * f_kappa^p and solves f_kappa_fire (theta -> 0.95 cooling_balance trigger),
  4. fits f_kappa_fire(nCore) as a power law and writes the calibration CSV + figure.

This REPLACES the conflated 3-anchor estimate (compact/mid/diffuse vary mCloud+sfe+nCore together) with a
clean single-variable f_kappa(n_H). Until the sweep runs, this prints a clear "no outputs yet" message.

REPRODUCE (after the HPC sweep -- see REPRODUCE.md Block C / sweep_fkappa_nH.param):
    ./docs/dev/transition/pdv-trigger/runs/sync.sh submit    # emit to /gpfs + sbatch the 819-task array
    ./docs/dev/transition/pdv-trigger/runs/sync.sh harvest   # runs THIS script on Helix against /gpfs
On Helix the run outputs live on /gpfs, not under the repo, so point this script at them with
FKAPPA_SWEEP_OUT (sync.sh harvest sets it for you):
    FKAPPA_SWEEP_OUT=/gpfs/bwfor/work/ws/hd_cq295-trinity/outputs/sweep_fkappa_nH \
        python docs/dev/transition/pdv-trigger/data/make_fkappa_nH_sweep.py
Default (unset) reads <repo>/outputs/sweep_fkappa_nH -- for outputs already pulled to the laptop.
Self-test only (no sweep data needed):  python .../make_fkappa_nH_sweep.py --selftest
Deliverables:
    docs/dev/transition/pdv-trigger/data/fkappa_nH_sweep.csv
    docs/dev/transition/pdv-trigger/fkappa_nH_sweep.png
"""

import csv
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_REPO = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 5)))
# On Helix the sweep outputs land on /gpfs (the repo's /home is read-only at runtime), so allow an
# override; default to <repo>/outputs/sweep_fkappa_nH for outputs run/pulled locally.
_OUT = os.environ.get("FKAPPA_SWEEP_OUT", os.path.join(_REPO, "outputs", "sweep_fkappa_nH"))
_TRIGGER = 0.95

# reuse the PROVEN theta_blowout harvest from the 3-anchor calibration harness
sys.path.insert(0, _HERE)
from make_kappa_blowout_calibration import harvest  # noqa: E402

# run-name -> (mCloud, sfe, nCore, f_kappa). emit-jobs names like
#   1e7_sfe030_n3e3_PL0_coolingBoostKappa12p0  (mCloud=1e7, sfe=30/100=0.30, nCore=3e3, f_kappa=12.0)
_RE_MCLOUD = re.compile(r"^(\d+e\d+)_")
_RE_SFE = re.compile(r"_sfe(\d+)_")
_RE_NCORE = re.compile(r"_n(\d+e\d+)_")
_RE_FK = re.compile(r"coolingBoostKappa(\d+)p(\d+)")


def parse_run_name(name):
    """(mCloud, sfe, nCore, f_kappa) from a sweep run-dir name, or all-None if it doesn't match."""
    mm, ms = _RE_MCLOUD.search(name), _RE_SFE.search(name)
    mn, mk = _RE_NCORE.search(name), _RE_FK.search(name)
    if not (mm and ms and mn and mk):
        return None, None, None, None
    mCloud = float(mm.group(1))
    sfe = int(ms.group(1)) / 100.0          # sfe encoded as round(sfe*100), 3-digit (003->0.03, 030->0.30)
    nCore = float(mn.group(1))
    fk = float(f"{mk.group(1)}.{mk.group(2)}")
    return mCloud, sfe, nCore, fk


def _selftest():
    cases = {
        "1e7_sfe030_n3e3_PL0_coolingBoostKappa12p0": (1e7, 0.30, 3e3, 12.0),
        "1e5_sfe003_n1e2_PL0_coolingBoostKappa1p0": (1e5, 0.03, 1e2, 1.0),
        "1e6_sfe010_n1e5_PL0_coolingBoostKappa64p0": (1e6, 0.10, 1e5, 64.0),
        "1e5_sfe003_n3e2_PL0_coolingBoostKappa1p5": (1e5, 0.03, 3e2, 1.5),
        "garbage_name": (None, None, None, None),
    }
    for name, want in cases.items():
        got = parse_run_name(name)
        assert got == want, f"parse_run_name({name!r}) = {got}, want {want}"
    print("selftest OK: run-name parser handles the emit-jobs naming (mCloud, sfe, nCore, f_kappa)")


def fit_fire(fks, thetas):
    """Fit theta = a*f_kappa^p (log-log LSQ) on >=2 finite points; return (a, p, f_kappa_fire)."""
    fks, thetas = np.asarray(fks, float), np.asarray(thetas, float)
    good = np.isfinite(fks) & np.isfinite(thetas) & (fks > 0) & (thetas > 0)
    if good.sum() < 2:
        return np.nan, np.nan, np.nan
    p, lna = np.polyfit(np.log(fks[good]), np.log(thetas[good]), 1)
    a = np.exp(lna)
    f_fire = (_TRIGGER / a) ** (1.0 / p) if p > 0 else np.nan
    return a, p, f_fire


def main():
    if "--selftest" in sys.argv:
        _selftest()
        return
    _selftest()  # always sanity-check the parser before trusting the harvest

    if not os.path.isdir(_OUT):
        print(f"No sweep outputs at {_OUT}.\n"
              "Run the HPC grid first (see REPRODUCE.md Block C / runs/run_fkappa.sbatch):\n"
              "  ./docs/dev/transition/pdv-trigger/runs/sync.sh submit\n"
              "  ./docs/dev/transition/pdv-trigger/runs/sync.sh harvest   # or set FKAPPA_SWEEP_OUT and re-run this")
        return

    # harvest theta_blowout for every run, grouped by (mCloud, sfe, nCore) cell
    by_cell = {}
    for name in sorted(os.listdir(_OUT)):
        mCloud, sfe, nCore, fk = parse_run_name(name)
        if nCore is None:
            continue
        h = harvest(os.path.join(_OUT, name))
        if not h["ok"]:
            print(f"  (skip {name}: no usable implicit phase)")
            continue
        by_cell.setdefault((mCloud, sfe, nCore), []).append((fk, h["theta_blowout"], h["cooling_fired"]))

    if not by_cell:
        print("No parseable runs found in the sweep output dir.")
        return

    rows = []
    for (mCloud, sfe, nCore) in sorted(by_cell):
        pts = sorted(by_cell[(mCloud, sfe, nCore)])
        fks = [p[0] for p in pts]
        ths = [p[1] for p in pts]
        a, p_exp, f_fire = fit_fire(fks, ths)
        fired_fks = [p[0] for p in pts if p[2]]          # MEASURED firing: lowest f_kappa that fired
        f_fire_measured = min(fired_fks) if fired_fks else np.nan
        rows.append(dict(mCloud=mCloud, sfe=sfe, nCore=nCore, n_points=len(pts),
                         theta_fk1=(ths[0] if fks and fks[0] == 1 else np.nan),
                         fit_a=a, fit_p=p_exp, f_kappa_fire_fit=f_fire,
                         f_kappa_fire_measured=f_fire_measured))
        print(f"mCloud={mCloud:.0e} sfe={sfe:.2f} nCore={nCore:.0e}: θ∝f_κ^{p_exp:.2f}  "
              f"f_κ_fire(fit)={f_fire:.1f}  f_κ_fire(meas)={f_fire_measured}")

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
        from _trinity_style import use_trinity_style
        use_trinity_style()
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return
    # De-conflation figure: f_kappa_fire vs nCore, one series per (mCloud, sfe). If they COLLAPSE onto
    # one curve, f_kappa is a clean function of n_H alone; if they spread, it also depends on mCloud/sfe.
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    cells = sorted({(r["mCloud"], r["sfe"]) for r in rows})
    cmap = plt.get_cmap("viridis")
    for i, (mC, sf) in enumerate(cells):
        sub = sorted([r for r in rows if r["mCloud"] == mC and r["sfe"] == sf], key=lambda r: r["nCore"])
        x = np.array([r["nCore"] for r in sub], float)
        y = np.array([r["f_kappa_fire_measured"] for r in sub], float)
        yfit = np.array([r["f_kappa_fire_fit"] for r in sub], float)
        y = np.where(np.isfinite(y), y, yfit)            # fall back to the fit where nothing fired in-grid
        col = cmap(i / max(1, len(cells) - 1))
        ax.loglog(x, y, "o-", color=col, lw=1.5, ms=6,
                  label=rf"$M_{{\rm cl}}{{=}}{mC:.0e}$, sfe$={sf:g}$")
    # overall power-law fit across ALL cells (the leading n_H trend)
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
    ax.set_title(r"$f_\kappa(n_{\rm H})$ calibration — do the $M_{\rm cl}$/sfe series collapse onto one $n_{\rm H}$ curve?",
                 fontsize=11.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    png = os.path.join(_PDV, "fkappa_nH_sweep.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
