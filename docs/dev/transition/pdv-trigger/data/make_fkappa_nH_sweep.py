#!/usr/bin/env python3
r"""Controlled f_kappa(n_H) calibration — harvest + fit the density sweep (HPC outputs).

Closes the loop on the controlled sweep `runs/params/sweep_fkappa_nH.param` (fixed mCloud+sfe, vary only
nCore x cooling_boost_kappa). After the HPC grid runs (28 combos -> outputs/sweep_fkappa_nH/<run>/), this:
  1. parses (nCore, f_kappa) from each run-name (e.g. `1e6_sfe010_n1e3_PL0_coolingBoostKappa8p0`),
  2. harvests the developed theta_blowout per run (REUSES the proven harvest() from
     make_kappa_blowout_calibration.py -- same definition, theta at first R2>rCloud),
  3. per density fits theta = a * f_kappa^p and solves f_kappa_fire (theta -> 0.95 cooling_balance trigger),
  4. fits f_kappa_fire(nCore) as a power law and writes the calibration CSV + figure.

This REPLACES the conflated 3-anchor estimate (compact/mid/diffuse vary mCloud+sfe+nCore together) with a
clean single-variable f_kappa(n_H). Until the sweep runs, this prints a clear "no outputs yet" message.

REPRODUCE (after the HPC sweep -- see REPRODUCE.md / sweep_fkappa_nH.param):
    python run.py docs/dev/transition/pdv-trigger/runs/params/sweep_fkappa_nH.param --emit-jobs jobs/
    sbatch jobs/submit_sweep.sbatch                      # -> outputs/sweep_fkappa_nH/<run>/
    python docs/dev/transition/pdv-trigger/data/make_fkappa_nH_sweep.py
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
_OUT = os.path.join(_REPO, "outputs", "sweep_fkappa_nH")
_TRIGGER = 0.95

# reuse the PROVEN theta_blowout harvest from the 3-anchor calibration harness
sys.path.insert(0, _HERE)
from make_kappa_blowout_calibration import harvest  # noqa: E402

# run-name -> (nCore, f_kappa). emit-jobs names like 1e6_sfe010_n1e3_PL0_coolingBoostKappa8p0
_RE_NCORE = re.compile(r"_n(\d+e\d+)_")
_RE_FK = re.compile(r"coolingBoostKappa(\d+)p(\d+)")


def parse_run_name(name):
    """(nCore, f_kappa) from a sweep run-dir name, or (None, None) if it doesn't match."""
    mn, mk = _RE_NCORE.search(name), _RE_FK.search(name)
    if not (mn and mk):
        return None, None
    nCore = float(mn.group(1))
    fk = float(f"{mk.group(1)}.{mk.group(2)}")
    return nCore, fk


def _selftest():
    cases = {
        "1e6_sfe010_n1e3_PL0_coolingBoostKappa8p0": (1e3, 8.0),
        "1e6_sfe010_n1e2_PL0_coolingBoostKappa1p0": (1e2, 1.0),
        "1e6_sfe010_n1e5_PL0_coolingBoostKappa64p0": (1e5, 64.0),
        "garbage_name": (None, None),
    }
    for name, want in cases.items():
        got = parse_run_name(name)
        assert got == want, f"parse_run_name({name!r}) = {got}, want {want}"
    print("selftest OK: run-name parser handles the emit-jobs naming")


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
              "Run the HPC grid first (see REPRODUCE.md / runs/params/sweep_fkappa_nH.param):\n"
              "  python run.py docs/dev/transition/pdv-trigger/runs/params/sweep_fkappa_nH.param --emit-jobs jobs/\n"
              "  sbatch jobs/submit_sweep.sbatch")
        return

    # harvest theta_blowout for every run, grouped by nCore
    by_density = {}
    for name in sorted(os.listdir(_OUT)):
        nCore, fk = parse_run_name(name)
        if nCore is None:
            continue
        h = harvest(os.path.join(_OUT, name))
        if not h["ok"]:
            print(f"  (skip {name}: no usable implicit phase)")
            continue
        by_density.setdefault(nCore, []).append((fk, h["theta_blowout"], h["cooling_fired"]))
        print(f"  nCore={nCore:.0e} f_κ={fk:<4g} θ_blowout={h['theta_blowout']:.3f} "
              f"cooling_fired={h['cooling_fired']}")

    if not by_density:
        print("No parseable runs found in the sweep output dir.")
        return

    rows = []
    for nCore in sorted(by_density):
        pts = sorted(by_density[nCore])
        fks = [p[0] for p in pts]
        ths = [p[1] for p in pts]
        a, p_exp, f_fire = fit_fire(fks, ths)
        # prefer the MEASURED firing point (lowest f_kappa that fired) over the extrapolation
        fired_fks = [p[0] for p in pts if p[2]]
        f_fire_measured = min(fired_fks) if fired_fks else np.nan
        rows.append(dict(nCore=nCore, n_points=len(pts), theta_fk1=ths[0] if fks[0] == 1 else np.nan,
                         fit_a=a, fit_p=p_exp, f_kappa_fire_fit=f_fire,
                         f_kappa_fire_measured=f_fire_measured))
        print(f"nCore={nCore:.0e}: θ∝f_κ^{p_exp:.2f}  f_κ_fire(fit)={f_fire:.1f}  "
              f"f_κ_fire(measured)={f_fire_measured}")

    csv_path = os.path.join(_HERE, "fkappa_nH_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["nCore", "n_points", "theta_fk1", "fit_a", "fit_p",
                                           "f_kappa_fire_fit", "f_kappa_fire_measured"])
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
    nC = np.array([r["nCore"] for r in rows], float)
    f_fit = np.array([r["f_kappa_fire_fit"] for r in rows], float)
    f_meas = np.array([r["f_kappa_fire_measured"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.loglog(nC, f_fit, "o-", color="#1f77b4", lw=2, ms=7, label=r"$f_\kappa$ to fire (power-law fit)")
    m = np.isfinite(f_meas)
    if m.any():
        ax.loglog(nC[m], f_meas[m], "D", color="crimson", ms=9, label=r"$f_\kappa$ to fire (measured: lowest fired)")
    # fit f_kappa_fire(nCore) = A * nCore^q on the fit points
    good = np.isfinite(nC) & np.isfinite(f_fit) & (f_fit > 0)
    if good.sum() >= 2:
        q, lnA = np.polyfit(np.log(nC[good]), np.log(f_fit[good]), 1)
        xx = np.logspace(np.log10(nC[good].min()), np.log10(nC[good].max()), 50)
        ax.loglog(xx, np.exp(lnA) * xx ** q, "--", color="0.5", lw=1.3,
                  label=rf"$f_\kappa^{{\rm fire}} \propto n_{{\rm core}}^{{{q:.2f}}}$")
    ax.set_xlabel(r"$n_{\rm core}$  [cm$^{-3}$]  (mCloud $=10^6$, sfe $=0.1$ fixed)")
    ax.set_ylabel(r"$f_\kappa$ to reach $\theta=0.95$ (cooling fires)")
    ax.set_title(r"Controlled $f_\kappa(n_{\rm H})$ calibration (single-variable density sweep)", fontsize=12)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    png = os.path.join(_PDV, "fkappa_nH_sweep.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
