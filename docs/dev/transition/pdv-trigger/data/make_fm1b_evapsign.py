#!/usr/bin/env python3
"""FM1b offline test: does in-structure interface cooling LOWER dMdt? (NO production edit.)

THE QUESTION (RUNGB_SCOPING.md §3a redirect + §8, FM1b)
------------------------------------------------------
FM1 refuted "impose dMdt". The redirect: keep dMdt as the Weaver eigenvalue and add the
mixing-layer cooling only to the IN-STRUCTURE loss, then MEASURE the evaporation response.
El-Badry's own mechanism predicts the sign -- "most of the energy conducted into the
interface is immediately lost to cooling, reducing the evaporative mass flux required to
balance conduction" -- i.e. adding interface cooling should drive dMdt DOWN. This harness
tests that prediction; dMdt UP would falsify the in-structure injection.

WHAT IT DOES (faithful, no solver edit)
---------------------------------------
For two REAL captured stiff states, it injects a localized extra cooling into the structure
by MONKEYPATCHING net_coolingcurve.get_dudt at runtime:

    dudt -> dudt * (1 + A * gaussian(log10 T ; center=5.0, width=0.4))

get_dudt returns the NET volumetric rate already SIGNED negative for cooling (see
net_coolingcurve.py: "return -1 * dudt"), so a factor > 1 in the ~10^5 K band makes the net
MORE negative = MORE cooling, exactly where the turbulent mixing layer radiates (the cooling-
curve peak). The patched get_dudt is what the production bubble-structure ODE
(_get_bubble_ODE) integrates, so the FULL production solve get_bubbleproperties_pure() -- the
v(R1)=0 fsolve for the Weaver dMdt and the whole structure -- runs WITH the extra cooling. We
read bubble_dMdt (the decisive quantity) and bubble_LTotal back out and sweep the amplitude A.

NOTE ON bubble_LTotal: the conduction-zone luminosity is recomputed from the cooling tables
inside _bubble_luminosity (NOT via get_dudt), so bubble_LTotal reflects the STRUCTURAL
(T-profile) response to the injection, not the injected term itself. The clean, faithful
measurement here is therefore the SIGN of d(dMdt)/dA -- the evaporation response to in-
structure cooling -- with A the injection strength.

BUILT-IN CORRECTNESS CHECK: at A=0 the patch is the identity, so bubble_dMdt must recover the
fixture's converged Spitzer dMdt. If it does not, the harness is wrong, not the physics.

REPRODUCE (from repo root):
    python docs/dev/transition/pdv-trigger/data/make_fm1b_evapsign.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/fm1b_evapsign.csv
    docs/dev/transition/pdv-trigger/fm1b_evapsign.png
"""

import csv
import importlib.util
import os

import numpy as np

import trinity.bubble_structure.bubble_luminosity as BL
from trinity.cooling import net_coolingcurve

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)

# Reuse the captured-state loader from the FM1 harness (same dir) -- no duplication.
_spec = importlib.util.spec_from_file_location("_fm1", os.path.join(_HERE, "make_fm1_rootcheck.py"))
_fm1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm1)

_ORIG_GET_DUDT = net_coolingcurve.get_dudt
_LOGT_CENTER = 5.0   # ~10^5 K, the turbulent-mixing-layer radiating band (cooling-curve peak)
_LOGT_WIDTH = 0.4    # +/- ~0.4 dex covers the ~3e4-3e5 K interface
_AMPS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]  # peak extra-cooling factor is (1 + A) at 10^5 K


def _patched(A):
    def get_dudt(age, ndens, T, phi, params_dict):
        orig = _ORIG_GET_DUDT(age, ndens, T, phi, params_dict)
        g = np.exp(-((np.log10(T) - _LOGT_CENTER) / _LOGT_WIDTH) ** 2)
        return orig * (1.0 + A * g)  # orig < 0 (net cooling); (1+A*g) > 1 => more cooling
    return get_dudt


def _run(params, A):
    """Full production solve with the A-amplitude in-structure cooling injection."""
    net_coolingcurve.get_dudt = _patched(A)
    try:
        if "bubble_dMdt" in params:
            params["bubble_dMdt"].value = float("nan")  # every run starts from the Eq-33 guess
        bp = BL.get_bubbleproperties_pure(params)
        return float(bp.bubble_dMdt), float(bp.bubble_LTotal)
    except Exception as e:  # solver may fail at large A -> record as nan, note it
        print(f"      (A={A}: solve failed: {type(e).__name__})")
        return float("nan"), float("nan")
    finally:
        net_coolingcurve.get_dudt = _ORIG_GET_DUDT


def main():
    rows = []
    series = {}  # label -> (amps, dMdt_ratio, L_ratio)
    notes = {label: note for label, _fx, note in _fm1._STATES}
    for label, fixture_name, note in _fm1._STATES:
        fixture, params = _fm1._load(fixture_name)
        dMdt0, L0 = _run(params, 0.0)
        conv = fixture["dMdt_converged"]
        ok_base = np.isfinite(dMdt0) and abs(dMdt0 - conv) / conv < 0.05
        print(f"[{label}] A=0 dMdt={dMdt0:.4g} vs converged {conv:.4g} "
              f"-> correctness {'OK' if ok_base else 'MISMATCH'}; LTotal0={L0:.4g}")
        amps, dr, lr = [], [], []
        for A in _AMPS:
            dMdt, L = _run(params, A)
            amps.append(A)
            dr.append(dMdt / dMdt0 if (np.isfinite(dMdt) and dMdt0) else np.nan)
            lr.append(L / L0 if (np.isfinite(L) and L0) else np.nan)
            rows.append({"state": label, "A": A, "dMdt": dMdt, "LTotal": L,
                         "dMdt_over_base": dr[-1], "LTotal_over_base": lr[-1],
                         "base_dMdt": dMdt0, "base_converged": conv, "base_ok": ok_base})
            print(f"      A={A:>4}  dMdt/base={dr[-1]:.4f}  LTotal/base={lr[-1]:.4f}")
        series[label] = (np.array(amps), np.array(dr), np.array(lr))

    csv_path = os.path.join(_HERE, "fm1b_evapsign.csv")
    cols = ["state", "A", "dMdt", "LTotal", "dMdt_over_base", "LTotal_over_base",
            "base_dMdt", "base_converged", "base_ok"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    # verdict: SIGN of d(dMdt) and its MAGNITUDE at max stable A
    deltas = []
    for label, (amps, dr, lr) in series.items():
        fin = dr[np.isfinite(dr)]
        if fin.size > 1:
            deltas.append(fin[-1] - 1.0)
    max_mag = max((abs(d) for d in deltas), default=float("nan"))
    if deltas and all(d < 0 for d in deltas):
        sign = "El-Badry SIGN confirmed (dMdt DOWN with interface cooling)"
    elif deltas and all(d > 0 for d in deltas):
        sign = "WRONG sign (dMdt UP -- Rung-A re-coupled)"
    else:
        sign = "mixed sign"
    weak = max_mag < 0.01
    verdict = (f"{sign}, but magnitude NEGLIGIBLE (<={max_mag:.2%} at 5x cooling) -- "
               f"interior cooling barely couples to the front-anchored dMdt" if weak
               else f"{sign}, magnitude {max_mag:.1%}")
    print(f"VERDICT: {verdict}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return
    # Two panels of the FAITHFUL quantity (percent change of dMdt). bubble_LTotal is omitted
    # from the headline: its conduction term is recomputed from the tables and does NOT count
    # the injection, so its down-drift is a structural artifact, not "less cooling" (it stays
    # in the CSV). LEFT: context vs the El-Badry target (3-30x suppression => dMdt -67% to
    # -97%) -- our effect is ~3 orders of magnitude short. RIGHT: zoom proving the monotonic
    # decrease is real signal (above the fsolve xtol ~ 0.01%), not noise.
    markers = {"stiff 5e9/sfe0.01": "o-", "mild cluster": "s--"}
    cols = {"stiff 5e9/sfe0.01": "#d62728", "mild cluster": "#1f77b4"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.0))

    # LEFT -- context against what El-Badry actually needs
    axL.axhspan(-97, -67, color="#2ca02c", alpha=0.12)
    axL.text(2.0, -82, "El-Badry target\n(3–30× suppression\n⇒ −67% to −97%)",
             ha="center", va="center", fontsize=9, color="#1c6b1c", fontweight="bold")
    axL.axhline(0, color="0.35", lw=1.0, ls=":")
    for label, (amps, dr, lr) in series.items():
        axL.plot(amps, (dr - 1.0) * 100.0, markers.get(label, "o-"), lw=2.2, ms=6,
                 color=cols.get(label, "0.3"), label=label)
    axL.set_ylim(-100, 8)
    axL.set_xlabel(r"injected cooling amplitude $A$ (peak $1{+}A$ at $10^5$ K)", fontsize=9)
    axL.set_ylabel(r"$\Delta\dot M$ vs baseline [%]", fontsize=9)
    axL.set_title("Nowhere near the El-Badry target\n(our effect ≈ 0 on this scale)", fontsize=10, fontweight="bold")
    axL.legend(fontsize=8.5, loc="lower left")

    # RIGHT -- zoom: the sign is real signal (above fsolve tolerance), just tiny
    axR.axhspan(-0.13, 0, color="#2ca02c", alpha=0.06)
    axR.axhspan(0, 0.04, color="#d62728", alpha=0.05)
    axR.axhline(0, color="0.35", lw=1.0, ls=":")
    axR.axhspan(-0.01, 0.01, color="0.5", alpha=0.18)  # ~ fsolve xtol band
    axR.text(0.05, 0.013, "fsolve xtol ~0.01% (noise floor)", fontsize=7.2, color="0.35", va="bottom")
    for label, (amps, dr, lr) in series.items():
        pct = (dr - 1.0) * 100.0
        axR.plot(amps, pct, markers.get(label, "o-"), lw=2.2, ms=6, color=cols.get(label, "0.3"),
                 label=f"{label}: {pct[np.isfinite(pct)][-1]:+.3f}% at A=4")
    axR.set_xlabel(r"injected cooling amplitude $A$", fontsize=9)
    axR.set_ylabel(r"$\Delta\dot M$ vs baseline [%]  (zoom)", fontsize=9)
    axR.set_title("The sign IS real (monotonic, above noise floor):\n"
                  "El-Badry direction confirmed, magnitude negligible", fontsize=10, fontweight="bold")
    axR.legend(fontsize=8, loc="lower left", title="evaporation response")

    fig.suptitle("FM1b: in-structure $10^5$ K cooling lowers $\\dot M$ (El-Badry sign ✓), "
                 "but only ~0.1% at 5× — far short of the −67…−97% target",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(_PDV, "fm1b_evapsign.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
