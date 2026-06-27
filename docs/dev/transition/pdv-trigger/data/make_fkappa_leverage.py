#!/usr/bin/env python3
"""f_kappa calibration — first cut: cooling-enhancement LEVERAGE + VIABILITY envelope. NO production edit.

THE GOAL (PLAN.md ⭐ synthesis): enhanced, density-dependent cooling matched to obs/3D, delivered by the
κ_eff mechanism (`cooling_boost_kappa`, Rung A, already built/gated) with f_κ(properties) CALIBRATED so the
emergent θ = L_cool/L_mech tracks the target θ(n_H) (El-Badry λδv=κ_eff + Lancaster ≈0.9–0.99). TRINITY
resolves θ ≈ 0.25 (diffuse) → 0.70 (dense) at blowout, so the calibration must lift it ~1.3–3.6×.

WHAT THIS MEASURES (and why it's a FIRST CUT). The committed captured states are EARLY snapshots (θ ≈ 0.009
stiff / 0.001 mild — a young bubble, far below the blowout values), so this does NOT read the absolute blowout
θ. Instead it measures the two things a captured snapshot CAN give cheaply, using the real gated knob
`cooling_boost_kappa` (no monkeypatch — this is production behaviour with the param set):
  1. LEVERAGE: bubble_LTotal(f_κ) / bubble_LTotal(1) — how much κ_eff multiplies the resolved cooling.
  2. VIABILITY: up to what f_κ does the FULL solve `get_bubbleproperties_pure` stay healthy (fsolve converges,
     dMdt > 0, T monotonic > 0)? The Rung-A back-reaction (dMdt up, Eb down) grows with f_κ, so there is a
     ceiling — and where it sits is decision-relevant (it bounds how much in-structure enhancement is reachable).

f_κ=1 is byte-identical (recovers the converged dMdt — the built-in correctness check). The ABSOLUTE
blowout-θ calibration across a density grid needs FULL runs (the documented next step); this first cut tells us
whether κ_eff has the leverage to plausibly reach the target and whether the required f_κ is viable.

REPRODUCE (from repo root):
    python docs/dev/transition/pdv-trigger/data/make_fkappa_leverage.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/fkappa_leverage.csv
    docs/dev/transition/pdv-trigger/fkappa_leverage.png
"""

import csv
import importlib.util
import os

import numpy as np

import trinity.bubble_structure.bubble_luminosity as BL

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)

# Reuse the captured-state loader from the FM1 harness (same dir) — no duplication.
_spec = importlib.util.spec_from_file_location("_fm1", os.path.join(_HERE, "make_fm1_rootcheck.py"))
_fm1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm1)

_FKAPPA = [1, 2, 4, 8, 16, 32, 64]          # conduction-coefficient multiplier sweep
_TARGET_MULT = (1.3, 3.6)                    # multiplier needed to lift blowout θ 0.25–0.70 → ~0.9


def _run(params, fk):
    """Full production solve with cooling_boost_kappa = fk. Returns (LTotal, dMdt, healthy)."""
    params["cooling_boost_kappa"].value = float(fk)
    if "bubble_dMdt" in params:
        params["bubble_dMdt"].value = float("nan")  # clean fsolve from the Eq-33 seed each time
    try:
        bp = BL.get_bubbleproperties_pure(params)
        T = np.asarray(bp.bubble_T_arr)
        mono = bool(np.all(np.diff(T) >= 0) or np.all(np.diff(T) <= 0)) and bool(np.all(T > 0))
        healthy = (np.isfinite(bp.bubble_dMdt) and bp.bubble_dMdt > 0
                   and np.isfinite(bp.bubble_LTotal) and bp.bubble_LTotal > 0 and mono)
        return float(bp.bubble_LTotal), float(bp.bubble_dMdt), bool(healthy)
    except Exception as e:
        print(f"      (f_κ={fk}: solve failed: {type(e).__name__})")
        return float("nan"), float("nan"), False


def main():
    rows = []
    series = {}  # label -> (fk[], mult[], dMdt_ratio[], healthy[])
    notes = {label: note for label, _fx, note in _fm1._STATES}
    for label, fixture_name, _note in _fm1._STATES:
        fixture, params = _fm1._load(fixture_name)
        Lmech = params["Lmech_total"].value
        L0, dM0, ok0 = _run(params, 1.0)
        theta0 = L0 / Lmech if np.isfinite(L0) else float("nan")
        print(f"[{label}] f_κ=1 baseline: LTotal={L0:.4g} dMdt={dM0:.4g} "
              f"θ_snapshot={theta0:.4g} healthy={ok0}")
        fk_l, mult_l, dr_l, ok_l = [], [], [], []
        for fk in _FKAPPA:
            L, dM, ok = _run(params, fk)
            mult = L / L0 if (np.isfinite(L) and L0) else float("nan")
            dr = dM / dM0 if (np.isfinite(dM) and dM0) else float("nan")
            fk_l.append(fk); mult_l.append(mult); dr_l.append(dr); ok_l.append(ok)
            rows.append({"state": label, "f_kappa": fk, "LTotal": L, "dMdt": dM,
                         "LTotal_mult": mult, "dMdt_ratio": dr,
                         "theta_snapshot": (L / Lmech if np.isfinite(L) else float("nan")),
                         "healthy": ok, "base_LTotal": L0, "base_dMdt": dM0})
            print(f"      f_κ={fk:>3}  Lcool×{mult:.3f}  dMdt×{dr:.3f}  healthy={ok}")
        series[label] = (np.array(fk_l, float), np.array(mult_l), np.array(dr_l), np.array(ok_l, bool))

    csv_path = os.path.join(_HERE, "fkappa_leverage.csv")
    cols = ["state", "f_kappa", "LTotal", "dMdt", "LTotal_mult", "dMdt_ratio",
            "theta_snapshot", "healthy", "base_LTotal", "base_dMdt"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    # viability ceiling per state + the f_κ that reaches the target multiplier band
    for label, (fk, mult, dr, ok) in series.items():
        ceil = fk[ok].max() if ok.any() else float("nan")
        reach = fk[(mult >= _TARGET_MULT[0]) & ok]
        reach_lo = reach.min() if reach.size else float("nan")
        print(f"VIABILITY [{label}]: healthy up to f_κ={ceil:g}; "
              f"f_κ for Lcool×{_TARGET_MULT[0]} (target floor) = {reach_lo:g}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.0))
    markers = {"stiff 5e9/sfe0.01": "o-", "mild cluster": "s--"}
    cols_c = {"stiff 5e9/sfe0.01": "#d62728", "mild cluster": "#1f77b4"}

    # LEFT — the cooling-enhancement leverage, with the target multiplier band
    axL.axhspan(_TARGET_MULT[0], _TARGET_MULT[1], color="#2ca02c", alpha=0.10)
    axL.text(_FKAPPA[1], (_TARGET_MULT[0] + _TARGET_MULT[1]) / 2,
             "target enhancement\n(lift blowout θ 0.25–0.70\n→ ~0.9: ×1.3–3.6)",
             fontsize=8.5, color="#1c6b1c", va="center")
    axL.axhline(1.0, color="0.4", lw=1.0, ls=":")
    for label, (fk, mult, dr, ok) in series.items():
        axL.plot(fk, mult, markers.get(label, "o-"), lw=2.0, ms=6, color=cols_c.get(label, "0.3"), label=label)
        if (~ok).any():  # mark first unhealthy f_κ
            bad = fk[~ok]
            axL.scatter(bad, [np.nan if not np.isfinite(m) else m for m in mult[~ok]],
                        marker="x", s=70, color="black", zorder=5)
    axL.set_xscale("log", base=2)
    axL.set_xlabel(r"$f_\kappa$  (conduction-coefficient multiplier)", fontsize=9.5)
    axL.set_ylabel(r"$L_{\rm cool}(f_\kappa)\,/\,L_{\rm cool}(1)$  (leverage)", fontsize=9.5)
    axL.set_title("Cooling-enhancement leverage of $\\kappa_{\\rm eff}$\n(× on the black-X = solve no longer healthy)",
                  fontsize=10, fontweight="bold")
    axL.legend(fontsize=8.5, loc="upper left")

    # RIGHT — the Rung-A back-reaction that sets the viability ceiling
    axR.axhline(1.0, color="0.4", lw=1.0, ls=":")
    for label, (fk, mult, dr, ok) in series.items():
        axR.plot(fk, dr, markers.get(label, "o-"), lw=2.0, ms=6, color=cols_c.get(label, "0.3"), label=label)
    axR.set_xscale("log", base=2)
    axR.set_xlabel(r"$f_\kappa$", fontsize=9.5)
    axR.set_ylabel(r"$\dot M(f_\kappa)\,/\,\dot M(1)$  (evaporation back-reaction)", fontsize=9.5)
    axR.set_title("The tolerated side effect grows with $f_\\kappa$\n(sets the viability ceiling)",
                  fontsize=10, fontweight="bold")
    axR.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("f_κ calibration — first cut: how far can κ_eff push the cooling, and where does it break? "
                 "(captured states; leverage + viability)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(_PDV, "fkappa_leverage.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
