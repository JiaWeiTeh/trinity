#!/usr/bin/env python3
"""The physical-cap reframing: don't FORCE cooling to fire — bound f_κ physically and let clouds
that can't reach θ=0.95 stay energy-driven ("not meant to be"). Predicts a CRITICAL COLUMN.

Motivation (maintainer, 2026-06-29). Searching f_κ up to 64 to make every cloud fire assumes every
cloud MUST become momentum-driven — which is not physical. Two facts make the case:

  1. f_κ=64 is not a physical "enhanced conduction". A large CONSTANT f_κ multiplies the Spitzer
     T^(5/2) prefactor EVERYWHERE, over-conducting in the hot interior where Spitzer should rule. The
     physically-motivated enhancement is El-Badry's TEMPERATURE-INDEPENDENT mixing term (Eq 21,
     verified): κ_mix = (λδv)·ρ k_B/(μ m_p) = (λδv)·n·k_B, used as κ=max(κ_mix, κ_Spitzer).

  2. The PHYSICAL f_κ(n) RISES with density, OPPOSITE to the empirical fire-threshold.
       f_κ_physical = κ_mix/κ_Spitzer ∝ n / T^(5/2)  → ∝ n^(+1) at fixed interface T.
       f_κ_fire (measured, to reach θ=0.95)          → ∝ n^(−0.60).
     So using the empirical −0.6 as a PRESCRIPTION gives diffuse clouds the MOST boost — that IS the
     "forcing" we want to avoid. The physical (rising) prescription gives diffuse the LEAST → dense
     transition, diffuse stay energy-driven. That is the honest reading.

THE EXPERIMENT (pure re-analysis of the committed `data/summary.csv` — NO new sims). Cap the
enhancement at a physical f_max. A cloud transitions iff its measured f_κ_fire ≤ f_max; otherwise it
blows out energy-driven. The transition boundary is a ~constant COLUMN N_H = n_core·rCloud (the cliff,
§9): clouds above N_crit(f_max) are momentum-driven, below are energy-driven. A physically-plausible
f_max ≈ 2–8 gives N_crit ≈ 1–4×10²³ cm⁻²; 6/63 cells never fire even at f_κ=64 (energy-driven under
ANY cap). This is a FALSIFIABLE prediction (the critical column) to compare against Lancaster/PHANGS —
NOT a knob tuned to force a transition.

Caveat kept honest: Lancaster's 3D finds cooling "generic over >3 dex in density", i.e. even diffuse
clouds may cool catastrophically via turbulent mixing a 1D model can't see. So a non-transitioning 1D
cloud is EITHER genuinely energy-driven OR 1D-under-cooled (the El-Badry κ_mix it is missing). The
critical-column prediction is the dividing line; which side is physically right is settled against obs.

REPRODUCE (from repo root; reads only the committed summary.csv, no sims):
    python docs/dev/transition/pdv-trigger/data/make_fkappa_physical_cap.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/fkappa_physical_cap.csv
    docs/dev/transition/pdv-trigger/fkappa_physical_cap.png
"""

import csv
import math
import os
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_SUMMARY = os.path.join(_HERE, "summary.csv")
_PC = 3.086e18
_GRID = [1.0, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
_CAPS = [2, 4, 8, 16, 32, 64]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    rows = list(csv.DictReader(open(_SUMMARY)))
    cell = defaultdict(dict)
    rcl = {}
    for r in rows:
        k = (_f(r["mCloud"]), _f(r["sfe"]), _f(r["nCore"]))
        cell[k][_f(r["cooling_boost_kappa"])] = str(r["cooling_fired"]).strip().lower() == "true"
        rcl[k] = _f(r["rCloud"])

    # measured f_κ to fire per cell (inf = never within the grid)
    fire, col = {}, {}
    for k, by in cell.items():
        fk = [f for f in _GRID if by.get(f)]
        fire[k] = min(fk) if fk else float("inf")
        col[k] = k[2] * rcl[k] * _PC
    never = sum(1 for v in fire.values() if math.isinf(v))

    # transition map vs physical cap: count + critical column (geometric boundary)
    print(f"{'f_max':>6} {'#momentum':>10} {'#energy':>8}  {'N_crit [cm^-2]':>14}")
    caprows = []
    for fmax in _CAPS:
        mom = [k for k, v in fire.items() if v <= fmax]
        eng = [k for k, v in fire.items() if v > fmax]
        cm = [col[k] for k in mom]
        ce = [col[k] for k in eng]
        ncrit = math.sqrt(min(cm) * max(ce)) if cm and ce else float("nan")
        caprows.append({"f_max": fmax, "n_momentum": len(mom), "n_energy": len(eng),
                        "N_crit_cm2": f"{ncrit:.3e}"})
        print(f"{fmax:6.0f} {len(mom):10d} {len(eng):8d}  {ncrit:.2e}")
    print(f"never fire @f_κ≤64: {never}/63 (energy-driven under ANY physical cap)")

    out = os.path.join(_HERE, "fkappa_physical_cap.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["f_max", "n_momentum", "n_energy", "N_crit_cm2"])
        w.writeheader()
        w.writerows(caprows)
        fh.write(f"# never-fire (energy-driven under any cap): {never}/63 cells (all low-n/high-sfe)\n")
        fh.write("# physical f_kappa ~ n^+1 (El-Badry kappa_mix/kappa_Spitzer); fire-threshold ~ n^-0.6 (OPPOSITE sign)\n")
        fh.write("# a cloud is momentum-driven iff f_kappa_fire <= f_max; else energy-driven ('not meant to be')\n")
    print(f"wrote {out}")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from _trinity_style import use_trinity_style
            use_trinity_style()
        except Exception:
            pass
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.3))

    # LEFT: f_κ_fire vs column, with physical caps -> the column boundary
    finite = [(col[k], fire[k]) for k in fire if math.isfinite(fire[k])]
    nf = [col[k] for k in fire if math.isinf(fire[k])]
    xs = np.array([c for c, _ in finite])
    ys = np.array([f for _, f in finite])
    axL.loglog(xs, ys, "o", color="#1f77b4", ms=5, alpha=0.6, label="cells (measured f_κ to fire)")
    axL.plot(nf, [70] * len(nf), "x", color="r", ms=9, mew=2, label=f"never fire @f_κ≤64 ({len(nf)})")
    for fmax, c in zip([2, 4, 8], ["#2ca02c", "#ff7f0e", "#9467bd"]):
        axL.axhline(fmax, ls="--", color=c, lw=1.3)
        axL.text(xs.min() * 1.1, fmax * 1.07, f"physical cap f_max={fmax}", fontsize=8, color=c)
    axL.set_xlabel(r"column $N_H = n_{\rm core}\,r_{\rm cloud}$ [cm$^{-2}$]")
    axL.set_ylabel(r"$f_\kappa$ needed to fire ($\theta=0.95$)")
    axL.set_title("Physical cap → a CRITICAL COLUMN\n(below a cap line ⇒ momentum; above ⇒ energy-driven)",
                  fontsize=10.5, fontweight="bold")
    axL.legend(fontsize=8.5, loc="upper right")
    axL.grid(True, which="both", alpha=0.2)

    # RIGHT: the momentum/energy split vs the assumed physical cap
    fm = [r["f_max"] for r in caprows]
    nm = [r["n_momentum"] for r in caprows]
    axR.plot(fm, nm, "o-", color="#2f9e44", lw=2, ms=7, label="momentum-driven (fires)")
    axR.plot(fm, [63 - n for n in nm], "o-", color="#d1495b", lw=2, ms=7, label="energy-driven (blows out)")
    axR.axhline(never, ls=":", color="0.4", lw=1)
    axR.text(fm[-1], never + 1, f"{never} never fire (any cap)", fontsize=8, ha="right", color="0.35")
    axR.set_xscale("log", base=2)
    axR.set_xlabel(r"assumed physical max enhancement $f_{\rm max}$")
    axR.set_ylabel("number of clouds (of 63)")
    axR.set_title("Don't force it: a physical f_max sets the split\n(more boost ⇒ more transition, but it's a CHOICE)",
                  fontsize=10.5, fontweight="bold")
    axR.legend(fontsize=9, loc="center right")
    axR.grid(True, which="both", alpha=0.2)

    fig.suptitle("Physically-bounded f_κ: clouds that can't reach θ=0.95 stay energy-driven — a falsifiable critical column",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "fkappa_physical_cap.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
