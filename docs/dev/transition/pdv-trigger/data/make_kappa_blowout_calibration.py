#!/usr/bin/env python3
"""f_kappa calibration on full runs — the DEVELOPED / CUMULATIVE cooling efficiency, across density.

NOTE ON THE EPOCH (why not "at blowout"): El-Badry/Lancaster/Gronke do NOT define θ at a dynamical
transition — they measure a *developed/sustained* cooling efficiency (El-Badry's θ is time-independent,
plateaued at t~3.5 Myr; Lancaster's bubbles are efficiently cooled throughout). So the calibration
target is TRINITY's DEVELOPED θ (θ at the end of the energy-driven phase = its most-developed
instantaneous θ, the plateau analog) and the CUMULATIVE radiated fraction ∫Lcool dt/∫Lmech dt (the
"efficiently cooled" metric). Blowout is kept only as a dynamical diagnostic (where the energy→momentum
handoff sits); it is roughly where TRINITY's energy phase ends, so θ_developed ≈ the old θ_at_blowout.

THE GOAL (PLAN.md ⭐ synthesis): calibrate f_κ(properties) via the κ_eff mechanism
(`cooling_boost_kappa`, Rung A, built/gated) so the EMERGENT θ = L_cool/L_mech tracks the
obs/3D target θ(n_H) (Lancaster ≈0.9–0.99). The leverage first-cut (`make_fkappa_leverage.py`)
showed L_cool ∝ f_κ^0.6 on captured SNAPSHOTS; this harness checks it on FULL runs at the real
transition epoch, for the configs that geometrically blow out (the compact + diffuse arms; the
dense arm cooling-fires before blowout, so its θ is already ~the trigger).

WHAT IT DOES (uses the real gated knob; NO production edit). For each config (compact
`simple_cluster`, diffuse `f1edge_lowdens`) × f_κ ∈ {1,2,4}, a full `run.py` sim was run to a
`stop_t` capped just past blowout (params `runs/params/cal_{compact,diffuse}__k{1,2,4}.param`,
output `outputs/kcal/`). This reads each run's `dictionary.jsonl` + `metadata.json` and reports,
at the energy→momentum handoff:
  - blowout_t  : first implicit-phase t with R2 > rCloud
  - theta_blowout : bubble_LTotal / Lmech_total at that row  (the resolved loss fraction)
  - fired_cooling : whether the cooling trigger fired before blowout (then there is no R2>rCloud
                    crossing and the transition row is the cooling-fire)
The f_κ=1 baseline must reproduce the known resolved θ at blowout (simple_cluster ≈0.667,
lowdens ≈0.25 — from `fmix_table.csv`); that is the built-in correctness check. Then we compare
θ(f_κ)/θ(1) to the f_κ^0.6 leverage and read off the f_κ that reaches the obs/3D target band.

REPRODUCE (from repo root):
    for c in compact diffuse; do for k in 1 2 4; do \
      python run.py docs/dev/transition/pdv-trigger/runs/params/cal_${c}__k${k}.param; done; done
    python docs/dev/transition/pdv-trigger/data/make_kappa_blowout_calibration.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/kappa_blowout_calibration.csv
    docs/dev/transition/pdv-trigger/kappa_blowout_calibration.png
"""

import csv
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_REPO = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 5)))
_OUT = os.path.join(_REPO, "outputs", "kcal")

_CONFIGS = [
    ("compact", "simple_cluster (mCloud 1e5, sfe 0.3)", 0.667),  # baseline θ from fmix_table
    ("diffuse", "f1edge_lowdens (nCore 1e2)", 0.250),
]
_FK = [1, 2, 4]
_TARGET = (0.90, 0.99)


def _find(o, k):
    if isinstance(o, dict):
        if k in o and not isinstance(o[k], (dict, list)):
            return o[k]
        for v in o.values():
            r = _find(v, k)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find(v, k)
            if r is not None:
                return r
    return None


def _g(d, k):
    try:
        return float(d.get(k))
    except (TypeError, ValueError):
        return float("nan")


def harvest(run_dir):
    """Return (blowout_t, theta_blowout, theta_handoff, fired_cooling, ok)."""
    meta_p = os.path.join(run_dir, "metadata.json")
    dict_p = os.path.join(run_dir, "dictionary.jsonl")
    if not (os.path.exists(meta_p) and os.path.exists(dict_p)):
        return dict(blowout_t=np.nan, theta_blowout=np.nan, theta_handoff=np.nan,
                    fired_cooling=None, ok=False)
    meta = json.load(open(meta_p))
    rCloud = _find(meta, "rCloud")
    rCloud = float(rCloud) if rCloud is not None else float("nan")
    rows = [json.loads(l) for l in open(dict_p) if l.strip()]
    impl = [d for d in rows if d.get("current_phase") == "implicit"]
    if not impl:
        return dict(blowout_t=np.nan, theta_blowout=np.nan, theta_handoff=np.nan,
                    fired_cooling=None, ok=False)
    t = np.array([_g(d, "t_now") for d in impl])
    R2 = np.array([_g(d, "R2") for d in impl])
    Lcool = np.array([_g(d, "bubble_LTotal") for d in impl])
    Lmech = np.array([_g(d, "Lmech_total") for d in impl])
    theta = Lcool / Lmech                                   # instantaneous cooling efficiency θ(t)

    # --- the LITERATURE-analog metrics (developed / sustained, not a blowout instant) ---
    # theta_developed: θ at the end of the energy-driven phase = TRINITY's most-developed
    #   instantaneous θ (the closest analog to El-Badry's plateaued θ at t~3.5 Myr).
    # cum_frac: cumulative radiated fraction ∫Lcool dt / ∫Lmech dt over the energy phase =
    #   the "efficiently cooled" metric (energy radiated / energy injected).
    good = np.isfinite(theta) & np.isfinite(t)
    theta_developed = float(theta[good][-1]) if good.any() else float("nan")
    theta_max = float(np.nanmax(theta)) if good.any() else float("nan")
    still_rising = bool(theta_developed > 0.97 * theta_max) if good.any() else None
    if good.sum() >= 2:
        cum_frac = float(np.trapz(Lcool[good], t[good]) / np.trapz(Lmech[good], t[good]))
    else:
        cum_frac = float("nan")

    # blowout-θ kept as a DYNAMICAL diagnostic only (where the energy->momentum handoff sits)
    cross = np.where(R2 > rCloud)[0] if rCloud == rCloud else np.array([], int)
    if len(cross):
        bi = int(cross[0])
        blowout_t = float(t[bi])
        theta_blowout = float(theta[bi])
        fired_cooling = bool(t[-1] < blowout_t - 1e-12)
    else:
        blowout_t = float("nan")
        theta_blowout = theta_developed
        fired_cooling = True
    return dict(blowout_t=blowout_t, theta_blowout=theta_blowout, theta_developed=theta_developed,
                theta_max=theta_max, cum_frac=cum_frac, still_rising=still_rising,
                fired_cooling=fired_cooling, ok=True)


def main():
    rows = []
    series = {}  # label -> (fk[], theta_developed[], cum_frac[])
    for cfg, label, theta_ref in _CONFIGS:
        fk_l, thd_l, cf_l = [], [], []
        base = None
        for fk in _FK:
            run_dir = os.path.join(_OUT, f"cal_{cfg}__k{fk}")
            h = harvest(run_dir)
            if fk == 1:
                base = h["theta_developed"]
                tag = ("OK" if (np.isfinite(base) and abs(base - theta_ref) / theta_ref < 0.20)
                       else "CHECK")
                print(f"[{cfg}] f_κ=1 θ_developed={base:.4g} vs fmix_table {theta_ref:.3g} "
                      f"-> correctness {tag}")
            mult = h["theta_developed"] / base if (base and np.isfinite(h["theta_developed"])) else np.nan
            fk_l.append(fk); thd_l.append(h["theta_developed"]); cf_l.append(h["cum_frac"])
            rows.append({"config": cfg, "f_kappa": fk,
                         "theta_developed": h["theta_developed"], "theta_max": h["theta_max"],
                         "cum_frac": h["cum_frac"], "still_rising": h["still_rising"],
                         "theta_developed_over_base": mult,
                         "blowout_t": h["blowout_t"], "theta_blowout": h["theta_blowout"],
                         "fired_cooling": h["fired_cooling"], "base_theta_dev": base,
                         "fmix_table_ref": theta_ref, "ok": h["ok"]})
            print(f"      f_κ={fk}  θ_dev={h['theta_developed']:.4g} (×{mult:.2f})  "
                  f"cum_frac={h['cum_frac']:.4g}  still_rising={h['still_rising']}  "
                  f"blowout_t={h['blowout_t']}")
        series[label] = (np.array(fk_l, float), np.array(thd_l), np.array(cf_l))

    csv_path = os.path.join(_HERE, "kappa_blowout_calibration.csv")
    cols = ["config", "f_kappa", "theta_developed", "theta_max", "cum_frac", "still_rising",
            "theta_developed_over_base", "blowout_t", "theta_blowout", "fired_cooling",
            "base_theta_dev", "fmix_table_ref", "ok"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    markers = ["o-", "s--", "^:"]
    # LEFT: the DEVELOPED θ (the literature-plateau analog) vs f_κ
    axL.axhspan(_TARGET[0], _TARGET[1], color="#2ca02c", alpha=0.10, label="obs/3D developed θ (0.9–0.99)")
    for (label, (fk, thd, cf)), m in zip(series.items(), markers):
        axL.plot(fk, thd, m, lw=2.2, ms=7, label=label)
        if np.isfinite(thd[0]):
            axL.plot(fk, np.minimum(0.99, thd[0] * fk ** 0.63), ":", color="0.6", lw=1.2)
    axL.set_xscale("log", base=2)
    axL.set_xlabel(r"$f_\kappa$  (cooling_boost_kappa)")
    axL.set_ylabel(r"developed $\theta = L_{\rm cool}/L_{\rm mech}$ (end of energy phase)")
    axL.set_title("Does κ_eff lift the DEVELOPED θ toward obs/3D?\n"
                  "(dotted = f_κ^0.63 snapshot-leverage prediction)", fontsize=10, fontweight="bold")
    axL.legend(fontsize=8.5, loc="best")
    # RIGHT: the CUMULATIVE radiated fraction (the "efficiently cooled" metric) vs f_κ
    axR.axhspan(_TARGET[0], _TARGET[1], color="#2ca02c", alpha=0.10, label="obs/3D (0.9–0.99)")
    for (label, (fk, thd, cf)), m in zip(series.items(), markers):
        axR.plot(fk, cf, m, lw=2.2, ms=7, label=label)
    axR.set_xscale("log", base=2)
    axR.set_xlabel(r"$f_\kappa$")
    axR.set_ylabel(r"cumulative radiated fraction  $\int L_{\rm cool}dt / \int L_{\rm mech}dt$")
    axR.set_title("…and the cumulative cooling efficiency?\n(the El-Badry/Lancaster 'efficiently cooled' metric)",
                  fontsize=10, fontweight="bold")
    axR.legend(fontsize=8.5, loc="best")
    fig.suptitle("f_κ calibration (MEASURED, full runs): the literature-analog DEVELOPED / CUMULATIVE θ, "
                 "not a blowout instant", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "kappa_blowout_calibration.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
