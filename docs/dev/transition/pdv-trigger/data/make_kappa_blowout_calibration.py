#!/usr/bin/env python3
"""f_kappa calibration on full runs — does κ_eff push θ to the 0.95 cooling-fire trigger?

THE PHYSICS (verified against the trigger code). TRINITY's DEFAULT energy→momentum trigger is
`cooling_balance`: it fires when Lloss/Lgain > 0.95 (run_energy_implicit_phase.py:1206, threshold
0.05). That is a COOLING-DRIVEN transition — the same picture as El-Badry/Lancaster/Gronke (cooling
creeps up; the bubble goes momentum naturally), with NO geometric blowout (blowout is opt-in,
default off). The problem: TRINITY's 1D cooling is too weak — the resolved θ = L_cool/L_mech develops
only to ~0.17–0.70, below 0.95 — so for normal clouds `cooling_balance` never fires and the bubble
just runs past cloud dispersal under-cooled. κ_eff (`cooling_boost_kappa`) is the knob to push θ up so
the cooling-driven transition fires; the calibration is HOW MUCH, per cloud property.

THE METRIC (corrected). θ(t) RISES through the in-cloud energy phase, PEAKS at cloud dispersal, then
DROPS as the bubble exits into low-density ISM. So the developed in-cloud value is `theta_blowout`
(θ at first R2>rCloud) / `theta_max` — NOT the last-row value (that is the post-dispersal decline).
The decisive outcome per run is `cooling_fired`: did θ reach 0.95 and the run enter the momentum phase
via cooling (vs running past dispersal). `cum_frac = ∫Lcool dt/∫Lmech dt` is the cumulative
"efficiently cooled" metric. Correctness check: `theta_blowout` at f_κ=1 reproduces the known baseline
(simple_cluster 0.667; f1edge_lowdens 0.169 — its OWN value, not fmix_table's large_diffuse 0.25).

WHAT IT DOES (uses the real gated knob; NO production edit). For compact `simple_cluster` + diffuse
`f1edge_lowdens` × f_κ ∈ {1,2,4}, full `run.py` sims (params `runs/params/cal_{compact,diffuse}__k*.param`,
output `outputs/kcal/`) are harvested for theta_blowout / theta_max / cum_frac / cooling_fired.

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
    # (cfg, label, baseline θ_blowout for the f_κ=1 correctness check). The compact ref is
    # simple_cluster's fmix_table value (0.667). The diffuse ref is f1edge_lowdens's OWN
    # resolved θ (~0.17) -- NOT fmix_table's large_diffuse_lowsfe (0.25, a different cloud).
    ("compact", "simple_cluster (mCloud 1e5, sfe 0.3)", 0.667),
    ("diffuse", "f1edge_lowdens (mCloud 1e7, sfe 0.5, nCore 1e2)", 0.169),
]
_FK = [1, 2, 4]
_TRIGGER = 0.95  # the cooling_balance trigger: Lloss/Lgain > 0.95 fires the cooling-driven transition


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
        return dict(blowout_t=np.nan, theta_blowout=np.nan, theta_developed=np.nan,
                    theta_max=np.nan, cum_frac=np.nan, still_rising=None,
                    reached_momentum=False, cooling_fired=False, ok=False)
    meta = json.load(open(meta_p))
    rCloud = _find(meta, "rCloud")
    rCloud = float(rCloud) if rCloud is not None else float("nan")
    rows = [json.loads(l) for l in open(dict_p) if l.strip()]
    reached_momentum = any(d.get("current_phase") in ("transition", "momentum") for d in rows)
    impl = [d for d in rows if d.get("current_phase") == "implicit"]
    if not impl:
        return dict(blowout_t=np.nan, theta_blowout=np.nan, theta_developed=np.nan,
                    theta_max=np.nan, cum_frac=np.nan, still_rising=None,
                    reached_momentum=False, cooling_fired=False, ok=False)
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

    # blowout-θ = θ at geometric cloud dispersal (first R2>rCloud); a DYNAMICAL diagnostic.
    cross = np.where(R2 > rCloud)[0] if rCloud == rCloud else np.array([], int)
    if len(cross):
        bi = int(cross[0])
        blowout_t = float(t[bi])
        theta_blowout = float(theta[bi])
    else:
        blowout_t = float("nan")          # no geometric blowout -> cooling fired first
        theta_blowout = theta_max
    # cooling_fired: the cooling-driven transition (cooling_balance, θ>0.95) actually fired
    # (reached the momentum phase via cooling, not by running past cloud dispersal).
    cooling_fired = bool(reached_momentum and (np.isnan(blowout_t) or theta_max >= _TRIGGER))
    return dict(blowout_t=blowout_t, theta_blowout=theta_blowout, theta_developed=theta_developed,
                theta_max=theta_max, cum_frac=cum_frac, still_rising=still_rising,
                reached_momentum=reached_momentum, cooling_fired=cooling_fired, ok=True)


def main():
    # PRIMARY metric = theta_blowout (= θ at cloud dispersal = TRINITY's developed IN-CLOUD
    # cooling efficiency; θ peaks here then DROPS as the bubble exits into low-density ISM, so
    # the last-row value is NOT the developed one). The decisive outcome is whether θ reaches the
    # 0.95 cooling_balance trigger -> the cooling-driven transition fires (momentum phase) instead
    # of the bubble running past cloud dispersal under-cooled.
    rows = []
    series = {}  # label -> (fk[], theta_blowout[], cum_frac[], cooling_fired[])
    for cfg, label, theta_ref in _CONFIGS:
        fk_l, tb_l, cf_l, fired_l = [], [], [], []
        base = None
        for fk in _FK:
            h = harvest(os.path.join(_OUT, f"cal_{cfg}__k{fk}"))
            if fk == 1:
                base = h["theta_blowout"]
                tag = ("OK" if (np.isfinite(base) and abs(base - theta_ref) / theta_ref < 0.10)
                       else "CHECK")
                print(f"[{cfg}] f_κ=1 θ_blowout={base:.4g} vs baseline {theta_ref:.3g} "
                      f"-> correctness {tag}")
            mult = h["theta_blowout"] / base if (base and np.isfinite(h["theta_blowout"])) else np.nan
            fk_l.append(fk); tb_l.append(h["theta_blowout"]); cf_l.append(h["cum_frac"])
            fired_l.append(h["cooling_fired"])
            rows.append({"config": cfg, "f_kappa": fk, "theta_blowout": h["theta_blowout"],
                         "theta_max": h["theta_max"], "theta_blowout_over_base": mult,
                         "cum_frac": h["cum_frac"], "cooling_fired": h["cooling_fired"],
                         "reached_momentum": h["reached_momentum"], "blowout_t": h["blowout_t"],
                         "theta_developed_lastrow": h["theta_developed"],
                         "base_theta": base, "baseline_ref": theta_ref, "ok": h["ok"]})
            print(f"      f_κ={fk}  θ_blowout={h['theta_blowout']:.4g} (×{mult:.2f})  "
                  f"θ_max={h['theta_max']:.4g}  cum_frac={h['cum_frac']:.4g}  "
                  f"COOLING_FIRED={h['cooling_fired']}  (blowout_t={h['blowout_t']})")
        series[label] = (np.array(fk_l, float), np.array(tb_l), np.array(cf_l), fired_l)

    csv_path = os.path.join(_HERE, "kappa_blowout_calibration.csv")
    cols = ["config", "f_kappa", "theta_blowout", "theta_max", "theta_blowout_over_base",
            "cum_frac", "cooling_fired", "reached_momentum", "blowout_t",
            "theta_developed_lastrow", "base_theta", "baseline_ref", "ok"]
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
    # LEFT: developed θ (at cloud dispersal) vs f_κ, with the 0.95 cooling-fire threshold
    axL.axhline(_TRIGGER, color="crimson", lw=1.5, ls="--", label="0.95 cooling_balance trigger")
    for (label, (fk, tb, cf, fired)), m in zip(series.items(), markers):
        axL.plot(fk, tb, m, lw=2.2, ms=7, label=label)
        if np.isfinite(tb[0]):
            axL.plot(fk, np.minimum(1.3, tb[0] * fk ** 0.63), ":", color="0.6", lw=1.1)
        for x, y, f in zip(fk, tb, fired):  # ring the f_κ where cooling actually fired
            if f:
                axL.plot(x, y, "o", ms=15, mfc="none", mec="crimson", mew=2, zorder=5)
    axL.set_xscale("log", base=2)
    axL.set_xlabel(r"$f_\kappa$  (cooling_boost_kappa)")
    axL.set_ylabel(r"developed $\theta = L_{\rm cool}/L_{\rm mech}$ (at cloud dispersal)")
    axL.set_title("Does κ_eff push θ to the 0.95 cooling-fire trigger?\n"
                  "(red ring = cooling_balance FIRED → momentum; dotted = f_κ^0.63 snapshot estimate)",
                  fontsize=9.5, fontweight="bold")
    axL.legend(fontsize=8, loc="upper left")
    # RIGHT: cumulative radiated fraction (the 'efficiently cooled' metric) vs f_κ
    for (label, (fk, tb, cf, fired)), m in zip(series.items(), markers):
        axR.plot(fk, cf, m, lw=2.2, ms=7, label=label)
    axR.set_xscale("log", base=2)
    axR.set_xlabel(r"$f_\kappa$")
    axR.set_ylabel(r"cumulative radiated fraction  $\int L_{\rm cool}dt / \int L_{\rm mech}dt$")
    axR.set_title("Cumulative cooling efficiency\n(El-Badry/Lancaster 'efficiently cooled' metric)",
                  fontsize=9.5, fontweight="bold")
    axR.legend(fontsize=8.5, loc="best")
    fig.suptitle("f_κ calibration (MEASURED, full runs): compact fires the cooling transition at f_κ≈4; "
                 "diffuse needs far more — the snapshot estimate was optimistic", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "kappa_blowout_calibration.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
