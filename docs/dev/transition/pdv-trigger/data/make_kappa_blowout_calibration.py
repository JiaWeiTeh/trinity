#!/usr/bin/env python3
"""f_kappa blowout-theta calibration — does the leverage hold at BLOWOUT, across density?

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
    theta = Lcool / Lmech
    cross = np.where(R2 > rCloud)[0] if rCloud == rCloud else np.array([], int)
    if len(cross):
        bi = int(cross[0])
        blowout_t = float(t[bi])
        theta_blowout = float(theta[bi])
        fired_cooling = bool(t[-1] < blowout_t - 1e-12)  # transition strictly before geometric blowout
    else:
        bi = len(impl) - 1
        blowout_t = float("nan")
        theta_blowout = float(theta[bi])  # no geometric blowout in-window -> transition-row θ
        fired_cooling = True
    return dict(blowout_t=blowout_t, theta_blowout=theta_blowout,
                theta_handoff=float(theta[-1]), fired_cooling=fired_cooling, ok=True)


def main():
    rows = []
    series = {}
    for cfg, label, theta_ref in _CONFIGS:
        fk_l, th_l = [], []
        base = None
        for fk in _FK:
            run_dir = os.path.join(_OUT, f"cal_{cfg}__k{fk}")
            h = harvest(run_dir)
            if fk == 1:
                base = h["theta_blowout"]
                tag = "OK" if (np.isfinite(base) and abs(base - theta_ref) / theta_ref < 0.15) else "CHECK"
                print(f"[{cfg}] f_κ=1 θ_blowout={base:.4g} vs fmix_table {theta_ref:.3g} -> correctness {tag}"
                      f" (fired_cooling={h['fired_cooling']})")
            mult = h["theta_blowout"] / base if (base and np.isfinite(h["theta_blowout"])) else np.nan
            fk_l.append(fk)
            th_l.append(h["theta_blowout"])
            rows.append({"config": cfg, "f_kappa": fk, "blowout_t": h["blowout_t"],
                         "theta_blowout": h["theta_blowout"], "theta_handoff": h["theta_handoff"],
                         "theta_over_base": mult, "fired_cooling": h["fired_cooling"],
                         "base_theta": base, "fmix_table_ref": theta_ref, "ok": h["ok"]})
            print(f"      f_κ={fk}  θ_blowout={h['theta_blowout']:.4g}  θ/base={mult:.3f}  "
                  f"fired_cooling={h['fired_cooling']}")
        series[label] = (np.array(fk_l, float), np.array(th_l))

    csv_path = os.path.join(_HERE, "kappa_blowout_calibration.csv")
    cols = ["config", "f_kappa", "blowout_t", "theta_blowout", "theta_handoff",
            "theta_over_base", "fired_cooling", "base_theta", "fmix_table_ref", "ok"]
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
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.axhspan(_TARGET[0], _TARGET[1], color="#2ca02c", alpha=0.10, label="obs/3D target θ (0.9–0.99)")
    markers = ["o-", "s--", "^:"]
    for (label, (fk, th)), m in zip(series.items(), markers):
        ax.plot(fk, th, m, lw=2.2, ms=7, label=label)
        # f_κ^0.6 leverage prediction anchored at f_κ=1
        if np.isfinite(th[0]):
            ax.plot(fk, np.minimum(0.99, th[0] * fk ** 0.6), ":", color="0.6", lw=1.3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$f_\kappa$  (cooling_boost_kappa)")
    ax.set_ylabel(r"emergent $\theta = L_{\rm cool}/L_{\rm mech}$ at blowout")
    ax.set_title("f_κ blowout-θ calibration: does κ_eff lift the emergent θ toward obs/3D?\n"
                 "(dotted grey = the f_κ^0.6 leverage from the snapshot first-cut)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    png = os.path.join(_PDV, "kappa_blowout_calibration.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
