#!/usr/bin/env python3
"""What if we include PdV in the transition trigger? — cooling_balance vs ebpeak, on the cal runs.

THE TWO TRIGGERS (verified in run_energy_implicit_phase.py).
  cooling_balance (DEFAULT): fires when Lloss/Lgain > 0.95  -> needs (Lcool+leak)/Lmech >= 0.95
                             (RADIATIVE only; ignores the PdV work the bubble does on the shell).
  ebpeak (opt-in):           fires when edot_balance <= 0   -> (Lcool+leak+PdV)/Lmech >= 1
                             edot_balance = Lmech - (Lcool+leak) - 4*pi*R2^2*v2*Pb  (PdV-INCLUSIVE;
                             the bubble's net energy stops growing -> Eb peaks -> momentum naturally).
Including PdV is NOT double-counting: PdV is already in dEb/dt; ebpeak just watches the full balance.

WHY IT MATTERS. TRINITY's 1D radiative cooling is weak (Lcool/Lmech ~ 0.14-0.67), so cooling_balance
needs a large f_kappa boost (diffuse ~60x) to fire. But PdV is the DOMINANT sink (PdV/Lmech ~ 0.2-0.7),
so the PdV-inclusive (ebpeak) ratio is already ~0.65-0.91 at f_kappa=1 -> the transition is mostly
there WITHOUT a big cooling boost. This harness quantifies that on the committed cal runs (outputs/kcal/),
reading each run's dictionary.jsonl at cloud dispersal (first R2>rCloud).

CAVEAT (honest): including PdV fixes WHEN the bubble transitions, not WHETHER it is efficiently COOLED.
The literature (Lancaster/El-Badry/Gronke) bubble radiates theta~0.9; TRINITY's still only radiates
0.14-0.67. So ebpeak addresses the transition-timing goal; kappa_eff is still needed for the cooling-
MAGNITUDE goal. They are complementary. There is also a trade-off: boosting cooling drains Eb -> lowers
Pb -> lowers PdV, so (Lcool+PdV) is less sensitive to f_kappa than Lcool alone (seen in the diffuse arm).

REPRODUCE (from repo root; needs the cal runs in outputs/kcal/, see make_kappa_blowout_calibration.py):
    python docs/dev/transition/pdv-trigger/data/make_pdv_trigger_compare.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/pdv_trigger_compare.csv
    docs/dev/transition/pdv-trigger/pdv_trigger_compare.png
"""

import csv
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_REPO = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 5)))
_OUT = os.path.join(_REPO, "outputs", "kcal")
_CONFIGS = [("compact", "simple_cluster"), ("diffuse", "f1edge_lowdens")]
_FK = [1, 2, 4]
_COOLING_BALANCE = 0.95
_EBPEAK = 1.00


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


def at_dispersal(run_dir):
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    rC = _find(meta, "rCloud")
    rC = float(rC) if rC is not None else float("nan")
    rows = [json.loads(x) for x in open(os.path.join(run_dir, "dictionary.jsonl")) if x.strip()]
    impl = [d for d in rows if d.get("current_phase") == "implicit"]

    def col(k):
        return np.array([float(d.get(k)) if d.get(k) is not None else np.nan for d in impl], float)

    R2, v2, Pb = col("R2"), col("v2"), col("Pb")
    Lcool, Lmech = col("bubble_LTotal"), col("Lmech_total")
    leak = col("bubble_Leak")
    leak = np.where(np.isfinite(leak), leak, 0.0)
    PdV = 4 * math.pi * R2 ** 2 * v2 * Pb
    cool_only = (Lcool + leak) / Lmech
    pdv_incl = (Lcool + leak + PdV) / Lmech
    cross = np.where(R2 > rC)[0] if rC == rC else np.array([], int)
    i = int(cross[0]) if len(cross) else int(np.nanargmax(cool_only))  # dispersal, or cool-fire row
    return dict(cool_only=float(cool_only[i]), pdv_over_lmech=float((PdV / Lmech)[i]),
                pdv_incl=float(pdv_incl[i]), pdv_incl_max=float(np.nanmax(pdv_incl)),
                fired_dispersal=bool(len(cross)))


def main():
    rows = []
    data = {}
    for cfg, name in _CONFIGS:
        data[cfg] = []
        for fk in _FK:
            h = at_dispersal(os.path.join(_OUT, f"cal_{cfg}__k{fk}"))
            data[cfg].append(h)
            rows.append({"config": cfg, "f_kappa": fk, **h})
            print(f"[{cfg}] f_κ={fk}: cool_only={h['cool_only']:.3f}  PdV/Lmech={h['pdv_over_lmech']:.3f}  "
                  f"PdV_incl={h['pdv_incl']:.3f}  (cooling_balance@{_COOLING_BALANCE}, ebpeak@{_EBPEAK})")

    csv_path = os.path.join(_HERE, "pdv_trigger_compare.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "f_kappa", "cool_only", "pdv_over_lmech",
                                           "pdv_incl", "pdv_incl_max", "fired_dispersal"])
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (cfg, name) in zip(axes, _CONFIGS):
        hs = data[cfg]
        x = np.arange(len(_FK))
        cool = [h["cool_only"] for h in hs]
        pdv = [h["pdv_over_lmech"] for h in hs]
        ax.bar(x, cool, color="#1f77b4", label="radiative  Lcool/Lmech")
        ax.bar(x, pdv, bottom=cool, color="#ff7f0e", label="PdV/Lmech  (the dominant sink)")
        ax.axhline(_COOLING_BALANCE, color="crimson", ls="--", lw=1.4, label="cooling_balance (0.95, radiative-only)")
        ax.axhline(_EBPEAK, color="#2ca02c", ls="-", lw=1.6, label="ebpeak (1.0, PdV-inclusive)")
        for xi, h in zip(x, hs):
            ax.text(xi, h["cool_only"] + h["pdv_over_lmech"] + 0.02, f"{h['pdv_incl']:.2f}",
                    ha="center", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([f"f_κ={k}" for k in _FK])
        ax.set_title(f"{cfg}  ({name})", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("loss / Lmech  (transition fires when the bar reaches the line)")
    axes[0].legend(fontsize=7.6, loc="upper left")
    fig.suptitle("Include PdV in the trigger? PdV is the DOMINANT sink — the PdV-inclusive (ebpeak) bar is "
                 "near 1.0 already,\nwhile radiative-only (blue) needs a huge boost to reach 0.95",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(_PDV, "pdv_trigger_compare.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
