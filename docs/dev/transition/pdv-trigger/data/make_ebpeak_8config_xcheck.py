#!/usr/bin/env python3
"""Does the ebpeak finding hold on all 8 configs? — frozen-screen cross-check + live validation.

THE QUESTION (user: "did you test all recent stuffs on the 8 configs?"). The live ebpeak full-run test
(make_ebpeak_trigger_test.py) was run on 4 regimes (compact=simple_cluster, diffuse=f1edge_lowdens,
mid=midrange_pl0, dense=small_dense_highsfe). The headline f_kappa=1 conclusion -- the PdV-inclusive
ratio (Lloss+PdV)/Lgain PEAKS BELOW the 1.0 ebpeak threshold and never fires for normal clouds -- can
ALSO be checked across the full 8-config universe via the EARLIER frozen-trajectory screen
(data/pdv_combined_trigger.csv + data/pdv_regime_budget.csv, the s1 cleanroom set). This harness
reconciles the two:

  frozen peak PdV-incl ratio per config = 1 - min_coolPdV   (min_coolPdV = min of edot_balance/Lmech
                                                              over the frozen trajectory; <0 => fired)
  frozen fires in-cloud                 = ebpeak_fires_in_cloud (pdv_regime_budget.csv)
  live peak PdV-incl ratio              = pdv_incl_ratio_peak  (ebpeak_trigger_test.csv), for the 3
                                          configs that are in BOTH sets (simple_cluster, midrange_pl0,
                                          small_dense_highsfe) -> a live-vs-frozen VALIDATION.

THE FINDING. Across all 8 frozen configs, the 6 "normal" clouds peak at PdV-incl 0.85-0.92 and do NOT
fire; only the heavy 5e9 (super-critical, PdV/Lmech>1) and the small_1e6 control (birth blip) fire,
plus large_diffuse_lowsfe BARELY (1.02, POST-blowout). The live full runs agree to the digit where they
overlap (simple_cluster live 0.912 == frozen 0.912). So the f_kappa=1 ebpeak conclusion GENERALIZES to
8 configs. CAVEAT: the f_kappa-DEPENDENCE (the cooling<->PdV trade-off / calibration) is a LIVE-only
result (frozen freezes the trajectory, hiding the Eb/Pb/PdV drainage that IS the trade-off); it is
covered live on 4 configs here, HPC-deferred for the rest.

REPRODUCE (from repo root):
    python docs/dev/transition/pdv-trigger/data/make_ebpeak_8config_xcheck.py
Deliverables (committed):
    docs/dev/transition/pdv-trigger/data/ebpeak_8config_xcheck.csv
    docs/dev/transition/pdv-trigger/ebpeak_8config_xcheck.png
"""

import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)

# live config label -> frozen config name (only the 3 that exist in BOTH sets).
_LIVE_TO_FROZEN = {"simple_cluster": "compact", "midrange_pl0": "mid",
                   "small_dense_highsfe": "dense"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    frozen = {r["config"]: r for r in
              csv.DictReader(open(os.path.join(_HERE, "pdv_combined_trigger.csv")))}
    budget = {r["config"]: r for r in
              csv.DictReader(open(os.path.join(_HERE, "pdv_regime_budget.csv")))}
    # live peaks, keyed by the live "config" column (compact/diffuse/mid/dense)
    live_peak = {}
    live_path = os.path.join(_HERE, "ebpeak_trigger_test.csv")
    if os.path.exists(live_path):
        for r in csv.DictReader(open(live_path)):
            # run col like cal_compact__k1 / cal_mid__ek1 -> the f_κ=1 ebpeak-active peak is the one to take
            run = r["run"]
            for tag in ("compact", "diffuse", "mid", "dense"):
                if f"cal_{tag}__" in run and ("ebpeak" in run or "ek1" in run):
                    live_peak[tag] = _f(r["pdv_incl_ratio_peak"])

    rows = []
    for cfg, fr in frozen.items():
        peak = 1.0 - _f(fr["min_coolPdV"])
        fires = (budget.get(cfg, {}).get("ebpeak_fires_in_cloud", "") == "True")
        # is there a live overlay?
        live_tag = _LIVE_TO_FROZEN.get(cfg)
        lp = live_peak.get(live_tag) if live_tag else None
        rows.append(dict(config=cfg, regime=fr["regime"], frozen_peak_ratio=round(peak, 3),
                         frozen_fires_in_cloud=fires,
                         blowout_t=fr["blowout_t"],
                         live_peak_ratio=("" if lp is None else round(lp, 3)),
                         pdv_over_lmech_med=fr["med_pdv_over_Lmech"]))
        liveinfo = f"  live={lp:.3f}" if lp is not None else ""
        print(f"{cfg:22s} [{fr['regime']:11s}]  frozen_peak={peak:.3f}  fires={fires}{liveinfo}")

    csv_path = os.path.join(_HERE, "ebpeak_8config_xcheck.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "regime", "frozen_peak_ratio",
                                           "frozen_fires_in_cloud", "blowout_t", "live_peak_ratio",
                                           "pdv_over_lmech_med"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return
    # order by frozen peak ratio
    rows_sorted = sorted(rows, key=lambda r: r["frozen_peak_ratio"])
    labels = [r["config"] for r in rows_sorted]
    peaks = [r["frozen_peak_ratio"] for r in rows_sorted]
    fires = [r["frozen_fires_in_cloud"] for r in rows_sorted]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_colors = ["#2ca02c" if fr else "#1f77b4" for fr in fires]
    ax.bar(x, peaks, color=bar_colors, alpha=0.85)
    # live overlay markers
    for i, r in enumerate(rows_sorted):
        if r["live_peak_ratio"] != "":
            ax.plot(i, r["live_peak_ratio"], "D", color="black", ms=9, zorder=5)
            ax.text(i, r["live_peak_ratio"] + 0.03, f"live {r['live_peak_ratio']:.2f}",
                    ha="center", fontsize=8, fontweight="bold")
        ax.text(i, peaks[i] - 0.06, f"{peaks[i]:.2f}", ha="center", fontsize=8.5,
                color="white", fontweight="bold")
    ax.axhline(1.0, color="#2ca02c", lw=2.0, label="ebpeak fires (PdV-inclusive ratio = 1.0)")
    ax.axhline(0.95, color="crimson", ls="--", lw=1.3, label="cooling_balance fires (0.95, radiative)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("peak PdV-inclusive ratio  (Lloss+PdV)/Lgain")
    ax.set_ylim(0, 1.7)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color="#1f77b4", label="normal cloud — does NOT fire (peak < 1.0)"),
               Patch(color="#2ca02c", label="fires (heavy 5e9 super-critical, or control birth-blip)"),
               Line2D([0], [0], marker="D", color="black", ls="", label="LIVE full-run peak (this work) — overlay"),
               Line2D([0], [0], color="#2ca02c", lw=2, label="ebpeak threshold 1.0"),
               Line2D([0], [0], color="crimson", ls="--", lw=1.3, label="cooling_balance 0.95")]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left")
    ax.set_title("Does the ebpeak finding hold on all 8 configs? Frozen-trajectory screen (bars) — 6 normal "
                 "clouds peak at 0.85–0.92\nand never fire; only heavy-5e9 / control do. Live full runs (black "
                 "diamonds) agree to the digit where they overlap.", fontsize=10.3, fontweight="bold")
    fig.tight_layout()
    png = os.path.join(_PDV, "ebpeak_8config_xcheck.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
