#!/usr/bin/env python3
"""Why every enhanced-cooling knob is weak: the Θ ∝ f^q exponents converge on the Weaver 2/7.

Pure re-reduction of `data/bench7_analysis.csv` — **no runs, no new measurement.** It reads the
committed EXPONENT and ENTRY rows and asks two questions the campaign recorded the inputs for but
never put side by side (kappa-3way `FINDINGS §17`):

  table=EXPONENT   Group the measured q by *what the knob acts through*. f_A and f_κ both act via
                   conduction; f_mix multiplies `L_cool` directly. If `§13`'s diagnosis is right —
                   TRINITY's Ṁ is the Weaver v(R1)=0 eigenvalue, absorbing any conduction-side
                   boost as f^{2/7} — then the two conduction knobs should land on **2/7 = 0.2857**
                   and f_mix should not. `§13` measured that exponent on the per-call mass flux;
                   this is the independent full-run Θ check of the same claim.

  table=SPREAD     Band entry is f = (0.90/Θ₀)^{1/q}, so the inter-bench dose spread the campaign
                   ranks knobs by is largely Θ₀ variation propagated through 1/q. This computes what
                   each knob's spread *would* be at its own mean q if q were identical across
                   benches, and compares that to the measured spread. If the two track, the spread
                   metric is measuring response steepness, not physical fidelity.

⚠️ Scope. This re-reads one committed CSV; it measures nothing new and cannot. It does not overturn
`§2`/`§11`/`§12` — those numbers stand — it reinterprets what the ranking metric is sensitive to.
f_κ's bench2/bench1 entries are flagged `no — EXTRAPOLATED` upstream (169, 143), so its measured
spread is partly an extrapolation artifact and is reported with that flag attached.

Usage (from the repo root):
    python docs/dev/transition/pdv-trigger/data/make_exponent_synthesis.py

Reads   docs/dev/transition/pdv-trigger/data/bench7_analysis.csv   (committed, stamped 2026-07-31)
Writes  docs/dev/transition/pdv-trigger/data/exponent_synthesis.csv
"""

import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _stamp import stamp  # noqa: E402

SRC = HERE / "bench7_analysis.csv"
OUT = HERE / "exponent_synthesis.csv"

WEAVER = 2.0 / 7.0  # 0.285714… — the v(R1)=0 evaporation eigenvalue exponent (FINDINGS §13)
BAND_LO = 0.90

# What each knob physically acts through. This grouping is the whole point of the section:
# it is the hypothesis §13/§15 imply, tested against exponents measured independently of them.
ACTS_THROUGH = {"fA": "conduction", "fkappa": "conduction", "fmix": "L_cool directly"}

# Θ₀ per bench, from the f=1 rung of every ENTRY ladder (cross-checked against FINDINGS §1 and
# re-measured post-merge in data/merge_rebaseline.csv, which moved bench1 by 1.3%).
THETA0 = {"bench3_m1e5_r5": 0.462, "bench2_m1e5_r10": 0.341, "bench1_m5e4_r20": 0.221}


def read_rows():
    lines = [ln for ln in SRC.open(encoding="utf-8") if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def main():
    if not SRC.exists():
        print(f"missing {SRC}")
        return 2
    rows = read_rows()
    # entry_dose carries the exponent on EXPONENT rows and the dose on ENTRY rows (upstream reuses
    # the column); bench == 'SPREAD(max/min)' marks the upstream-computed spread row.
    exps, spreads = {}, {}
    for r in rows:
        knob, bench, val = r.get("knob"), r.get("bench"), r.get("entry_dose")
        if not knob or not val:
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if r["table"] == "EXPONENT":
            exps.setdefault(knob, {})[bench] = v
        elif r["table"] == "ENTRY":
            if bench == "SPREAD(max/min)":
                spreads[knob] = (v, r.get("measured_in_grid", ""))

    out = []
    print("=== EXPONENT — grouped by what the knob acts through ===")
    print(f"Weaver v(R1)=0 eigenvalue exponent 2/7 = {WEAVER:.4f}  (FINDINGS §13, per-call Ṁ)\n")
    print(
        f"{'knob':8} {'acts through':17} {'q per bench':>26} {'mean q':>8} {'q/(2/7)':>8} {'spread':>7}"
    )
    for knob in ("fA", "fkappa", "fmix"):
        d = exps.get(knob, {})
        qs = [v for v in d.values()]
        if not qs:
            continue
        mean = statistics.fmean(qs)
        per = " ".join(f"{v:.3f}" for v in qs)
        print(
            f"{knob:8} {ACTS_THROUGH[knob]:17} {per:>26} {mean:8.3f} {mean / WEAVER:8.2f} "
            f"{max(qs) / min(qs):6.2f}x"
        )
        out.append(
            {
                "table": "EXPONENT",
                "knob": knob,
                "acts_through": ACTS_THROUGH[knob],
                "q_per_bench": per,
                "q_mean": f"{mean:.6g}",
                "q_over_weaver_2_7": f"{mean / WEAVER:.4g}",
                "q_spread_max_over_min": f"{max(qs) / min(qs):.4g}",
                "n_benches": len(qs),
            }
        )

    ratios = [BAND_LO / t for t in THETA0.values()]
    t0_spread = max(ratios) / min(ratios)
    print("\n=== SPREAD — is the dose spread just Θ₀ variation through 1/q? ===")
    print(
        f"(0.90/Θ₀) across benches = {', '.join(f'{x:.3f}' for x in ratios)} → ratio {t0_spread:.3f}×\n"
    )
    print(
        f"{'knob':8} {'mean q':>7} {'predicted spread':>17} {'measured':>9} {'pred/meas':>10}  flag"
    )
    for knob in ("fA", "fkappa", "fmix"):
        qs = list(exps.get(knob, {}).values())
        if not qs or knob not in spreads:
            continue
        mean = statistics.fmean(qs)
        pred = t0_spread ** (1.0 / mean)
        meas, flag = spreads[knob]
        print(f"{knob:8} {mean:7.3f} {pred:16.1f}× {meas:8.1f}× {pred / meas:10.2f}  {flag}")
        out.append(
            {
                "table": "SPREAD",
                "knob": knob,
                "acts_through": ACTS_THROUGH[knob],
                "q_mean": f"{mean:.6g}",
                "theta0_ratio_spread": f"{t0_spread:.6g}",
                "predicted_dose_spread_at_mean_q": f"{pred:.6g}",
                "measured_dose_spread": f"{meas:.6g}",
                "predicted_over_measured": f"{pred / meas:.4g}",
                "upstream_flag": flag,
            }
        )

    cols = [
        "table",
        "knob",
        "acts_through",
        "q_per_bench",
        "q_mean",
        "q_over_weaver_2_7",
        "q_spread_max_over_min",
        "n_benches",
        "theta0_ratio_spread",
        "predicted_dose_spread_at_mean_q",
        "measured_dose_spread",
        "predicted_over_measured",
        "upstream_flag",
    ]
    with OUT.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# SOURCES READ: bench7_analysis.csv <- re-reduction only, no runs.\n")
        fh.write("# EXPONENT: measured Theta ~ f^q grouped by the channel the knob acts through.\n")
        fh.write(f"# Weaver v(R1)=0 eigenvalue exponent 2/7 = {WEAVER:.6f} (FINDINGS 13).\n")
        fh.write(
            "# SPREAD: predicted = (Theta0 ratio spread)^(1/q_mean), i.e. what the inter-bench\n"
        )
        fh.write(
            "# dose spread would be from Theta0 variation ALONE at that knob's own exponent.\n"
        )
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
