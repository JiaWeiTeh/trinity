#!/usr/bin/env python3
"""bench6 analysis — the f_A vs f_mix HEAD-TO-HEAD that feeds the Phase-6 decision.

Combines the bench5 baseline (f_A ≤ 16 + the __none arms) with the bench6 extension
(f_A 24–128 on the diffuse benches + f_mix 2–8 on all benches) into one dose–response table
per (bench, knob), and computes the decision metrics:

  1. Θ_cum over the blowout window per arm (diagnostic arms; the L21b metric, §15h) — via the
     same trapezoid as make_bench5_analysis.theta_cum_prefire.
  2. BAND-ENTRY DOSE per (bench, knob): the dose at which Θ_cum first enters the L21b band
     [0.90, 0.99] (log-dose linear interpolation between grid points; '>max' if never).
  3. DOSE-UNIFORMITY: max/min band-entry dose across the clean-blowout benches, per knob —
     the knob whose calibrated dose varies LESS across density is the better single-constant.
     (The physical asymmetry — f_A responds in-structure and suppresses dMdt (El-Badry Eq-47
     sign, theta5s-measured), f_mix leaves the structure frozen — is decided without sims;
     THIS script decides the empirical calibration side.)
  4. Fire map at the extended doses (does bench1/bench2 ever fire?).

Prefers HPC data when present (bench5_summary_hpc.csv / bench5_traj_hpc), else falls back to
the in-container bench5 artifacts. bench6 data is HPC-only (runs/data/bench6_summary.csv +
bench6_traj/ via ./sync_bench.sh bench6 run && down). Reports gracefully if bench6 hasn't
landed yet.

    python docs/dev/transition/pdv-trigger/data/make_bench6_analysis.py
Deliverable: data/bench6_analysis.csv + console head-to-head table.
"""

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDV = HERE.parent
RDATA = PDV / "runs" / "data"

sys.path.insert(0, str(HERE))
from make_bench5_analysis import _fnum, _read_csv, theta_cum_prefire  # noqa: E402

L21B_BAND = (0.90, 0.99)
BENCHES = [
    "bench1_m5e4_r20",
    "bench2_m1e5_r10",
    "bench3_m1e5_r5",
    "bench4_m1e5_r2p5",
    "bench5_m5e5_r2p5",
]
CLEAN_BLOWOUT = {"bench1_m5e4_r20", "bench2_m1e5_r10", "bench3_m1e5_r5"}  # §15h end-state check


def _knob_dose(run_name):
    """'bench2__fa24_diag' -> ('fA', 24); '...__fm3' -> ('fmix', 3); '__none' -> ('fA', 1)."""
    tag = run_name.split("__", 1)[1].replace("_diag", "")
    if tag == "none":
        return "fA", 1.0
    if tag.startswith("fa"):
        return "fA", float(tag[2:])
    if tag.startswith("fm"):
        return "fmix", float(tag[2:])
    return "?", math.nan


def _load(summary, traj_dir):
    """(run_name -> summary row), plus theta_cum per diag arm from its trajectory."""
    if not summary.exists():
        return {}
    rows = {r["run_name"]: r for r in _read_csv(summary)}
    for name, r in rows.items():
        tp = traj_dir / f"{name}.csv"
        r["_theta_cum"] = None
        if name.endswith("_diag") and tp.exists():
            tcum, _, _ = theta_cum_prefire(_read_csv(tp))
            r["_theta_cum"] = tcum
    return rows


def band_entry(doses_tcum):
    """First dose where Θ_cum enters the band, log-dose interpolated; None if never."""
    pts = sorted((d, t) for d, t in doses_tcum if t is not None)
    for (d0, t0), (d1, t1) in zip(pts, pts[1:]):
        if t0 < L21B_BAND[0] <= t1:
            f = (L21B_BAND[0] - t0) / (t1 - t0)
            return math.exp(math.log(d0) + f * (math.log(d1) - math.log(d0)))
    if pts and pts[0][1] >= L21B_BAND[0]:
        return pts[0][0]
    return None


def main():
    b5 = _load(
        RDATA
        / (
            "bench5_summary_hpc.csv"
            if (RDATA / "bench5_summary_hpc.csv").exists()
            else "bench5_summary.csv"
        ),
        RDATA / ("bench5_traj_hpc" if (RDATA / "bench5_traj_hpc").is_dir() else "bench5_traj"),
    )
    b6 = _load(RDATA / "bench6_summary.csv", RDATA / "bench6_traj")
    if not b6:
        print(
            "NOTE: no bench6 data yet (runs/data/bench6_summary.csv missing) — run the bench6 "
            "array on Helix first (./sync_bench.sh bench6 submit/run/down). Showing the "
            "bench5-only dose curves meanwhile."
        )
    src = {**b5, **b6}

    out_rows = []
    for name in sorted(src):
        r = src[name]
        knob, dose = _knob_dose(name)
        out_rows.append(
            {
                "run_name": name,
                "bench": name.split("__")[0],
                "knob": knob,
                "dose": f"{dose:g}",
                "arm": "diag" if name.endswith("_diag") else "prod",
                "fired": r.get("fired_cooling_balance"),
                "theta_max": r.get("theta_max", ""),
                "theta_cum": f"{r['_theta_cum']:.4f}" if r.get("_theta_cum") else "",
                "fate": r.get("outcome", ""),
                "t_final": r.get("t_final", ""),
            }
        )
    out = HERE / "bench6_analysis.csv"
    with out.open("w", newline="") as fh:
        fh.write(
            "# bench6 head-to-head analysis (f_A vs f_mix) — combined bench5+bench6 arms; "
            "theta_cum = Theta over the blowout window (diag arms; clean L21b metric only "
            "for bench3/2/1, see FINDINGS 15h). Band-entry + uniformity printed by "
            "make_bench6_analysis.py. PROVISIONAL until sourced from HPC summaries.\n"
        )
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows -> {out}\n")

    print("Θ_cum dose–response per (bench, knob) — diag arms; band [0.90,0.99]:")
    entries = {}
    for knob in ("fA", "fmix"):
        for b in BENCHES:
            series = [
                (float(r["dose"]), _fnum(r["theta_cum"]))
                for r in out_rows
                if r["bench"] == b and r["knob"] == knob and r["arm"] == "diag" and r["theta_cum"]
            ]
            # fm arms share the fA dose-1 baseline (__none = unboosted)
            if knob == "fmix":
                base = [
                    (1.0, _fnum(r["theta_cum"]))
                    for r in out_rows
                    if r["bench"] == b
                    and r["knob"] == "fA"
                    and r["arm"] == "diag"
                    and float(r["dose"]) == 1
                    and r["theta_cum"]
                ]
                series = base + series
            if not series:
                continue
            e = band_entry(series)
            clean = b in CLEAN_BLOWOUT
            if clean:
                entries.setdefault(knob, {})[b] = e
            track = "  ".join(f"{d:g}:{t:.3f}" for d, t in sorted(series) if t is not None)
            tag = (f"entry≈{e:.3g}" if e else f">{max(d for d, _ in series):g}") + (
                "" if clean else "  [collapse-window, not clean L21b]"
            )
            print(f"  {b:18s} {knob:4s}  {track}   -> {tag}")

    print("\nBAND-ENTRY DOSE UNIFORMITY (clean-blowout benches only — the decision metric):")
    for knob, per in entries.items():
        vals = [v for v in per.values() if v]
        missing = [b for b, v in per.items() if not v]
        spread = (max(vals) / min(vals)) if len(vals) > 1 else None
        print(
            f"  {knob:4s}: entries {{"
            + ", ".join(
                f"{b.split('_')[0]}: {v:.3g}" if v else f"{b.split('_')[0]}: never"
                for b, v in sorted(per.items())
            )
            + "}"
            + (f"  spread(max/min)={spread:.2f}" if spread else "")
            + (f"  UNREACHED on {len(missing)} bench(es)" if missing else "")
        )
    print(
        "\nDecision read: smaller spread = better single-constant knob; 'never'/'UNREACHED' "
        "rows feed the 'no whole-band dose' branch of the Phase-6 tree "
        "(SOURCE_TERM_DESIGN §3 Phase 6)."
    )


if __name__ == "__main__":
    main()
