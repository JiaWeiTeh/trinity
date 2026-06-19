#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate the per-config F1-resample CSVs (bubble_resample_<config>.csv, each
one row per captured bubble-call x variant) into ONE master P0 table: a tidy CSV
(single source of truth) plus rendered markdown tables for the plan doc.

Reads every docs/dev/performance/data/bubble_resample_*.csv that exists (partial
sweeps are fine) and emits:
  - data/master_p0_table.csv   long format: config,phase,variant,<metrics>
  - stdout                     markdown tables (config x phase x variant)

Per-row schema (written by capture_replay_bubble.py):
  config, phase, call_index, variant, npts, bubble_dMdt, bubble_LTotal,
  bubble_T_r_Tb, bubble_mass, bubble_Tavg, R1, Pb, time_ms,
  rel_dMdt, rel_LTotal, rel_T_r_Tb, rel_mass, ok
The `baseline` variant (current 60k dense resample) is the timing + accuracy
reference: its rel_* are 0; mean speedup of a variant = mean(baseline time_ms for
the captured calls) / mean(variant time_ms) -- how much faster than the 60k
baseline.

REPRODUCE
    cd /home/user/trinity
    python docs/dev/performance/harness/aggregate_p0.py
"""
import csv
import glob
import math
import statistics
from pathlib import Path

TRINITY_ROOT = Path(__file__).resolve().parents[4]
DATA = TRINITY_ROOT / "docs" / "dev" / "performance" / "data"
MASTER = DATA / "master_p0_table.csv"

PHASE_ORDER = {"energy": 0, "implicit": 1, "transition": 2, "momentum": 3, "": 9}
BASELINE = "baseline"
# Method axis (matches the plan's verification matrix). baseline first; the rest
# in increasing coarseness. Unknown variants present in a CSV are appended so
# nothing is silently dropped.
VARIANT_ORDER = ["baseline", "M2000", "M1000", "M500", "M200", "Mnodes"]

REL_FIELDS = ["rel_dMdt", "rel_LTotal", "rel_T_r_Tb", "rel_mass"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _med(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.median(xs) if xs else math.nan


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.fmean(xs) if xs else math.nan


def _max(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return max(xs) if xs else math.nan


def _fmt_e(x):
    return f"{x:.2e}" if not (x is None or (isinstance(x, float) and math.isnan(x))) else ""


def _fmt_x(x):
    return f"{x:.2f}" if not (x is None or (isinstance(x, float) and math.isnan(x))) else ""


def load_rows():
    rows = []
    for path in sorted(glob.glob(str(DATA / "bubble_resample_*.csv"))):
        config = Path(path).stem.replace("bubble_resample_", "")
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                r["_config"] = config
                rows.append(r)
    return rows


def _variant_list(rows):
    """Known variants first (in plan order), then any extras, baseline always present."""
    seen = {r["variant"] for r in rows}
    ordered = [v for v in VARIANT_ORDER if v in seen]
    extras = sorted(v for v in seen if v not in VARIANT_ORDER)
    out = ordered + extras
    if BASELINE not in out:
        out = [BASELINE] + out
    return out


def aggregate(rows):
    """Return ({(config, phase, variant): metrics}, ordered variant list)."""
    variants = _variant_list(rows)
    cells = {}
    keys = sorted({(r["_config"], r.get("phase", "")) for r in rows},
                  key=lambda k: (k[0], PHASE_ORDER.get(k[1], 9)))
    for config, phase in keys:
        grp = [r for r in rows
               if r["_config"] == config and r.get("phase", "") == phase]
        # baseline timing per call_index -> denominator of the speedup ratio.
        base_rows = [r for r in grp if r["variant"] == BASELINE]
        base_time_by_call = {}
        for r in base_rows:
            base_time_by_call[r["call_index"]] = _f(r["time_ms"])
        mean_base_ms = _mean(list(base_time_by_call.values()))
        for v in variants:
            vr = [r for r in grp if r["variant"] == v]
            if not vr:
                cells[(config, phase, v)] = None
                continue
            n = len(vr)
            ok_count = sum(1 for r in vr if r.get("ok", "") in ("1", "True", "true"))
            mean_var_ms = _mean([_f(r["time_ms"]) for r in vr])
            # mean speedup = mean(baseline time for these calls) / mean(variant time).
            # baseline-vs-itself is 1.0 by construction.
            speedup = (mean_base_ms / mean_var_ms
                       if mean_var_ms and not math.isnan(mean_var_ms)
                       and mean_base_ms and not math.isnan(mean_base_ms)
                       and mean_var_ms != 0 else math.nan)
            cell = {"config": config, "phase": phase, "variant": v,
                    "n": n, "ok_count": ok_count,
                    "mean_speedup": speedup,
                    "mean_base_ms": mean_base_ms,
                    "mean_var_ms": mean_var_ms}
            for fld in REL_FIELDS:
                vals = [abs(_f(r.get(fld, ""))) for r in vr]
                cell["worst_" + fld] = _max(vals)
                cell["median_" + fld] = _med(vals)
            cells[(config, phase, v)] = cell
    return cells, variants


def write_master_csv(cells, variants):
    cols = ["config", "phase", "variant", "n", "ok_count", "mean_speedup",
            "mean_base_ms", "mean_var_ms"]
    for fld in REL_FIELDS:
        cols += ["worst_" + fld, "median_" + fld]
    keys = sorted(cells.keys(),
                  key=lambda k: (k[0], PHASE_ORDER.get(k[1], 9),
                                 variants.index(k[2]) if k[2] in variants else 99))
    with open(MASTER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k in keys:
            c = cells[k]
            if c is None:
                continue
            row = {"config": c["config"], "phase": c["phase"],
                   "variant": c["variant"], "n": c["n"],
                   "ok_count": c["ok_count"],
                   "mean_speedup": (round(c["mean_speedup"], 3)
                                    if not math.isnan(c["mean_speedup"]) else ""),
                   "mean_base_ms": (round(c["mean_base_ms"], 4)
                                    if not math.isnan(c["mean_base_ms"]) else ""),
                   "mean_var_ms": (round(c["mean_var_ms"], 4)
                                   if not math.isnan(c["mean_var_ms"]) else "")}
            for fld in REL_FIELDS:
                w_ = c["worst_" + fld]
                m_ = c["median_" + fld]
                row["worst_" + fld] = (f"{w_:.3e}" if not math.isnan(w_) else "")
                row["median_" + fld] = (f"{m_:.3e}" if not math.isnan(m_) else "")
            w.writerow(row)
    print(f"# wrote {MASTER}")


def render_full(cells, variants):
    """One row per config x phase x variant. Worst rel-diffs + mean speedup."""
    out = []
    out.append("| config | phase | variant | n | ok | speedup | "
               "worst rel_dMdt | worst rel_LTotal | worst rel_T_r_Tb | "
               "worst rel_mass |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    keys = sorted({(k[0], k[1]) for k in cells},
                  key=lambda k: (k[0], PHASE_ORDER.get(k[1], 9)))
    for config, phase in keys:
        for v in variants:
            c = cells.get((config, phase, v))
            if c is None:
                continue
            sp = "1.00 (ref)" if v == BASELINE else (_fmt_x(c["mean_speedup"]) + "x"
                                                     if not math.isnan(c["mean_speedup"])
                                                     else "n/a")
            out.append(
                f"| {config} | {phase} | {v} | {c['n']} | "
                f"{c['ok_count']}/{c['n']} | {sp} | "
                f"{_fmt_e(c['worst_rel_dMdt'])} | {_fmt_e(c['worst_rel_LTotal'])} | "
                f"{_fmt_e(c['worst_rel_T_r_Tb'])} | {_fmt_e(c['worst_rel_mass'])} |")
    return "\n".join(out)


def render_speedup(cells, variants):
    """Speedup-first digest: config x phase rows, one speedup column per variant."""
    vcols = [v for v in variants if v != BASELINE]
    out = []
    out.append("| config | phase | baseline ms | " +
               " | ".join(f"{v} speed" for v in vcols) + " |")
    out.append("|---|---|---|" + "|".join("---" for _ in vcols) + "|")
    keys = sorted({(k[0], k[1]) for k in cells},
                  key=lambda k: (k[0], PHASE_ORDER.get(k[1], 9)))
    for config, phase in keys:
        base = cells.get((config, phase, BASELINE))
        base_ms = (f"{base['mean_base_ms']:.3f}"
                   if base and not math.isnan(base["mean_base_ms"]) else "")
        cellvals = []
        for v in vcols:
            c = cells.get((config, phase, v))
            if c is None or math.isnan(c["mean_speedup"]):
                cellvals.append("n/a")
            else:
                cellvals.append(f"{c['mean_speedup']:.2f}x")
        out.append(f"| {config} | {phase} | {base_ms} | " +
                   " | ".join(cellvals) + " |")
    return "\n".join(out)


def render_worst_dMdt(cells, variants):
    """Accuracy-first digest (the G2 metric): worst rel_dMdt per variant."""
    vcols = [v for v in variants if v != BASELINE]
    out = []
    out.append("| config | phase | " +
               " | ".join(f"{v} worst dMdt" for v in vcols) + " |")
    out.append("|---|---|" + "|".join("---" for _ in vcols) + "|")
    keys = sorted({(k[0], k[1]) for k in cells},
                  key=lambda k: (k[0], PHASE_ORDER.get(k[1], 9)))
    for config, phase in keys:
        cellvals = []
        for v in vcols:
            c = cells.get((config, phase, v))
            if c is None:
                cellvals.append("n/a")
            else:
                cellvals.append(_fmt_e(c["worst_rel_dMdt"]) or "n/a")
        out.append(f"| {config} | {phase} | " + " | ".join(cellvals) + " |")
    return "\n".join(out)


def main():
    rows = load_rows()
    if not rows:
        print("# no resample CSVs yet (bubble_resample_*.csv)")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    cells, variants = aggregate(rows)
    write_master_csv(cells, variants)
    print("\n### speedup (config x phase, per variant)\n")
    print(render_speedup(cells, variants))
    print("\n### worst rel_dMdt (the G2 accuracy gate)\n")
    print(render_worst_dMdt(cells, variants))
    print("\n### full (config x phase x variant)\n")
    print(render_full(cells, variants))


if __name__ == "__main__":
    main()
