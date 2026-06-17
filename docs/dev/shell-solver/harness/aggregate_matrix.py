#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate the per-config matrix CSVs (replay_variants_matrix_<config>.csv, each
config x phase x variant) into ONE master table: a tidy CSV (single source of
truth) plus a rendered markdown headline table for the plan doc.

Reads every docs/dev/shell-solver/data/replay_variants_matrix_*.csv that exists
(partial sweeps are fine) and emits:
  - data/master_table.csv          long format: config,phase,variant,<metrics>
  - stdout                          markdown headline table (config x phase rows)

REPRODUCE
    cd /home/user/trinity
    python docs/dev/shell-solver/harness/aggregate_matrix.py
"""
import csv
import glob
import math
import statistics
from pathlib import Path

TRINITY_ROOT = Path(__file__).resolve().parents[4]
DATA = TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "data"
MASTER = DATA / "master_table.csv"

PHASE_ORDER = {"energy": 0, "implicit": 1, "transition": 2, "momentum": 3, "": 9}
VARIANTS = ["V_lsoda_teval", "V_lsoda_event", "V_lsoda_dense",
            "V_radau_teval", "V_bdf_teval", "V_odeint_hi"]

# variant -> (solver, stopping, human label)
VARIANT_META = {
    "V_lsoda_teval": ("LSODA", "normal (full grid)", "LSODA + t_eval (drop-in)"),
    "V_lsoda_event": ("LSODA", "SMART: stop at ion. front", "LSODA + front event"),
    "V_lsoda_dense": ("LSODA", "normal (dense interp)", "LSODA + dense_output"),
    "V_radau_teval": ("Radau", "normal", "Radau"),
    "V_bdf_teval":   ("BDF", "normal", "BDF"),
    "V_odeint_hi":   ("LSODA (odeint)", "normal, mxstep=50k", "odeint mxstep=50k"),
}
DEGENERATE = {"sfe0.3", "sfe0.6"}  # simple_cluster overflow regime


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _med(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.median(xs) if xs else math.nan


def load_rows():
    rows = []
    for path in sorted(glob.glob(str(DATA / "replay_variants_matrix_*.csv"))):
        config = Path(path).stem.replace("replay_variants_matrix_", "")
        for r in csv.DictReader(open(path)):
            r["_config"] = config
            rows.append(r)
    return rows


def aggregate(rows):
    """Return {(config, phase): {context + per-variant metrics}}."""
    cells = {}
    # group by config, phase
    keys = sorted({(r["_config"], r.get("phase", "")) for r in rows},
                  key=lambda k: (k[0], PHASE_ORDER.get(k[1], 9)))
    for config, phase in keys:
        grp = [r for r in rows if r["_config"] == config and r.get("phase", "") == phase]
        # per-call context (use V_lsoda_teval rows; one per captured call)
        base = [r for r in grp if r["variant"] == "V_lsoda_teval"]
        ncalls = len(base)
        if ncalls == 0:
            continue
        nion = sum(1 for r in base if r["is_ionised"] == "1")
        masslim = sum(1 for r in base if r.get("idx_phi", "0") == "-1")
        odeint_ms = _med([_f(r["baseline_odeint_time_ms"]) for r in base])
        excess = sum(1 for r in base if _f(r.get("baseline_odeint_py_warns", "0")) > 0)
        cell = {"config": config, "phase": phase, "n_calls": ncalls,
                "n_ion": nion, "n_neu": ncalls - nion,
                "mass_limited_frac": round(masslim / ncalls, 3),
                "odeint_ms_med": round(odeint_ms, 3) if not math.isnan(odeint_ms) else "",
                "excesswork_frac": round(excess / ncalls, 3), "variants": {}}
        for v in VARIANTS:
            vr = [r for r in grp if r["variant"] == v]
            ok = sum(1 for r in vr if r["success"] == "1")
            sp = _med([_f(r["speedup_vs_odeint"]) for r in vr if r["success"] == "1"])
            rel = [_f(r["max_rel_diff_n"]) for r in vr
                   if r["success"] == "1" and not math.isnan(_f(r["max_rel_diff_n"]))]
            worst = max(rel) if rel else math.nan
            ev = sum(int(_f(r["event_fired"])) for r in vr if not math.isnan(_f(r["event_fired"])))
            cell["variants"][v] = {
                "ok": ok, "n": len(vr),
                "speedup_med": round(sp, 2) if not math.isnan(sp) else "",
                "worst_rel_n": f"{worst:.2e}" if not math.isnan(worst) else "",
                "event_fired": ev}
        cells[(config, phase)] = cell
    return cells


def write_master_csv(cells):
    cols = ["config", "phase", "n_calls", "n_ion", "n_neu", "odeint_ms_med",
            "excesswork_frac", "mass_limited_frac", "variant", "ok", "n",
            "speedup_med", "worst_rel_n", "event_fired"]
    with open(MASTER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for (config, phase), c in cells.items():
            for v in VARIANTS:
                vm = c["variants"][v]
                w.writerow({"config": config, "phase": phase, "n_calls": c["n_calls"],
                            "n_ion": c["n_ion"], "n_neu": c["n_neu"],
                            "odeint_ms_med": c["odeint_ms_med"],
                            "excesswork_frac": c["excesswork_frac"],
                            "mass_limited_frac": c["mass_limited_frac"],
                            "variant": v, "ok": vm["ok"], "n": vm["n"],
                            "speedup_med": vm["speedup_med"],
                            "worst_rel_n": vm["worst_rel_n"],
                            "event_fired": vm["event_fired"]})
    print(f"# wrote {MASTER}")


def render_markdown(cells):
    """Headline table: config x phase rows; key columns."""
    out = []
    out.append("| config | phase | calls (ion/neu) | odeint ms | excess-work | "
               "mass-lim | t_eval ok·speed·rel_n | event ok·speed·rel_n | "
               "Radau/BDF ok |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for (config, phase), c in cells.items():
        te = c["variants"]["V_lsoda_teval"]
        ev = c["variants"]["V_lsoda_event"]
        ra = c["variants"]["V_radau_teval"]
        bd = c["variants"]["V_bdf_teval"]
        out.append(
            f"| {config} | {phase} | {c['n_calls']} ({c['n_ion']}/{c['n_neu']}) | "
            f"{c['odeint_ms_med']} | {int(c['excesswork_frac']*100)}% | "
            f"{int(c['mass_limited_frac']*100)}% | "
            f"{te['ok']}/{te['n']}·{te['speedup_med']}x·{te['worst_rel_n']} | "
            f"{ev['ok']}/{ev['n']}·{ev['speedup_med']}x·{ev['worst_rel_n']} | "
            f"{ra['ok']}/{ra['n']} · {bd['ok']}/{bd['n']} |")
    return "\n".join(out)


def render_by_variant(rows):
    """Variant-first digest: each variant a row, split by regime class
    (degenerate simple_cluster vs the 4 realistic configs), phases pooled."""
    out = []
    out.append("| variant | solver | stopping | degenerate (sfe0.3/0.6): "
               "ok·speed·rel_n | realistic (typ/steep/dense/mock): ok·speed·rel_n |")
    out.append("|---|---|---|---|---|")
    for v in VARIANTS:
        solver, stop, _ = VARIANT_META[v]
        cells_txt = []
        for cls in (DEGENERATE, None):  # None = realistic complement
            vr = [r for r in rows if r["variant"] == v
                  and ((r["_config"] in DEGENERATE) if cls else (r["_config"] not in DEGENERATE))]
            ok = sum(1 for r in vr if r["success"] == "1")
            sp = _med([_f(r["speedup_vs_odeint"]) for r in vr if r["success"] == "1"])
            rel = [_f(r["max_rel_diff_n"]) for r in vr
                   if r["success"] == "1" and not math.isnan(_f(r["max_rel_diff_n"]))]
            worst = f"{max(rel):.1e}" if rel else "n/a"
            sp_s = f"{sp:.2f}x" if not math.isnan(sp) else "n/a"
            cells_txt.append(f"{ok}/{len(vr)}·{sp_s}·{worst}")
        out.append(f"| {VARIANT_META[v][2]} | {solver} | {stop} | "
                   f"{cells_txt[0]} | {cells_txt[1]} |")
    return "\n".join(out)


def render_long(cells):
    """Comprehensive table: one row per config x phase x variant, keeping the
    per-cell context columns AND an explicit CURRENT-DEFAULT row (the baseline
    `odeint` the production code uses today = LSODA over the full grid, the 1.00x
    reference). Nothing dropped vs the context view."""
    out = []
    out.append("| config | phase | ion/neu | odeint ms | excess-work | mass-lim | "
               "variant (solver · stopping) | ok | speed | worst rel_n | ev_fired |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (config, phase), c in cells.items():
        ctx = (f"{c['n_ion']}/{c['n_neu']} | {c['odeint_ms_med']} | "
               f"{int(c['excesswork_frac'] * 100)}% | "
               f"{int(c['mass_limited_frac'] * 100)}%")
        # CURRENT PRODUCTION DEFAULT: baseline odeint (LSODA, full grid). It is
        # the reference, so speed = 1.00x and rel_n = 0 by definition.
        out.append(f"| {config} | {phase} | {ctx} | "
                   f"**odeint — CURRENT DEFAULT** (LSODA · full grid) | "
                   f"{c['n_calls']}/{c['n_calls']} | 1.00× (ref) | 0 (ref) | 0 |")
        for v in VARIANTS:
            vm = c["variants"][v]
            solver, stop, label = VARIANT_META[v]
            sp = f"{vm['speedup_med']}×" if vm["speedup_med"] != "" else "n/a"
            out.append(f"| {config} | {phase} | {ctx} | "
                       f"{label} ({solver} · {stop}) | {vm['ok']}/{vm['n']} | {sp} | "
                       f"{vm['worst_rel_n'] or 'n/a'} | {vm['event_fired']} |")
    return "\n".join(out)


def main():
    rows = load_rows()
    if not rows:
        print("# no matrix CSVs yet (replay_variants_matrix_*.csv)")
        return
    cells = aggregate(rows)
    write_master_csv(cells)
    print("\n### by variant (solver + stopping, regime-pooled)\n")
    print(render_by_variant(rows))
    print("\n### context (config x phase)\n")
    print(render_markdown(cells))
    print("\n### full (config x phase x variant)\n")
    print(render_long(cells))


if __name__ == "__main__":
    main()
