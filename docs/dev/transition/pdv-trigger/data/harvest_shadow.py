#!/usr/bin/env python3
"""Harvest the Stage-A theta_elbadry shadow (and baseline) runs into one fate table.

Reads each run dir under OUT_BASE (default outputs/shadow_te), pulls from dictionary.jsonl the
first momentum-phase time (= transition fired), the final (t, R2, v2), and the SimulationEndCode;
folds in the per-run theta range from theta_elbadry_diag.json when present. Emits a CSV to
docs/dev/transition/pdv-trigger/data/shadow_te_fate.csv (override with OUT_CSV).

Usage:
  OUT_BASE=outputs/shadow_te   OUT_CSV=.../data/shadow_te_fate.csv     python harvest_shadow.py
  OUT_BASE=outputs/baseline_te OUT_CSV=.../data/baseline_te_fate.csv   python harvest_shadow.py
"""
import csv
import json
import os
import sys

OUT_BASE = os.environ.get("OUT_BASE", "outputs/shadow_te")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.environ.get("OUT_CSV", os.path.join(HERE, "shadow_te_fate.csv"))

ENDCODE = {0: "shell_dissolved", 1: "stopping_time", 2: "large_radius", 3: "rcloud_boundary",
           4: "SHELL_COLLAPSED", 51: "energy_collapsed", 50: "velocity_runaway", 99: "unknown"}
AU2CGS = 1.0 / 2.937998946096347e+55  # ndens AU -> cm^-3


def f(rec, *names):
    for n in names:
        if n in rec:
            return rec[n]
    return None


def harvest(rundir):
    p = os.path.join(rundir, "dictionary.jsonl")
    if not os.path.exists(p):
        return None
    first = last = None
    fire_t = None
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if first is None:
                first = rec
            last = rec
            ph = str(f(rec, "current_phase", "phase", "SimulationPhase") or "")
            if fire_t is None and "momentum" in ph.lower():
                fire_t = f(rec, "t", "t_now", "time")
    end = f(last, "SimulationEndCode", "end_code", "sim_end_code")
    # metadata's termination.exit_code is authoritative for clean stop-time exits
    # (dictionary.jsonl carries no endcode on STOPPING_TIME). Empty termination => crashed early.
    crashed = False
    mp0 = os.path.join(rundir, "metadata.json")
    if os.path.exists(mp0):
        term = json.load(open(mp0)).get("termination", {})
        if "exit_code" in term:
            end = term["exit_code"]
        elif end is None:
            crashed = True
    row = {
        "config": os.path.basename(rundir),
        "fire_t_Myr": fire_t,
        "end_t_Myr": f(last, "t", "t_now", "time"),
        "end_R2_pc": f(last, "R2", "r2"),
        "end_v2_kms": f(last, "v2"),
        "endcode": end,
        "fate": "CRASHED_EARLY" if crashed else ENDCODE.get(end, "running/none"),
    }
    # cloud params from metadata
    mp = os.path.join(rundir, "metadata.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        nc = m.get("nCore")
        row["nCore_cm3"] = round(nc * AU2CGS, 3) if isinstance(nc, (int, float)) else nc
        row["mCloud_Msun"] = m.get("mCloud")
        row["sfe"] = m.get("sfe")
    # theta from diag
    dp = os.path.join(rundir, "theta_elbadry_diag.json")
    if os.path.exists(dp):
        d = json.load(open(dp))
        row["theta_min"] = round(d["theta_min"], 4)
        row["theta_max"] = round(d["theta_max_seen"], 4)
        row["resolved_wins"] = d["n_resolved_wins"]
        row["n_calls"] = d["n_calls"]
    return row


def main():
    rows = []
    for name in sorted(os.listdir(OUT_BASE)):
        d = os.path.join(OUT_BASE, name)
        if os.path.isdir(d):
            r = harvest(d)
            if r:
                rows.append(r)
    cols = ["config", "nCore_cm3", "mCloud_Msun", "sfe", "theta_min", "theta_max",
            "fire_t_Myr", "end_t_Myr", "end_R2_pc", "end_v2_kms", "endcode", "fate",
            "resolved_wins", "n_calls"]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    for r in rows:
        print(f"{r['config']:24s} n={r.get('nCore_cm3','?'):>9} theta={r.get('theta_max','-')!s:>6} "
              f"fire={r['fire_t_Myr']!s:>10.10} end_t={r['end_t_Myr']!s:>8.8} "
              f"R2={r['end_R2_pc']!s:>8.8} v2={r['end_v2_kms']!s:>8.8} -> {r['fate']}")
    print(f"\nwrote {OUT_CSV}  ({len(rows)} runs)")


if __name__ == "__main__":
    sys.exit(main())
