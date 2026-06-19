#!/usr/bin/env python3
"""Aggregate the END-TO-END science-gate matrix into eval_endtoend.csv.

Reads the per-run metrics (the final ENDTOEND_METRICS json line of each
/tmp/eteo_<config>_<idea>.log) and, for each variant, compares its
outputs/<config>__<idea>/dictionary.jsonl against the config's baseline run via
compare_endtoend (final-row + trajectory max rel diff over the key science cols).

Emits docs/dev/shell-solver/data/eval_endtoend.csv with columns:
  config, idea, total_run_s, total_run_s_baseline, overflow_warns_total,
  endtoend_final_maxrel, endtoend_traj_maxrel, n_timesteps,
  n_timesteps_baseline, notes

Usage:  python docs/dev/shell-solver/harness/aggregate_endtoend.py
"""
import os
import sys
import csv
import json
from pathlib import Path

TRINITY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "harness"))
import compare_endtoend as cmp  # noqa: E402

DATA_DIR = TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "data"
OUT_CSV = DATA_DIR / "eval_endtoend.csv"

CONFIGS = ["simple_cluster", "probe_typical_hybr"]
IDEAS = ["baseline", "phiguard", "clip", "cgs"]


def _read_metrics(config, idea):
    log = Path(f"/tmp/eteo_{config}_{idea}.log")
    if not log.exists():
        return None
    last = None
    for line in log.read_text().splitlines():
        if line.startswith("ENDTOEND_METRICS "):
            last = line[len("ENDTOEND_METRICS "):]
    return json.loads(last) if last else None


def _compare(base_jsonl, var_jsonl):
    base = cmp._load(base_jsonl)
    var = cmp._load(var_jsonl)
    n = min(len(base), len(var))
    final_per, traj_per = {}, {}
    bf, vf = base[-1], var[-1]
    for c in cmp.SCI_COLS:
        if c not in bf or c not in vf:
            continue
        final_per[c] = cmp._relemax([bf[c]], [vf[c]])
        traj_per[c] = cmp._relemax([r.get(c) for r in base[:n]],
                                   [r.get(c) for r in var[:n]])
    return (max(final_per.values()) if final_per else None,
            max(traj_per.values()) if traj_per else None,
            max(final_per, key=final_per.get) if final_per else None,
            max(traj_per, key=traj_per.get) if traj_per else None,
            len(base), len(var))


def main():
    rows = []
    for config in CONFIGS:
        base_m = _read_metrics(config, "baseline")
        base_jsonl = str(TRINITY_ROOT / "outputs" / f"{config}__baseline" / "dictionary.jsonl")
        base_wall = base_m["wall_s"] if base_m else None
        base_nts = base_m["n_timesteps"] if base_m else None
        base_ovf = base_m["overflow_warns"] if base_m else None
        for idea in IDEAS:
            m = _read_metrics(config, idea)
            if m is None:
                rows.append(dict(config=config, idea=idea, total_run_s="",
                                 total_run_s_baseline=base_wall or "",
                                 overflow_warns_total="", endtoend_final_maxrel="",
                                 endtoend_traj_maxrel="", n_timesteps="",
                                 n_timesteps_baseline=base_nts or "",
                                 notes="MISSING run log"))
                continue
            if idea == "baseline":
                f_rel = t_rel = 0.0
                fcol = tcol = "self"
                nb, nv = m["n_timesteps"], m["n_timesteps"]
                note = (f"baseline; flood={base_ovf} overflow-warns; "
                        f"final_t={m.get('final_t_now'):.6g}")
            else:
                var_jsonl = str(TRINITY_ROOT / "outputs" / f"{config}__{idea}" / "dictionary.jsonl")
                if not Path(var_jsonl).exists() or not Path(base_jsonl).exists():
                    f_rel = t_rel = None
                    fcol = tcol = ""
                    nb, nv = (base_nts or 0), m["n_timesteps"]
                    note = "jsonl missing for compare"
                else:
                    f_rel, t_rel, fcol, tcol, nb, nv = _compare(base_jsonl, var_jsonl)
                    flood_delta = (f"flood {base_ovf}->{m['overflow_warns']}"
                                   if base_ovf is not None else "")
                    speed = (f"{base_wall/m['wall_s']:.2f}x"
                             if base_wall and m["wall_s"] else "n/a")
                    note = (f"{flood_delta}; speed {speed} vs base; "
                            f"final_worst={fcol}; traj_worst={tcol}; "
                            f"nrows {nb}vs{nv}")
            rows.append(dict(
                config=config, idea=idea,
                total_run_s=m["wall_s"],
                total_run_s_baseline=base_wall if base_wall is not None else "",
                overflow_warns_total=m["overflow_warns"],
                endtoend_final_maxrel=("" if f_rel is None else f"{f_rel:.3e}"),
                endtoend_traj_maxrel=("" if t_rel is None else f"{t_rel:.3e}"),
                n_timesteps=m["n_timesteps"],
                n_timesteps_baseline=base_nts if base_nts is not None else "",
                notes=note,
            ))

    cols = ["config", "idea", "total_run_s", "total_run_s_baseline",
            "overflow_warns_total", "endtoend_final_maxrel",
            "endtoend_traj_maxrel", "n_timesteps", "n_timesteps_baseline", "notes"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")
    # Echo a compact verdict table.
    print(f"\n{'config':20s} {'idea':9s} {'wall_s':>8s} {'ovf':>5s} "
          f"{'final_rel':>11s} {'traj_rel':>11s} {'n_ts':>5s}")
    for r in rows:
        print(f"{r['config']:20s} {r['idea']:9s} {str(r['total_run_s']):>8s} "
              f"{str(r['overflow_warns_total']):>5s} "
              f"{str(r['endtoend_final_maxrel']):>11s} "
              f"{str(r['endtoend_traj_maxrel']):>11s} {str(r['n_timesteps']):>5s}")


if __name__ == "__main__":
    main()
