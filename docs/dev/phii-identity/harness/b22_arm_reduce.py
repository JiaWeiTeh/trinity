#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 22 stage 2 — reduce the B3MW001 three-arm local run against G22.8–G22.11.

Gates are pre-registered in PLAN.md (§Batch 22 STAGE 2 GATES) and were committed at
`1c46410c`, BEFORE any run started. This script only measures.

Reads each run's OWN `dictionary.jsonl` and `trinity.log` and nothing else — no reduced
ledger, no import of `compare_trajectories.py` — on the 2026-08-29 precedent, where an
independent re-read of the raw runs caught two of my own errors in the O1 arm report.

G22.8   the runs EXIST: >=1 snapshot and a recorded terminal state. Artifact check, not
        exit code — the local interpreter is 3.8.8 so `run_batch.py` is expected to raise
        on `Path.is_relative_to` AFTER the outputs are written.
G22.9   matched-`t` dR2 + phase sequences ENUMERATED, all three pairings, fate table.
G22.10  beta-delta unconverged-segment counts from `trinity.log`. Bar: arm == baseline.
G22.11  the floor: `P_drive` vs the ionised layer's own thermal pressure, per arm, and
        drive_K11/drive_O1 on matched rows.

    python docs/dev/phii-identity/harness/b22_arm_reduce.py --runs <dir> \\
        --out docs/dev/phii-identity/data/b22_b3mw001_arms.csv
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402

from _stamp import stamp  # noqa: E402

ARMS = ("base", "o1", "k11")
BENCH = REPO / ("docs/dev/transition/pdv-trigger/runs/params/bench5/"
                "bench3_m1e5_r5__none_diag.param")

# the beta-delta distress lines the PRB incident showed no gate was reading
BD_PAT = re.compile(r"beta-delta.*(no physical|unconverged)", re.I)
WARN_PAT = re.compile(r"\b(WARNING|ERROR|CRITICAL)\b")

FIELDS = ["arm", "i", "t", "phase", "R2", "v2", "Pb", "P_ram", "P_HII", "P_drive",
          "R_IF", "n_IF", "shell_n0", "shell_fIonisedDust", "P_layer_thermal",
          "drive_over_floor", "below_floor"]


def load(run_dir):
    """Snapshots from one run dir. Returns (rows, terminal, log_path)."""
    dj = next(run_dir.rglob("dictionary.jsonl"), None)
    if dj is None:
        return None, None, None
    rows = []
    with open(dj) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    meta = next(run_dir.rglob("metadata.json"), None)
    terminal = None
    if meta:
        try:
            m = json.load(open(meta))
            terminal = (m.get("stopping_reason") or m.get("fate")
                        or m.get("termination_reason") or m.get("simulation_end_reason"))
            if terminal is None:
                terminal = next((str(v) for k, v in m.items()
                                 if "stop" in k.lower() or "fate" in k.lower()), None)
        except (ValueError, OSError):
            terminal = None
    return rows, terminal, next(run_dir.rglob("trinity.log"), None)


def g(row, key, default=None):
    v = row.get(key, default)
    return v if isinstance(v, (int, float)) else default


def phases(rows):
    """Ordered, de-duplicated phase sequence — the F1HI lesson: enumerate, don't summarise."""
    out = []
    for r in rows:
        p = r.get("current_phase")
        if p and (not out or out[-1] != p):
            out.append(p)
    return out


def interp(rows, t):
    """R2 at matched simulation time by linear interpolation (C-5)."""
    ts = [g(r, "t_now") for r in rows]
    rs = [g(r, "R2") for r in rows]
    if t <= ts[0]:
        return rs[0]
    for i in range(1, len(ts)):
        if ts[i] >= t:
            f = (t - ts[i - 1]) / (ts[i] - ts[i - 1]) if ts[i] > ts[i - 1] else 0.0
            return rs[i - 1] + f * (rs[i] - rs[i - 1])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/dev/phii-identity/data/b22_b3mw001_arms.csv")
    args = ap.parse_args()

    p = read_param(str(BENCH))
    pref = (p["mu_convert"].value / p["mu_ion_shell"].value
            * p["k_B"].value * p["TShell_ion"].value)

    data = {}
    print("=" * 78)
    print("G22.8 — the runs EXIST (artifact check, not exit code)")
    print("=" * 78)
    for a in ARMS:
        rows, terminal, log = load(args.runs / a)
        if not rows:
            print(f"  {a:5} ⛔ NO dictionary.jsonl — run VOID, not a null")
            continue
        data[a] = (rows, terminal, log)
        print(f"  {a:5} {len(rows):4d} snapshots  t=[{g(rows[0],'t_now'):.4e}, "
              f"{g(rows[-1],'t_now'):.4e}]  R2_end={g(rows[-1],'R2'):.5f}  "
              f"terminal={terminal!r}")
        print(f"        phases: {' > '.join(phases(rows))}")
    if len(data) < 3:
        print("\n⛔ fewer than three arms produced output — comparison VOID")

    print("\n" + "=" * 78)
    print("G22.10 — beta-delta unconverged segments (bar: arm == baseline)")
    print("=" * 78)
    bd = {}
    for a, (rows, _, log) in data.items():
        if log is None:
            print(f"  {a:5} no trinity.log")
            continue
        txt = open(log, errors="replace").read().splitlines()
        warn = [ln for ln in txt if WARN_PAT.search(ln)]
        bd[a] = [ln for ln in warn if BD_PAT.search(ln)]
        print(f"  {a:5} {len(bd[a]):3d} beta-delta lines, {len(warn):3d} WARNING/ERROR total"
              f"  -> {'clean' if not bd[a] else 'NON-CONVERGENCE'}")
        for ln in bd[a][:4]:
            print(f"        {ln.strip()[:150]}")
    if "base" in bd:
        for a in ("o1", "k11"):
            if a in bd:
                print(f"  G22.10 {a} vs base: {len(bd[a])} vs {len(bd['base'])}"
                      f"  -> {'PASS' if len(bd[a]) == len(bd['base']) else 'FAIL'}")

    print("\n" + "=" * 78)
    print("G22.9 — matched-t dR2, phase sequences, fate table")
    print("=" * 78)
    for base_a, new_a in (("base", "o1"), ("base", "k11"), ("o1", "k11")):
        if base_a not in data or new_a not in data:
            continue
        b, n = data[base_a][0], data[new_a][0]
        t_hi = min(g(b[-1], "t_now"), g(n[-1], "t_now"))
        ts = [g(r, "t_now") for r in b if 0 < g(r, "t_now") <= t_hi]
        d = [(interp(n, t) / interp(b, t) - 1.0) * 100.0 for t in ts
             if interp(b, t) and interp(n, t)]
        worst = max(d, key=abs) if d else float("nan")
        same = phases(b) == phases(n)
        print(f"  {new_a:4} vs {base_a:4}: matched to t={t_hi:.4f} on {len(d):3d} points  "
              f"dR2_max={worst:+7.3f}%  dR2_end={d[-1] if d else float('nan'):+7.3f}%")
        print(f"        phases {'IDENTICAL' if same else 'DIFFER'}"
              f"{'' if same else f': {phases(b)} -> {phases(n)}'}"
              f"   fate {data[base_a][1]!r} -> {data[new_a][1]!r}")

    print("\n" + "=" * 78)
    print("G22.11 — THE FLOOR. drive vs the ionised layer's own thermal pressure")
    print("=" * 78)
    print("  floor := pref * n_IF  (the layer's own thermal pressure at the front, using")
    print("  the shell solve's OWN measured density — the guard the 2026-08-29 photo-limit")
    print("  decision made a CONDITION of accepting O1)")
    out_rows = []
    for a, (rows, _, _) in data.items():
        for i, r in enumerate(rows):
            n_IF = g(r, "n_IF")
            floor = pref * n_IF if n_IF and n_IF > 0 else None
            pdrive = g(r, "P_drive")
            out_rows.append(dict(
                arm=a, i=i, t=g(r, "t_now"), phase=r.get("current_phase"),
                R2=g(r, "R2"), v2=g(r, "v2"), Pb=g(r, "Pb"), P_ram=g(r, "P_ram"),
                P_HII=g(r, "P_HII"), P_drive=pdrive, R_IF=g(r, "R_IF"), n_IF=n_IF,
                shell_n0=g(r, "shell_n0"),
                shell_fIonisedDust=g(r, "shell_fIonisedDust"),
                P_layer_thermal=floor,
                drive_over_floor=(pdrive / floor) if (floor and pdrive) else None,
                below_floor=(pdrive < floor) if (floor and pdrive is not None) else None,
            ))
    for a in ARMS:
        sel = [r for r in out_rows if r["arm"] == a and r["drive_over_floor"] is not None]
        if not sel:
            continue
        vals = [r["drive_over_floor"] for r in sel]
        nb = sum(1 for r in sel if r["below_floor"])
        print(f"  {a:5} n={len(sel):4d}  drive/floor median {statistics.median(vals):9.4f}"
              f"  min {min(vals):9.4f}  rows BELOW the floor: {nb} ({100.0*nb/len(sel):.1f}%)")

    # K11 vs O1 on matched t — the D5 number this run exists to produce
    if "o1" in data and "k11" in data:
        o, k = data["o1"][0], data["k11"][0]
        t_hi = min(g(o[-1], "t_now"), g(k[-1], "t_now"))

        def drive_at(rows, t):
            ts = [g(r, "t_now") for r in rows]
            ds = [g(r, "P_drive") for r in rows]
            for i in range(1, len(ts)):
                if ts[i] >= t:
                    f = ((t - ts[i - 1]) / (ts[i] - ts[i - 1])) if ts[i] > ts[i - 1] else 0.0
                    return ds[i - 1] + f * (ds[i] - ds[i - 1])
            return None
        rat = [drive_at(k, g(r, "t_now")) / g(r, "P_drive")
               for r in o if 0 < g(r, "t_now") <= t_hi and g(r, "P_drive")
               and drive_at(k, g(r, "t_now"))]
        if rat:
            print(f"\n  drive_K11 / drive_O1 at matched t, n={len(rat)}: "
                  f"median {statistics.median(rat):.4f}  min {min(rat):.4f}  max {max(rat):.4f}")
            print("  [E, registered before running] stage 1 predicted K11 > O1 here "
                  "(photon-dominated),\n  i.e. the floor sits ABOVE O1 rather than under it. "
                  f"-> {'CONFIRMED' if statistics.median(rat) > 1.0 else 'REFUTED — my stage-1 reading is wrong'}")

    with open(args.out, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Batch 22 stage 2: B3MW001 (Lw x 0.01) three-arm LOCAL run, stop_t 1.5.\n")
        fh.write("# Gates G22.8-G22.11 registered in PLAN.md at 1c46410c BEFORE any run.\n")
        fh.write("# Read from each run's own dictionary.jsonl; no reduced ledger imported.\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {args.out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
