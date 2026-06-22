#!/usr/bin/env python3
"""Drive ONE (config, variant) sim for the H3 Eb-floor experiment and append a
CSV row + (optionally) a per-snapshot trajectory CSV. Production untouched
(h3_variants.py monkeypatches module attributes only).

Run one cell per process (isolates crashes + leaked global sim/logging state):

    OMP_NUM_THREADS=1 timeout 600 python h3_run_variant.py \
        --variant EBFLOOR --param <cfg>.param --stop_t 0.05 \
        --csv h3_eval.csv --traj h3_traj_<cfg>_<variant>.csv

Records crash vs clean end-state, the final (t,R2,v2,Eb), the SimulationEndReason,
the deepest phase reached, whether the Eb floor activated (drive/state hit counts +
min Eb the RHS saw), and whether the cooling-balance trigger (Lgain-Lloss)/Lgain<0.05
ever fired. Trajectory pulled from the run's dictionary.jsonl (the production
snapshot stream), so nothing is recomputed.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `import h3_variants` works regardless of cwd

TRIGGER = 0.05  # cooling-balance transition trigger: (Lgain-Lloss)/Lgain < TRIGGER


def _val(params, key, default=None):
    try:
        p = params[key]
        return p.value if hasattr(p, "value") else p
    except Exception:
        return default


def _reached_phase(rows):
    order = {"energy": 1, "implicit": 2, "transition": 3, "momentum": 4}
    deepest, drank = "", 0
    for r in rows:
        ph = r.get("current_phase", "")
        # map 'energy' (1a) and 'implicit' (1b) onto the report's phase ladder
        rk = order.get(ph, 0)
        if rk > drank:
            drank, deepest = rk, {1: "1a", 2: "1b", 3: "1c", 4: "momentum"}[rk]
    return deepest


def _load_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda d: d.get("t_now", 0.0))
    return rows


def _ratio(r):
    """(Lgain-Lloss)/Lgain cooling-balance ratio, or None if not computable."""
    lg, ll = r.get("bubble_Lgain"), r.get("bubble_Lloss")
    try:
        if lg is None or ll is None or lg != lg or ll != ll or lg <= 0:
            return None
        return (lg - ll) / lg
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--param", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--traj", default=None, help="write per-snapshot trajectory CSV here")
    ap.add_argument("--stop_t", type=float, default=None)
    ap.add_argument("--floor", type=float, default=None, help="Eb floor [au]")
    ap.add_argument("--out", default=None, help="override path2output")
    args = ap.parse_args()

    import h3_variants
    h3_variants.apply(args.variant, floor=args.floor)

    from trinity._input import read_param
    from trinity import main as trinity_main
    from trinity._functions.logging_setup import setup_logging

    params = read_param.read_param(args.param)
    cfg = os.path.splitext(os.path.basename(args.param))[0]
    out_dir = args.out or _val(params, "path2output") or f"/tmp/h3/{cfg}_{args.variant}"
    params["path2output"].value = out_dir
    if args.stop_t is not None and "stop_t" in params:
        params["stop_t"].value = args.stop_t

    setup_logging(
        log_level=_val(params, "log_level", "INFO"),
        console_output=bool(_val(params, "log_console", False)),
        file_output=bool(_val(params, "log_file", True)),
        log_file_path=out_dir,
        log_file_name="trinity.log",
        use_colors=False,
    )

    # capture the run's initial bubble energy E0 for floor-sensitivity context
    E0 = None

    crashed = False
    crash_excpt = ""
    crash_phase = ""
    t0 = time.time()
    try:
        trinity_main.start_expansion(params)
    except SystemExit as e:
        crash_excpt = f"SystemExit:{e}"
    except BaseException as e:  # noqa: BLE001 -- catching the crash is the point
        crashed = True
        crash_excpt = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        for ph, needle in (("1a", "run_energy_phase"), ("1b", "run_energy_implicit"),
                            ("1c", "run_transition")):
            if needle in tb:
                crash_phase = ph
    runtime = time.time() - t0

    # read the production snapshot stream for trajectory + trigger + reached-phase
    rows = []
    hits = list(Path(out_dir).rglob("dictionary.jsonl"))
    if hits:
        jsonl = str(max(hits, key=lambda p: p.stat().st_mtime))
        try:
            rows = _load_jsonl(jsonl)
        except Exception:
            rows = []

    # cooling-balance trigger ever fire?
    ratios = [(_ratio(r)) for r in rows]
    ratios = [x for x in ratios if x is not None]
    ratio_min = min(ratios) if ratios else None
    trigger_fired = any(x < TRIGGER for x in ratios) if ratios else False

    if rows:
        E0 = rows[0].get("Eb")

    act = h3_variants.ACTIVATED
    floor_used = h3_variants.FLOOR
    floor_activated = (act["drive"] + act["state"]) > 0

    row = {
        "config": cfg,
        "variant": args.variant,
        "crashed": crashed,
        "crash_phase": crash_phase,
        "crash_excpt": crash_excpt[:140],
        "end_reason": _val(params, "SimulationEndReason", ""),
        "end_code": _val(params, "SimulationEndCode", ""),
        "reached_phase": _reached_phase(rows),
        "final_t": _val(params, "t_now"),
        "final_R2": _val(params, "R2"),
        "final_v2": _val(params, "v2"),
        "final_Eb": _val(params, "Eb"),
        "E0": E0,
        "floor": floor_used,
        "floor_activated": floor_activated,
        "act_drive": act["drive"],
        "act_state": act["state"],
        "min_Eb_seen": (None if act["min_Eb_seen"] == float("inf") else act["min_Eb_seen"]),
        "trigger_fired": trigger_fired,
        "ratio_min": ratio_min,
        "n_rows": len(rows),
        "runtime_s": round(runtime, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    if args.traj and rows:
        cols = ["t_now", "phase", "R2", "v2", "Eb", "Pb", "R1", "T0",
                "bubble_Lgain", "bubble_Lloss", "ratio", "rCloud"]
        os.makedirs(os.path.dirname(os.path.abspath(args.traj)), exist_ok=True)
        with open(args.traj, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({
                    "t_now": r.get("t_now"), "phase": r.get("current_phase"),
                    "R2": r.get("R2"), "v2": r.get("v2"), "Eb": r.get("Eb"),
                    "Pb": r.get("Pb"), "R1": r.get("R1"), "T0": r.get("T0"),
                    "bubble_Lgain": r.get("bubble_Lgain"),
                    "bubble_Lloss": r.get("bubble_Lloss"),
                    "ratio": _ratio(r), "rCloud": r.get("rCloud"),
                })

    print(f"[{args.variant} {cfg}] crashed={crashed} reached={row['reached_phase']} "
          f"reason={str(row['end_reason'])[:50]!r} floor_act={floor_activated} "
          f"(drive={act['drive']},state={act['state']}) trig={trigger_fired} "
          f"final_Eb={row['final_Eb']} t={row['runtime_s']}s")


if __name__ == "__main__":
    main()
