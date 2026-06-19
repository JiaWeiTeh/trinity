#!/usr/bin/env python3
"""Drive ONE (variant, config) sim cell for the failed-large-clouds matrix and
append a CSV row. Production code is untouched (variants.py monkeypatches only).

Run one cell per process (isolates crashes + global sim/logging state):

    python run_variant.py --variant V1 --param params/fail.param \
        --stop_t 0.05 --csv ../data/eval_V1.csv

Wrap with `timeout` in the caller. Records crash vs clean end-state, the final
(R2,v2,Eb,t), the recorded SimulationEndReason, and the deepest phase reached
(scanned from the run's trinity.log).
"""
import argparse
import csv
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `import variants` works regardless of cwd


def _val(params, key, default=None):
    try:
        p = params[key]
        return p.value if hasattr(p, "value") else p
    except Exception:
        return default


def _reached_phase(logpath):
    """Deepest phase marker found in the run log (1a < 1b < 1c < momentum)."""
    order = [
        ("PHASE 1a", "1a"), ("PHASE 1b", "1b"), ("PHASE 1c", "1c"),
        ("momentum", "momentum"), ("MOMENTUM", "momentum"),
    ]
    deepest = ""
    rank = {"1a": 1, "1b": 2, "1c": 3, "momentum": 4}
    try:
        with open(logpath, errors="ignore") as f:
            text = f.read()
        for needle, name in order:
            if needle in text and rank.get(name, 0) > rank.get(deepest, 0):
                deepest = name
    except Exception:
        pass
    return deepest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--param", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--stop_t", type=float, default=None,
                    help="override stop_t [Myr] to bound runtime")
    ap.add_argument("--out", default=None, help="override path2output")
    args = ap.parse_args()

    import variants
    variants.apply(args.variant)

    from trinity._input import read_param
    from trinity import main as trinity_main

    params = read_param.read_param(args.param)
    out_dir = args.out or _val(params, "path2output")
    if args.out is not None:
        params["path2output"].value = args.out
    if args.stop_t is not None and "stop_t" in params:
        params["stop_t"].value = args.stop_t

    crashed = False
    crash_excpt = ""
    crash_phase = ""
    t0 = time.time()
    try:
        trinity_main.start_expansion(params)
    except SystemExit as e:  # GMC validation etc. -- record, don't treat as crash
        crash_excpt = f"SystemExit:{e}"
    except BaseException as e:  # noqa: BLE001 -- the whole point is to catch the crash
        crashed = True
        crash_excpt = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        # crude phase attribution from the traceback frames
        for ph, needle in (("1a", "run_energy_phase"), ("1b", "run_energy_implicit"),
                            ("1c", "run_transition")):
            if needle in tb:
                crash_phase = ph
    runtime = time.time() - t0

    logpath = os.path.join(out_dir, "trinity.log") if out_dir else ""
    row = {
        "config": os.path.splitext(os.path.basename(args.param))[0],
        "variant": args.variant,
        "crashed": crashed,
        "crash_phase": crash_phase,
        "crash_excpt": crash_excpt[:160],
        "end_reason": _val(params, "SimulationEndReason", ""),
        "reached_phase": _reached_phase(logpath),
        "final_t": _val(params, "t_now"),
        "final_R2": _val(params, "R2"),
        "final_v2": _val(params, "v2"),
        "final_Eb": _val(params, "Eb"),
        "runtime_s": round(runtime, 1),
        "notes": "",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"[{args.variant} {row['config']}] crashed={crashed} "
          f"reached={row['reached_phase']} reason={row['end_reason']!r} "
          f"excpt={crash_excpt[:80]!r} t={row['runtime_s']}s")


if __name__ == "__main__":
    main()
