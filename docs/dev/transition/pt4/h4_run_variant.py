#!/usr/bin/env python3
"""Drive ONE (config, variant) sim for the H4 PdV-drain-cap experiment and append
a CSV row + (optionally) a per-snapshot trajectory CSV. Production untouched
(h4_variants.py monkeypatches module attributes only).

Run one cell per process (isolates crashes + leaked global sim/logging state):

    OMP_NUM_THREADS=1 timeout 600 python h4_run_variant.py \
        --variant PDVCAP --param <cfg>.param --stop_t 0.03 \
        --t_window 3e-3 --kappa 0.9 \
        --csv h4_eval.csv --traj traj/h4_traj_<cfg>_<variant>.csv

Records crash vs clean end-state, the final (t,R2,v2,Eb), the SimulationEndReason,
the deepest phase reached, whether the cap activated (1a/1b hit counts + max
PdV/Lmech ratio in/after the window), and per-snapshot PdV/Lmech so we can SEE
whether the bubble self-sustains (Eb growing past the cap) or re-collapses.
Trajectory pulled from the run's dictionary.jsonl (production snapshot stream),
so nothing is recomputed beyond the per-row PdV/Lmech ratio.

Adapted from the H3 sibling h3_run_variant.py (same structure).
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `import h4_variants` works regardless of cwd

FOURPI = 4.0 * math.pi


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


def _pdv_ratio(r):
    """PdV/Lmech = 4*pi*R2^2*Pb*v2 / Lmech_total, from the production snapshot."""
    R2, Pb, v2, Lm = r.get("R2"), r.get("Pb"), r.get("v2"), r.get("Lmech_total")
    try:
        if None in (R2, Pb, v2, Lm) or Lm is None or Lm <= 0:
            return None
        val = FOURPI * R2 * R2 * Pb * v2 / Lm
        return val if val == val else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--param", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--traj", default=None, help="write per-snapshot trajectory CSV here")
    ap.add_argument("--stop_t", type=float, default=None)
    ap.add_argument("--t_window", type=float, default=None, help="cap window [Myr]")
    ap.add_argument("--kappa", type=float, default=None, help="cap fraction of Lmech")
    ap.add_argument("--out", default=None, help="override path2output")
    args = ap.parse_args()

    import h4_variants

    h4_variants.apply(args.variant, t_window=args.t_window, kappa=args.kappa)

    from trinity._input import read_param
    from trinity import main as trinity_main
    from trinity._functions.logging_setup import setup_logging

    params = read_param.read_param(args.param)
    cfg = os.path.splitext(os.path.basename(args.param))[0]
    out_dir = args.out or _val(params, "path2output") or f"/tmp/h4/{cfg}_{args.variant}"
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
        for ph, needle in (
            ("1a", "run_energy_phase"),
            ("1b", "run_energy_implicit"),
            ("1c", "run_transition"),
        ):
            if needle in tb:
                crash_phase = ph
    runtime = time.time() - t0

    # read the production snapshot stream for trajectory + reached-phase + ratios
    rows = []
    hits = list(Path(out_dir).rglob("dictionary.jsonl"))
    if hits:
        jsonl = str(max(hits, key=lambda p: p.stat().st_mtime))
        try:
            rows = _load_jsonl(jsonl)
        except Exception:
            rows = []

    E0 = rows[0].get("Eb") if rows else None
    Ebs = [r.get("Eb") for r in rows if isinstance(r.get("Eb"), (int, float))]
    min_Eb_seen = min(Ebs) if Ebs else None

    # self-sustain check: is Eb still growing at the end of the (truncated) run?
    # compare the last two snapshots whose t exceeds t_window (post-cap).
    tw = h4_variants.T_WINDOW
    post = [
        r
        for r in rows
        if isinstance(r.get("t_now"), (int, float))
        and r["t_now"] >= tw
        and isinstance(r.get("Eb"), (int, float))
    ]
    self_sustained = None
    final_Eb_growth = None
    if len(post) >= 2:
        final_Eb_growth = post[-1]["Eb"] - post[-2]["Eb"]
        self_sustained = bool(post[-1]["Eb"] > post[-2]["Eb"] and post[-1]["Eb"] > 0)

    # survived past window: any accepted snapshot with t >= t_window AND Eb still > 0?
    survived_past_window = bool(post and post[-1]["Eb"] > 0)

    act = h4_variants.ACT
    cap_activated = (act["n1a"] + act["n1b"]) > 0

    row = {
        "config": cfg,
        "variant": args.variant,
        "t_window": tw,
        "kappa": h4_variants.KAPPA,
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
        "min_Eb_seen": min_Eb_seen,
        "cap_activated": cap_activated,
        "n1a": act["n1a"],
        "n1b": act["n1b"],
        "max_pdv_ratio": round(act["max_pdv_ratio"], 4) if act["max_pdv_ratio"] else 0.0,
        "max_pdv_ratio_in_window": round(act["max_pdv_ratio_in_window"], 4),
        "max_pdv_ratio_after": round(act["max_pdv_ratio_after"], 4),
        "survived_past_window": survived_past_window,
        "self_sustained": self_sustained,
        "final_Eb_growth": final_Eb_growth,
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
        cols = [
            "t_now",
            "phase",
            "R2",
            "v2",
            "Eb",
            "Pb",
            "R1",
            "T0",
            "Lmech_total",
            "bubble_LTotal",
            "pdv_over_lmech",
            "rCloud",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(args.traj)), exist_ok=True)
        with open(args.traj, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "t_now": r.get("t_now"),
                        "phase": r.get("current_phase"),
                        "R2": r.get("R2"),
                        "v2": r.get("v2"),
                        "Eb": r.get("Eb"),
                        "Pb": r.get("Pb"),
                        "R1": r.get("R1"),
                        "T0": r.get("T0"),
                        "Lmech_total": r.get("Lmech_total"),
                        "bubble_LTotal": r.get("bubble_LTotal"),
                        "pdv_over_lmech": _pdv_ratio(r),
                        "rCloud": r.get("rCloud"),
                    }
                )

    print(
        f"[{args.variant} {cfg}] tw={tw} k={h4_variants.KAPPA} "
        f"crashed={crashed} reached={row['reached_phase']} "
        f"reason={str(row['end_reason'])[:42]!r} cap_act={cap_activated} "
        f"(1a={act['n1a']},1b={act['n1b']}) maxPdV/L={row['max_pdv_ratio']} "
        f"survived={survived_past_window} selfsust={self_sustained} "
        f"final_Eb={row['final_Eb']} t={row['runtime_s']}s"
    )


if __name__ == "__main__":
    main()
