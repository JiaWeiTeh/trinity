#!/usr/bin/env python3
"""Drive ONE (config, box-width) LEGACY sim for the H5 clamp-width sweep, append a
summary row to h5_sweep.csv, and write a per-snapshot trajectory CSV. Production
untouched (h5_variants.py monkeypatches the get_betadelta box constants only).

One sim per process (isolates crashes + leaked module-global sim/logging state):

    OMP_NUM_THREADS=1 timeout 1200 python h5_run_variant.py \
        --width W1 --param ../../cleanroom/configs/simple_cluster.param \
        --stop_t 0.5 --csv h5_sweep.csv --traj data/h5_traj_simple_cluster_W1.csv

The run uses the LEGACY beta-delta solver (the whole point: "legacy solver with a
wider box"), forced via params['betadelta_solver']='legacy'. The box is widened by
h5_variants.apply(width) BEFORE start_expansion (constants read at call time inside
the solver — get_betadelta.py:969-972, 1044-1045, 1053).

Recorded per row: config, box width id + (beta_min,beta_max,delta_min,delta_max),
whether the cooling ratio (Lgain-Lloss)/Lgain ever crosses < 0.05 and at what t,
ratio_min, the boundary-pin fraction (segments with beta/delta within EPS of the
INSTALLED box edge, over the segments leading to the crossing — or all segments if
no crossing), beta at the crossing, the deepest phase reached, and runtime.
"""
import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `import h5_variants` works regardless of cwd
# c0_consistency lives in the cleanroom dir; reuse its run_config + annotate.
CLEANROOM = os.path.normpath(os.path.join(HERE, "..", "..", "cleanroom"))
sys.path.insert(0, CLEANROOM)

TRIGGER = 0.05      # cooling-balance transition trigger
PIN_EPS = 0.02      # within 2% of an installed box edge counts as "on the boundary"


def _ratio(r):
    lg, ll = r.get("bubble_Lgain"), r.get("bubble_Lloss")
    try:
        if lg is None or ll is None or lg != lg or ll != ll or lg <= 0:
            return None
        return (lg - ll) / lg
    except (TypeError, ValueError):
        return None


def _reached_phase(rows):
    order = {"energy": 1, "implicit": 2, "transition": 3, "momentum": 4}
    drank, deepest = 0, ""
    for r in rows:
        rk = order.get(r.get("phase", ""), 0)
        if rk > drank:
            drank, deepest = rk, {1: "1a", 2: "1b", 3: "1c", 4: "momentum"}[rk]
    return deepest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", required=True, help="box-width id (W0/W1/W2/W3)")
    ap.add_argument("--param", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--stop_t", type=float, default=None)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    import h5_variants
    bmin, bmax, dmin, dmax = h5_variants.apply(args.width)

    import c0_consistency as c0

    cfg = os.path.splitext(os.path.basename(args.param))[0]
    run_dir = args.run_dir or f"/tmp/h5/{cfg}_{args.width}"

    crashed, crash_excpt = False, ""
    rows = []
    t0 = time.time()
    try:
        jsonl = c0.run_config(args.param, args.stop_t, refine=1.0,
                              solver="legacy", run_dir=run_dir)
        rows = c0.annotate(c0.load_rows(jsonl))
    except SystemExit as e:
        crash_excpt = f"SystemExit:{e}"
        # still try to read whatever snapshots landed
        hits = list(Path(run_dir).rglob("dictionary.jsonl"))
        if hits:
            jl = str(max(hits, key=lambda p: p.stat().st_mtime))
            try:
                rows = c0.annotate(c0.load_rows(jl))
            except Exception:
                rows = []
    except BaseException as e:  # noqa: BLE001 -- catching the crash is the point
        crashed = True
        crash_excpt = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        hits = list(Path(run_dir).rglob("dictionary.jsonl"))
        if hits:
            jl = str(max(hits, key=lambda p: p.stat().st_mtime))
            try:
                rows = c0.annotate(c0.load_rows(jl))
            except Exception:
                rows = []
    runtime = time.time() - t0

    # ratio trajectory + crossing
    seq = []  # (t, ratio, beta, delta)
    for r in rows:
        t, b, d = r.get("t_now"), r.get("cool_beta"), r.get("cool_delta")
        ra = _ratio(r)
        if t is None or t <= 0 or ra is None:
            continue
        try:
            b = float(b) if b is not None and b == b else None
            d = float(d) if d is not None and d == d else None
        except (TypeError, ValueError):
            b = d = None
        seq.append((t, ra, b, d))

    cross_i = next((i for i, (t, ra, b, d) in enumerate(seq) if ra < TRIGGER), None)
    crosses = cross_i is not None
    cross_t = seq[cross_i][0] if crosses else None
    beta_at_cross = seq[cross_i][2] if crosses else None
    ratio_min = min((ra for _, ra, _, _ in seq), default=None)

    # boundary-pin fraction relative to the INSTALLED box, over the pre-crossing
    # segments (or all segments if no crossing).
    def on_boundary(b, d):
        hit = False
        if b is not None:
            hit = hit or abs(b - bmin) <= PIN_EPS or abs(b - bmax) <= PIN_EPS
        if d is not None:
            hit = hit or abs(d - dmin) <= PIN_EPS or abs(d - dmax) <= PIN_EPS
        return hit

    pre = seq[:cross_i + 1] if crosses else seq
    pin_frac = (sum(on_boundary(b, d) for _, _, b, d in pre) / len(pre)) if pre else None

    row = {
        "config": cfg,
        "box_width": args.width,
        "beta_min": bmin, "beta_max": bmax, "delta_min": dmin, "delta_max": dmax,
        "crosses": crosses,
        "cross_t": ("" if cross_t is None else f"{cross_t:.6g}"),
        "ratio_min": ("" if ratio_min is None else f"{ratio_min:.6g}"),
        "beta_at_cross": ("" if beta_at_cross is None else f"{beta_at_cross:.4f}"),
        "boundary_pin_frac": ("" if pin_frac is None else f"{pin_frac:.4f}"),
        "reached_phase": _reached_phase(rows),
        "n_rows": len(rows),
        "crashed": crashed,
        "crash_excpt": crash_excpt[:140],
        "runtime_s": round(runtime, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    if args.traj and seq:
        os.makedirs(os.path.dirname(os.path.abspath(args.traj)), exist_ok=True)
        with open(args.traj, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_now", "ratio", "cool_beta", "cool_delta"])
            for t, ra, b, d in seq:
                w.writerow([t, ra, ("" if b is None else b), ("" if d is None else d)])

    print(f"[{args.width} {cfg}] crosses={crosses} cross_t={cross_t} "
          f"ratio_min={ratio_min} pin={pin_frac} reached={row['reached_phase']} "
          f"crashed={crashed} n={len(rows)} {runtime:.0f}s box=b[{bmin},{bmax}]d[{dmin},{dmax}]")


if __name__ == "__main__":
    main()
