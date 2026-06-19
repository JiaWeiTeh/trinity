#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 full-run A/B: whole-simulation wall time, baseline vs F1 (coarse t_eval).

⚠️⚠️ BUGGED — DO NOT USE. This harness runs BOTH variants in ONE Python process
(``_run_one`` x2), but trinity carries module-level global state (crash-safe
snapshot handlers, caches) that LEAKS between ``start_expansion`` calls, so the
second run is corrupted. On 2026-06-18 this produced a *false* "F1 divergence"
(t_now 0.005 vs 0.3) that was a process-state artifact, NOT the fix. Use
``f1_fullrun_equiv.sh`` instead — it runs each variant in a SEPARATE ``run.py``
process and compares at matched ``t_now``. Kept only as the cautionary example.

This is the REAL "headline" for the F1 resample optimization. P0 showed the
per-bubble-call speedup is a uniform ~1.4-1.5x across all configs (the 60k
resample is fixed-size), so the interesting number is the FULL-RUN wall-time
reduction -- largest where the run spends the most wall time inside bubble
calls (the degenerate `simple_cluster`).

For each config we run the entire `main.start_expansion` twice in-process:
  * baseline -- the production `_get_velocity_residuals` (60k dense resample);
  * F1       -- `residual_variants.make_variant(_RESIDUAL_NPTS)` monkeypatched
                onto `BL._get_velocity_residuals` (what P3 will ship).
We time each run, count residual calls, and read the FINAL snapshot from the
run's `dictionary.jsonl` to assert physical equivalence within the G2 band
(dMdt shifts ~1e-6, so the trajectory is NOT byte-identical -- compare within
tolerance, not exact).

HEAVY -- runs full sims (the degenerate configs take many minutes EACH, x2
variants). Do NOT run while the timing sweep (or anything CPU-heavy) is live:
contention corrupts the wall-time ratio. Run on a quiet box, post-sweep.

    cd /home/user/trinity
    python docs/dev/performance/harness/ab_fullrun.py \
        mock_hybr=docs/dev/transition/harness/mock_hybr.param \
        simple_cluster=param/simple_cluster.param

Writes docs/dev/performance/data/ab_fullrun.csv (one row per config x variant).

NB: VERIFY ON FIRST RUN -- this harness has not been executed yet (it was staged
during the P0 sweep). Confirm the final-snapshot keys + the output-dir discovery
below against a real run before trusting the numbers.
"""

import os
import sys
import csv
import glob
import json
import time
import logging
from pathlib import Path

import numpy as np

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))
HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import trinity.bubble_structure.bubble_luminosity as BL  # noqa: E402
import residual_variants  # noqa: E402

DATA_DIR = TRINITY_ROOT / "docs" / "dev" / "performance" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# The N P1 locked in (RESAMPLE_PLAN / P3_PRODUCTION_PATCH.md).
NPTS = int(os.environ.get("RESIDUAL_NPTS", "500"))

# Final-snapshot scalars compared baseline-vs-F1 (verified present in
# outputs/*/dictionary.jsonl: 97 keys incl. these).
_COMPARE_KEYS = ("t_now", "R2", "rShell", "Eb", "v2")
_G2_TOL = 3e-3  # 0.3%, the F1 equivalence band


def _count_wrap(fn):
    """Wrap a residual fn with a call counter."""
    state = {"n": 0}

    def wrapped(*a, **k):
        state["n"] += 1
        return fn(*a, **k)

    return wrapped, state


def _newest_dictionary_since(t0):
    """The dictionary.jsonl written/touched after t0 (this run's output)."""
    best, best_m = None, t0
    for p in glob.glob(str(TRINITY_ROOT / "outputs" / "*" / "dictionary.jsonl")):
        m = os.path.getmtime(p)
        if m >= best_m:
            best, best_m = p, m
    return best


def _final_snapshot(path):
    """Last JSON line of dictionary.jsonl -> dict of the compared scalars."""
    if path is None or not os.path.exists(path):
        return {k: np.nan for k in _COMPARE_KEYS}, None
    last = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return {k: np.nan for k in _COMPARE_KEYS}, None
    d = json.loads(last)
    out = {}
    for k in _COMPARE_KEYS:
        v = d.get(k, np.nan)
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = np.nan
    return out, d.get("current_phase")


def _run_one(config, param_path, residual_fn):
    """Run the full sim once under ``residual_fn``; return timing + final state."""
    from trinity._input import read_param
    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    from trinity import main

    params = read_param.read_param(str(param_path))
    gmc = validate_gmc_from_params(params)
    if not gmc.valid:
        raise RuntimeError("GMC validation failed: " + "; ".join(gmc.errors))

    wrapped, state = _count_wrap(residual_fn)
    BL._get_velocity_residuals = wrapped
    logging.disable(logging.CRITICAL)
    t0 = time.time()
    try:
        main.start_expansion(params)
    except SystemExit:
        pass
    finally:
        wall = time.time() - t0
        logging.disable(logging.NOTSET)
        BL._get_velocity_residuals = _REAL_RESID
    snap, phase = _final_snapshot(_newest_dictionary_since(t0))
    return {"wall_s": wall, "n_resid": state["n"], "final_phase": phase, **snap}


_REAL_RESID = BL._get_velocity_residuals


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    configs = []
    for a in argv:
        name, _, path = a.partition("=")
        configs.append((name, path or name))

    f1 = residual_variants.make_variant(NPTS)
    variants = (("baseline", _REAL_RESID), (f"F1_M{NPTS}", f1))

    rows = []
    for name, path in configs:
        print(f"\n=== {name} ({path}) ===", file=sys.stderr)
        base_row = None
        for vname, vfn in variants:
            print(f"  running {vname} ...", file=sys.stderr, flush=True)
            r = _run_one(name, path, vfn)
            r.update(config=name, variant=vname)
            if vname == "baseline":
                base_row = r
                r["speedup"] = 1.0
            else:
                r["speedup"] = (base_row["wall_s"] / r["wall_s"]
                                if r["wall_s"] else float("nan"))
            # physical equivalence vs baseline (within the G2 band)
            r["max_rel_state"] = max(
                (abs(r[k] - base_row[k]) / max(abs(base_row[k]), 1e-300)
                 for k in _COMPARE_KEYS
                 if np.isfinite(r[k]) and np.isfinite(base_row[k])),
                default=float("nan"),
            )
            r["equiv_ok"] = bool(r["max_rel_state"] <= _G2_TOL)
            print(f"    wall={r['wall_s']:.1f}s  n_resid={r['n_resid']}  "
                  f"speedup={r['speedup']:.2f}x  max_rel_state={r['max_rel_state']:.2e}"
                  f"  phase={r['final_phase']}", file=sys.stderr)
            rows.append(r)

    cols = (["config", "variant", "wall_s", "n_resid", "speedup",
             "max_rel_state", "equiv_ok", "final_phase"] + list(_COMPARE_KEYS))
    out = DATA_DIR / "ab_fullrun.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nWrote {len(rows)} rows -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
