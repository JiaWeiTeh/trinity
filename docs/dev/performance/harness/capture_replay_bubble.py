#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-process capture + replay + timing harness for the HOTPATH F1 resample.

One live ``trinity.main.start_expansion`` run per config. On each gated bubble
call (``get_bubbleproperties_pure``) we:

  1. run the BASELINE result (``real_gbp(params)``) -- and return it to the host
     so the host trajectory stays byte-identical (capture is a side effect);
  2. for every variant in ``residual_variants.VARIANTS`` monkeypatch the module
     global ``BL._get_velocity_residuals`` (which the dMdt fsolve wrapper at
     bubble_luminosity.py:458 reads), re-run ``get_bubbleproperties_pure``,
     restore the global, and record the BubbleProperties outputs + min-of-K
     timing;
  3. write one CSV row per call x variant, with ``rel_*`` measured against the
     baseline row for the same call.

The phase gate copies the matrix structure from the shell harness
(docs/dev/shell-solver/harness/capture_replay_variants.py:88-105,371-414),
adapted to read ``params['current_phase'].value`` (queryable through the
``BubbleParamsView`` used in the implicit phase).

REPRODUCE
---------
    cd /home/user/trinity
    N_ENERGY=20 N_IMPLICIT=100 python \
        docs/dev/performance/harness/capture_replay_bubble.py \
        docs/dev/transition/harness/mock_hybr.param

Writes docs/dev/performance/data/bubble_resample_<config>.csv (one row per
call x variant) and 2-3 state pickles under data/states/ for offline replay
(replay_from_dump.py). Authored env: python 3.11.x, numpy 1.26.4, scipy 1.17.1.
"""

import os
import sys
import csv
import time
import pickle
from pathlib import Path
from collections import Counter

import numpy as np

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))
HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import trinity.bubble_structure.bubble_luminosity as BL  # noqa: E402
import residual_variants  # noqa: E402  (same dir, on sys.path above)

DATA_DIR = TRINITY_ROOT / "docs" / "dev" / "performance" / "data"
STATE_DIR = DATA_DIR / "states"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Min wall time over TIMING_REPS reps (matches the shell harness convention).
TIMING_REPS = int(os.environ.get("TIMING_REPS", "3"))

# ---- matrix phase gate (copied from capture_replay_variants.py:88-105) ------
_PHASE_ORDER = {"energy": 0, "implicit": 1, "transition": 2, "momentum": 3}
_TARGET_PHASES = ("energy", "implicit")
_PHASE_N = {"energy": int(os.environ.get("N_ENERGY", "20")),
            "implicit": int(os.environ.get("N_IMPLICIT", "100"))}
_MATRIX_MAX_S = float(os.environ.get("MATRIX_MAX_S", "5400"))  # global wall safety
_phase_counts = Counter()
_max_phase_order = -1

# Number of view-aware state pickles to dump for the offline replay.
N_STATE_DUMPS = int(os.environ.get("N_STATE_DUMPS", "3"))

_rows = []
_start_time = None
_n_pickled = 0
_REAL_GBP = BL.get_bubbleproperties_pure
_REAL_RESID = BL._get_velocity_residuals


class _CaptureDone(Exception):
    pass


class _HostTimeout(Exception):
    pass


def _current_phase(params):
    try:
        return params['current_phase'].value
    except Exception:
        return ""


def _time_call(thunk):
    """Min wall time (s) over TIMING_REPS bare calls; np.nan if it raises."""
    best = np.inf
    for _ in range(TIMING_REPS):
        t0 = time.perf_counter()
        try:
            thunk()
        except Exception:  # noqa: BLE001 - a failing variant still gets a (fast) time
            return np.nan
        best = min(best, time.perf_counter() - t0)
    return best


def _props(bp):
    """Pull the compared scalars off a BubbleProperties dataclass."""
    return {
        "bubble_dMdt": float(bp.bubble_dMdt),
        "bubble_LTotal": float(bp.bubble_LTotal),
        "bubble_T_r_Tb": float(bp.bubble_T_r_Tb),
        "bubble_mass": float(bp.bubble_mass),
        "bubble_Tavg": float(bp.bubble_Tavg),
        "R1": float(bp.R1),
        "Pb": float(bp.Pb),
    }


def _rel(a, b):
    """Relative diff of variant ``a`` vs baseline ``b`` (0 when both ~0)."""
    denom = max(abs(b), 1e-300)
    if abs(a) < 1e-300 and abs(b) < 1e-300:
        return 0.0
    return abs(a - b) / denom


def _dump_state(params, config, call_idx):
    """Pickle one view-aware param snapshot for offline replay (skip cubes)."""
    global _n_pickled
    if _n_pickled >= N_STATE_DUMPS:
        return
    real = getattr(params, '_params', params)
    pvals, skipped = {}, []
    for k in real.keys():
        try:
            v = params[k].value
            pickle.dumps(v)
            pvals[k] = v
        except Exception:
            skipped.append(k)
    phase = _current_phase(params)
    state = {"param_values": pvals, "skipped_param_keys": skipped,
             "config": config, "phase": phase, "call_index": call_idx}
    fname = STATE_DIR / f"state_{config}_{phase}_{call_idx:04d}.pkl"
    with open(fname, "wb") as fh:
        pickle.dump(state, fh)
    _n_pickled += 1
    print(f"[state] dumped {fname.name} (skipped {len(skipped)} cubes)",
          file=sys.stderr, flush=True)


def _capture_one(params, config, phase):
    """Baseline + every variant on one bubble call; append one row per variant.
    Returns the baseline BubbleProperties so the host run is unperturbed."""
    call_idx = len({r["call_index"] for r in _rows})

    # --- baseline (recursion guard: real_gbp installed by the hook) ---
    base_bp = _REAL_GBP(params)
    base = _props(base_bp)
    base_t = _time_call(lambda: _REAL_GBP(params)) * 1e3

    _record_row(config, phase, call_idx, "baseline", base, base, base_t, ok=True)

    # --- every coarse variant (swap the module-global residual fn) ---
    for vname, vfn in residual_variants.VARIANTS.items():
        if vname == "baseline":
            continue
        BL._get_velocity_residuals = vfn
        try:
            ok = True
            try:
                v_bp = _REAL_GBP(params)
                vout = _props(v_bp)
            except Exception as exc:  # noqa: BLE001 - record the failure, keep going
                ok = False
                vout = {k: np.nan for k in base}
                print(f"  [variant {vname} failed: {type(exc).__name__}: "
                      f"{str(exc)[:80]}]", file=sys.stderr, flush=True)
            v_t = _time_call(lambda: _REAL_GBP(params)) * 1e3 if ok else np.nan
        finally:
            BL._get_velocity_residuals = _REAL_RESID
        _record_row(config, phase, call_idx, vname, vout, base, v_t, ok=ok)

    _dump_state(params, config, call_idx)

    rel = {r["variant"]: r["rel_dMdt"] for r in _rows
           if r["call_index"] == call_idx}
    print(f"[capture {call_idx + 1} phase={phase}] base_dMdt={base['bubble_dMdt']:.4e} "
          f"base={base_t:.1f}ms  rel_dMdt: "
          + " ".join(f"{k}={rel[k]:.1e}" for k in rel if k != "baseline"),
          file=sys.stderr, flush=True)
    return base_bp


def _record_row(config, phase, call_idx, variant, out, base, time_ms, ok):
    _rows.append({
        "config": config, "phase": phase, "call_index": call_idx,
        "variant": variant, "npts": residual_variants.VARIANT_NPTS.get(variant, ""),
        "bubble_dMdt": out["bubble_dMdt"], "bubble_LTotal": out["bubble_LTotal"],
        "bubble_T_r_Tb": out["bubble_T_r_Tb"], "bubble_mass": out["bubble_mass"],
        "bubble_Tavg": out["bubble_Tavg"], "R1": out["R1"], "Pb": out["Pb"],
        "time_ms": time_ms,
        "rel_dMdt": _rel(out["bubble_dMdt"], base["bubble_dMdt"]),
        "rel_LTotal": _rel(out["bubble_LTotal"], base["bubble_LTotal"]),
        "rel_T_r_Tb": _rel(out["bubble_T_r_Tb"], base["bubble_T_r_Tb"]),
        "rel_mass": _rel(out["bubble_mass"], base["bubble_mass"]),
        "ok": int(bool(ok)),
    })


def _make_hook(config):
    def hook(params):
        global _start_time, _max_phase_order
        if _start_time is None:
            _start_time = time.time()

        phase = _current_phase(params)
        if phase in _PHASE_ORDER:
            _max_phase_order = max(_max_phase_order, _PHASE_ORDER[phase])

        # A phase is "done" once its target is met OR the run moved past it.
        def _done(p):
            return (_phase_counts[p] >= _PHASE_N[p]
                    or _max_phase_order > _PHASE_ORDER[p])

        if all(_done(p) for p in _TARGET_PHASES):
            raise _CaptureDone()
        if (time.time() - _start_time) > _MATRIX_MAX_S:
            raise _HostTimeout(f"matrix wall budget {_MATRIX_MAX_S:.0f}s reached")

        target_now = phase in _PHASE_N and _PHASE_N[phase] > 0
        if not target_now or _phase_counts[phase] >= _PHASE_N[phase]:
            return _REAL_GBP(params)

        if _phase_counts[phase] == 0:
            print(f"[matrix] entering phase '{phase}' (target {_PHASE_N[phase]}, "
                  f"wall={time.time() - _start_time:.0f}s)", file=sys.stderr, flush=True)
        _phase_counts[phase] += 1

        # Recursion guard: real_gbp during the capture, restore the hook after.
        BL.get_bubbleproperties_pure = _REAL_GBP
        try:
            return _capture_one(params, config, phase)
        finally:
            BL.get_bubbleproperties_pure = hook
    return hook


def _write_csv(csv_path):
    if not _rows:
        print("No captures; nothing written.", file=sys.stderr)
        return
    cols = ["config", "phase", "call_index", "variant", "npts",
            "bubble_dMdt", "bubble_LTotal", "bubble_T_r_Tb", "bubble_mass",
            "bubble_Tavg", "R1", "Pb", "time_ms",
            "rel_dMdt", "rel_LTotal", "rel_T_r_Tb", "rel_mass", "ok"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in _rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nWrote {len(_rows)} rows -> {csv_path}", file=sys.stderr)


def _drive_host_run(param_path):
    import logging
    logging.disable(logging.CRITICAL)
    from trinity._input import read_param
    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    from trinity import main
    params = read_param.read_param(str(param_path))
    gmc_check = validate_gmc_from_params(params)
    if not gmc_check.valid:
        raise RuntimeError("GMC validation failed: " + "; ".join(gmc_check.errors))
    main.start_expansion(params)


def main():
    if len(sys.argv) < 2:
        print("usage: capture_replay_bubble.py <config.param>", file=sys.stderr)
        return 2
    param_path = Path(sys.argv[1])
    if not param_path.exists():
        print(f"config not found: {param_path}", file=sys.stderr)
        return 2
    # Name the run by CONFIG_NAME if given (the sweep's tag), else the param
    # stem. The stem is wrong when configs share a .param (sfe0.3/sfe0.6 both use
    # simple_cluster) or use a temp file -> the sweep passes CONFIG_NAME=<tag>.
    config = os.environ.get("CONFIG_NAME") or param_path.stem
    csv_path = DATA_DIR / f"bubble_resample_{config}.csv"

    import numpy as _np
    import scipy as _sp
    print("=" * 70, file=sys.stderr)
    print(f"F1 resample CAPTURE+REPLAY+TIMING  (config={config})", file=sys.stderr)
    print(f"  python {sys.version.split()[0]}  numpy {_np.__version__}  "
          f"scipy {_sp.__version__}  timing_reps={TIMING_REPS}", file=sys.stderr)
    print(f"  targets: {dict(_PHASE_N)}  variants: "
          f"{', '.join(residual_variants.VARIANTS)}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    BL.get_bubbleproperties_pure = _make_hook(config)
    try:
        _drive_host_run(param_path)
        print(f"Host run finished early ({len({r['call_index'] for r in _rows})} "
              f"captured).", file=sys.stderr)
    except _CaptureDone:
        print(f"Capture targets met ({dict(_PHASE_N)}); host aborted cleanly.",
              file=sys.stderr)
    except _HostTimeout as exc:
        print(f"WARNING: {exc}. Writing what we have.", file=sys.stderr)
    except SystemExit as exc:
        print(f"Host run sys.exit({exc.code}); "
              f"{len({r['call_index'] for r in _rows})} captured.", file=sys.stderr)
    except BaseException as exc:  # noqa: BLE001 - a live crash is still a data point
        print(f"Host run ended: {type(exc).__name__}: {str(exc)[:120]}",
              file=sys.stderr)
    finally:
        BL.get_bubbleproperties_pure = _REAL_GBP
        BL._get_velocity_residuals = _REAL_RESID
        _write_csv(csv_path)

    import logging
    logging.disable(logging.NOTSET)
    cap = {r["call_index"]: r["phase"] for r in _rows}
    per = Counter(cap.values())
    print("\n" + "=" * 70, file=sys.stderr)
    print(f"SUMMARY {config}  ({len(cap)} calls)  captures/phase={dict(per)} "
          f"(targets {dict(_PHASE_N)})", file=sys.stderr)
    for v in residual_variants.VARIANTS:
        vr = [r for r in _rows if r["variant"] == v]
        if not vr:
            continue
        nok = sum(r["ok"] for r in vr)
        rels = [r["rel_dMdt"] for r in vr if r["variant"] != "baseline"]
        worst = f"{max(rels):.2e}" if rels else "0 (baseline)"
        ts = [r["time_ms"] for r in vr if not np.isnan(r["time_ms"])]
        tmed = f"{sorted(ts)[len(ts) // 2]:.2f}ms" if ts else "n/a"
        print(f"  {v:10s} ok={nok:3d}/{len(vr)}  med_time={tmed:>8s}  "
              f"worst_rel_dMdt={worst:>12s}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
