#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variant capture-and-replay (+ timing + ionization-front event): which scipy
integrator configuration is the best replacement for ``scipy.integrate.odeint``
in the TRINITY shell-structure solver, and is it FASTER?

WHY THIS EXISTS
---------------
``capture_replay.py`` showed the bubble-precedent call
``solve_ivp('LSODA', t_eval=grid, dense_output=True)`` crashes on every ionized
shell solve (``ValueError: `ts` must be strictly increasing``) because the
micro-scale grid collapses LSODA's internal breakpoints. This harness pins down
the production-faithful configs AND measures wall time, AND tests an
out-of-the-box idea: a terminal EVENT at the ionization front so the integrator
stops at phi<=1e-9 and never enters the float64-overflow tail that the fixed 1k
grid otherwise integrates and then discards.

It captures each real in-run shell solve ONCE and replays it through:

  V_lsoda_teval   solve_ivp('LSODA', t_eval=grid)                  # the recommended drop-in
  V_lsoda_event   solve_ivp('LSODA', t_eval=grid, events=phi-1e-9) # stop at the I-front
  V_lsoda_dense   solve_ivp('LSODA', dense_output=True); sol.sol(grid)
  V_radau_teval   solve_ivp('Radau', t_eval=grid)
  V_bdf_teval     solve_ivp('BDF',   t_eval=grid)
  V_odeint_hi     odeint(..., mxstep=50000)                        # Option-B noise fix

Every solve_ivp variant uses rtol=atol=1.49012e-8 (odeint defaults). Accuracy is
compared against the baseline odeint result on the PHYSICALLY-USED prefix only
(up to phi-depletion / first non-finite row; production truncates there at idx).
Wall time is the MIN of TIMING_REPS bare solver calls (no fd/warns capture).

front-event verification: per call we also record idx_phi = first index where the
BASELINE odeint phi <= 1e-9. If idx_phi is small and matches the event stop, the
slice is phi-limited and the event restructure is valid; if phi never depletes in
a slice (idx_phi = -1) the slice is mass-limited and a phi-event alone is
insufficient (would need cumulative mass carried as an ODE state).

REPRODUCE
---------
    cd /home/user/trinity
    python docs/dev/shell-solver/harness/capture_replay_variants.py          # sfe=0.3 (param/simple_cluster.param)
    python docs/dev/shell-solver/harness/capture_replay_variants.py 0.6      # sfe override -> temp param

Writes docs/dev/shell-solver/data/replay_variants_sfe<sfe>.csv (long format:
one row per captured call x variant). Authored env: python 3.11.15, numpy
1.26.4, scipy 1.17.1.
"""

import os
import sys
import csv
import time
import tempfile
import warnings
import contextlib
from pathlib import Path
from collections import Counter

import numpy as np
import scipy.integrate

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

DATA_DIR = TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BASE_PARAM = TRINITY_ROOT / "param" / "simple_cluster.param"

MAX_CAPTURES = int(os.environ.get("CAP_N", "40"))
HARD_TIMEOUT_S = 300.0
RTOL = 1.49012e-8
ATOL = 1.49012e-8
TIMING_REPS = 5
PHI_THRESH = 1e-9  # matches shell_structure.py:173 phiCondition

_LSODA_MARKERS = ("lsoda", "t + h = t", "t+h=t", "excess work", "intdy")
_REAL_ODEINT = scipy.integrate.odeint
_rows = []
_start_time = None
_SFE = None
_CONFIG_PATH = None
_PARAM_FILE = BASE_PARAM
# Optional: skip captures until the run reaches this evolution phase
# (energy/implicit/transition/momentum). Lets us sample a phase other than the
# early energy phase, at the cost of running the host sim until it gets there.
_FROM_PHASE = os.environ.get("FROM_PHASE") or None
_armed = _FROM_PHASE is None
# Matrix mode: in ONE long run, capture a per-phase TARGET number of solves in
# each phase as the host sim passes through energy->implicit->transition->
# momentum. Per-phase targets come from N_ENERGY/N_IMPLICIT/N_TRANSITION/
# N_MOMENTUM (fallback PER_PHASE_N, default 15); target 0 = don't sample that
# phase. The point of the diagnostic is the SAMPLES taken IN a phase, not the
# wall time spent reaching it -- so MATRIX_MAX_S is a safety cap, not the limiter.
_PHASE_ORDER = {"energy": 0, "implicit": 1, "transition": 2, "momentum": 3}
_TARGET_PHASES = tuple(_PHASE_ORDER)
_PER_PHASE_N = os.environ.get("PER_PHASE_N")  # legacy default for all phases
_MATRIX = bool(_PER_PHASE_N) or any(
    os.environ.get("N_" + p.upper()) for p in _TARGET_PHASES)
_PHASE_N = {p: int(os.environ.get("N_" + p.upper(), _PER_PHASE_N or "15"))
            for p in _TARGET_PHASES}
_MATRIX_MAX_S = float(os.environ.get("MATRIX_MAX_S", "5400"))  # global wall safety
_phase_counts = Counter()
_max_phase_order = -1  # highest phase order the host run has reached


class _CaptureDone(Exception):
    pass


class _HostTimeout(Exception):
    pass


@contextlib.contextmanager
def _fd_capture():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    captured = {"text": ""}
    try:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        yield lambda: captured["text"]
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        try:
            tmp.flush()
            tmp.seek(0)
            captured["text"] = tmp.read().decode("utf-8", errors="replace")
        finally:
            tmp.close()


def _count_lsoda_lines(text):
    if not text:
        return 0
    low = text.lower()
    return sum(1 for line in low.splitlines()
               if any(m in line for m in _LSODA_MARKERS))


def _max_rel_diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(a), 1e-300)
    rel = np.abs(a - b) / denom
    both_tiny = (np.abs(a) < 1e-300) & (np.abs(b) < 1e-300)
    return float(np.max(np.where(both_tiny, 0.0, rel)))


def _phys_cutoff(od, n_state):
    """Index of the physically-USED prefix (phi-depletion / first non-finite row,
    whichever first); production discards everything past it. Returns >= 2."""
    od = np.asarray(od, dtype=float)
    finite_rows = np.isfinite(od).all(axis=1)
    cut = len(od) if finite_rows.all() else int(np.argmax(~finite_rows))
    if n_state == 3:
        below = np.where(od[:, 1] <= PHI_THRESH)[0]
        if below.size:
            cut = min(cut, int(below[0]) + 1)
    return max(cut, 2)


def _idx_phi(od, n_state):
    """First index where baseline phi <= threshold (ionized only); -1 if never."""
    if n_state != 3:
        return -1
    below = np.where(np.asarray(od, dtype=float)[:, 1] <= PHI_THRESH)[0]
    return int(below[0]) if below.size else -1


def _compare(od, y, n_state):
    rel = {"n": np.nan, "phi": np.nan, "tau": np.nan}
    endp = {"n": np.nan, "phi": np.nan, "tau": np.nan}
    od = np.asarray(od, dtype=float)
    if y is None:
        return rel, endp, False, -1, 0
    y = np.asarray(y, dtype=float)
    # event variant returns a SHORT array (stops at the front); compare on the
    # overlapping prefix rather than requiring identical shape.
    m = min(len(od), len(y))
    if m < 2 or y.shape[1] != od.shape[1]:
        return rel, endp, False, -1, 0
    cut = min(_phys_cutoff(od, n_state), m)
    odc, yc = od[:cut], y[:cut]
    common = np.isfinite(odc).all(axis=1) & np.isfinite(yc).all(axis=1)
    cols = ("n", "phi", "tau") if n_state == 3 else ("n", "tau")
    for j, name in enumerate(cols):
        a, b = odc[common, j], yc[common, j]
        if a.size:
            rel[name] = _max_rel_diff(a, b)
            endp[name] = _max_rel_diff(a[-1:], b[-1:])
    return rel, endp, True, cut, int(common.sum())


def _time_call(thunk):
    """Min wall time (s) over TIMING_REPS bare calls; np.nan if it raises."""
    best = np.inf
    for _ in range(TIMING_REPS):
        t0 = time.perf_counter()
        try:
            thunk()
        except Exception:  # noqa: BLE001 - failing variants still get a (fast) time
            return np.nan
        best = min(best, time.perf_counter() - t0)
    return best


def _solve(name, func, y0, t, args, is_ionised):
    """Run one configuration once; return (y_or_None, success, status, message,
    error, n_pts_out, event_fired, event_r)."""
    fun = lambda r, y: np.asarray(func(y, r, *args))  # noqa: E731  odeint(y,t)->ivp(t,y)
    t0, t1 = float(t[0]), float(t[-1])
    y0 = np.asarray(y0, dtype=float)
    teval = np.asarray(t, dtype=float)
    event_fired, event_r = 0, np.nan
    try:
        if name == "V_odeint_hi":
            y = _REAL_ODEINT(func, y0, t, args=args, mxstep=50000)
            return y, True, 0, "", "", len(y), event_fired, event_r
        if name == "V_lsoda_dense":
            sol = scipy.integrate.solve_ivp(
                fun, (t0, t1), y0, method="LSODA", dense_output=True,
                rtol=RTOL, atol=ATOL)
            y = sol.sol(teval).T if sol.sol is not None else None
            return y, bool(sol.success), int(sol.status), str(sol.message), "", \
                (-1 if y is None else len(y)), event_fired, event_r
        if name == "V_lsoda_event":
            if is_ionised:
                def phi_event(r, y):
                    return y[1] - PHI_THRESH
                phi_event.terminal = True
                phi_event.direction = -1
                events = phi_event
            else:
                events = None  # neutral region has no phi; falls back to t_eval
            sol = scipy.integrate.solve_ivp(
                fun, (t0, t1), y0, method="LSODA", t_eval=teval,
                events=events, rtol=RTOL, atol=ATOL)
            if events is not None and sol.t_events is not None and len(sol.t_events[0]):
                event_fired = 1
                event_r = float(sol.t_events[0][0])
            return sol.y.T, bool(sol.success), int(sol.status), str(sol.message), "", \
                len(sol.y.T), event_fired, event_r
        method = {"V_lsoda_teval": "LSODA", "V_radau_teval": "Radau",
                  "V_bdf_teval": "BDF"}[name]
        sol = scipy.integrate.solve_ivp(
            fun, (t0, t1), y0, method=method, t_eval=teval, rtol=RTOL, atol=ATOL)
        return sol.y.T, bool(sol.success), int(sol.status), str(sol.message), "", \
            len(sol.y.T), event_fired, event_r
    except Exception as exc:  # noqa: BLE001
        return None, False, -99, "", f"{type(exc).__name__}: {exc}", -1, event_fired, event_r


def _thunk(name, func, y0, t, args, is_ionised):
    """A bare (no-capture) callable of the solve, for timing."""
    fun = lambda r, y: np.asarray(func(y, r, *args))  # noqa: E731
    t0, t1 = float(t[0]), float(t[-1])
    y0a = np.asarray(y0, dtype=float)
    teval = np.asarray(t, dtype=float)
    if name == "V_odeint_hi":
        return lambda: _REAL_ODEINT(func, y0, t, args=args, mxstep=50000)
    if name == "V_lsoda_dense":
        def _c():
            s = scipy.integrate.solve_ivp(fun, (t0, t1), y0a, method="LSODA",
                                          dense_output=True, rtol=RTOL, atol=ATOL)
            _ = s.sol(teval) if s.sol is not None else None
        return _c
    if name == "V_lsoda_event" and is_ionised:
        def pe(r, y):
            return y[1] - PHI_THRESH
        pe.terminal = True
        pe.direction = -1
        return lambda: scipy.integrate.solve_ivp(fun, (t0, t1), y0a, method="LSODA",
                                                  t_eval=teval, events=pe,
                                                  rtol=RTOL, atol=ATOL)
    method = {"V_lsoda_teval": "LSODA", "V_lsoda_event": "LSODA",
              "V_radau_teval": "Radau", "V_bdf_teval": "BDF"}[name]
    return lambda: scipy.integrate.solve_ivp(fun, (t0, t1), y0a, method=method,
                                             t_eval=teval, rtol=RTOL, atol=ATOL)


_VARIANTS = ("V_lsoda_teval", "V_lsoda_event", "V_lsoda_dense",
             "V_radau_teval", "V_bdf_teval", "V_odeint_hi")


def _run_variant(name, func, y0, t, args, od_ref, n_state, is_ionised):
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        with _fd_capture() as chatter:
            y, success, status, message, error, n_out, ev_fired, ev_r = \
                _solve(name, func, y0, t, args, is_ionised)
    lsoda = _count_lsoda_lines(chatter()) if error == "" else -1
    pyw = len(wlist) if error == "" else -1
    t_ms = _time_call(_thunk(name, func, y0, t, args, is_ionised)) * 1e3

    rel, endp, shapes_match, cutoff, n_common = _compare(od_ref, y, n_state)
    return {
        "variant": name, "success": int(bool(success)), "status": status,
        "shapes_match": int(shapes_match), "cutoff_idx": cutoff,
        "n_common_finite": n_common, "n_pts_out": n_out,
        "event_fired": ev_fired, "event_r": ev_r,
        "time_ms": t_ms, "lsoda_warns": lsoda, "py_warns": pyw,
        "max_rel_diff_n": rel["n"], "max_rel_diff_phi": rel["phi"],
        "max_rel_diff_tau": rel["tau"], "endpoint_rel_diff_n": endp["n"],
        "endpoint_rel_diff_phi": endp["phi"], "endpoint_rel_diff_tau": endp["tau"],
        "message": message, "error": error,
    }


def _current_phase(args):
    params = args[2] if len(args) >= 3 else None
    if params is not None:
        try:
            return params["current_phase"].value
        except Exception:
            return ""
    return ""


def _capture_one(func, y0, t, args, kwargs, phase_tag):
    """Baseline odeint + all variants on one shell solve; append rows tagged with
    phase_tag. Returns the real odeint result so the host run is unperturbed."""
    call_idx = len({r["call_idx"] for r in _rows})
    with warnings.catch_warnings(record=True) as base_warns:
        warnings.simplefilter("always")
        with _fd_capture() as base_chatter:
            od_ref = _REAL_ODEINT(func, y0, t, args=args, **kwargs)
    base_lsoda = _count_lsoda_lines(base_chatter())
    base_py = len(base_warns)
    base_t_ms = _time_call(lambda: _REAL_ODEINT(func, y0, t, args=args, **kwargs)) * 1e3

    n_state = np.asarray(od_ref).shape[1]
    is_ionised = bool(args[1]) if len(args) >= 2 else (n_state == 3)
    idx_phi = _idx_phi(od_ref, n_state)

    for vname in _VARIANTS:
        res = _run_variant(vname, func, y0, t, args, od_ref, n_state, is_ionised)
        res.update({
            "call_idx": call_idx, "phase": phase_tag,
            "is_ionised": int(is_ionised), "n_state": n_state,
            "n_pts": int(len(t)), "r_start": float(t[0]), "r_stop": float(t[-1]),
            "idx_phi": idx_phi,
            "baseline_odeint_time_ms": base_t_ms,
            "baseline_odeint_lsoda_warns": base_lsoda,
            "baseline_odeint_py_warns": base_py,
            "speedup_vs_odeint": (base_t_ms / res["time_ms"]
                                  if res["time_ms"] and not np.isnan(res["time_ms"])
                                  and res["time_ms"] > 0 else np.nan),
        })
        _rows.append(res)

    speeds = {r["variant"]: r["speedup_vs_odeint"]
              for r in _rows if r["call_idx"] == call_idx}
    print(f"[capture {call_idx + 1} phase={phase_tag or '?'}] ion={int(is_ionised)} "
          f"npts={len(t)} idx_phi={idx_phi} base={base_t_ms:.2f}ms "
          f"teval_x={speeds.get('V_lsoda_teval', float('nan')):.2f} "
          f"event_x={speeds.get('V_lsoda_event', float('nan')):.2f}",
          file=sys.stderr, flush=True)
    return od_ref


def _patched_odeint(func, y0, t, args=(), **kwargs):
    global _start_time, _armed
    if _start_time is None:
        _start_time = time.time()

    # ---- MATRIX MODE: one pass, per-phase target captures ----
    if _MATRIX:
        global _max_phase_order
        phase = _current_phase(args)
        if phase in _PHASE_ORDER:
            _max_phase_order = max(_max_phase_order, _PHASE_ORDER[phase])
        # A phase is "done" once its target is met OR the run has moved past it
        # (so a phase shorter than its target stops the run instead of stalling).
        def _done(p):
            return (_phase_counts[p] >= _PHASE_N[p]
                    or _max_phase_order > _PHASE_ORDER[p])
        if all(_done(p) for p in _TARGET_PHASES):
            raise _CaptureDone()
        if (time.time() - _start_time) > _MATRIX_MAX_S:
            raise _HostTimeout(f"matrix wall budget {_MATRIX_MAX_S:.0f}s reached")
        if (phase not in _PHASE_ORDER or _PHASE_N[phase] == 0
                or _phase_counts[phase] >= _PHASE_N[phase]):
            return _REAL_ODEINT(func, y0, t, args=args, **kwargs)
        if _phase_counts[phase] == 0:
            print(f"[matrix] entering phase '{phase}' (target {_PHASE_N[phase]}, "
                  f"wall={time.time()-_start_time:.0f}s)", file=sys.stderr, flush=True)
        _phase_counts[phase] += 1
        return _capture_one(func, y0, t, args, kwargs, phase)

    # ---- SINGLE / FROM_PHASE MODE ----
    if not _armed:
        if _current_phase(args) == _FROM_PHASE:
            _armed = True
            print(f"[armed] reached phase '{_FROM_PHASE}'; capturing {MAX_CAPTURES} solves",
                  file=sys.stderr, flush=True)
        else:
            return _REAL_ODEINT(func, y0, t, args=args, **kwargs)
    n_calls = len({r["call_idx"] for r in _rows})
    if (not _FROM_PHASE and (time.time() - _start_time) > HARD_TIMEOUT_S
            and n_calls < MAX_CAPTURES):
        raise _HostTimeout(f"Stalled: {n_calls} captures after {HARD_TIMEOUT_S:.0f}s")
    if n_calls >= MAX_CAPTURES:
        raise _CaptureDone()
    return _capture_one(func, y0, t, args, kwargs, _current_phase(args))


def _write_csv(csv_path):
    if not _rows:
        print("No captures; nothing written.", file=sys.stderr)
        return
    cols = ["call_idx", "phase", "is_ionised", "n_state", "n_pts", "r_start", "r_stop",
            "idx_phi", "variant", "success", "status", "shapes_match",
            "cutoff_idx", "n_common_finite", "n_pts_out", "event_fired", "event_r",
            "baseline_odeint_time_ms", "time_ms", "speedup_vs_odeint",
            "baseline_odeint_lsoda_warns", "baseline_odeint_py_warns",
            "lsoda_warns", "py_warns",
            "max_rel_diff_n", "max_rel_diff_phi", "max_rel_diff_tau",
            "endpoint_rel_diff_n", "endpoint_rel_diff_phi", "endpoint_rel_diff_tau",
            "message", "error"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in _rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nWrote {len(_rows)} rows -> {csv_path}", file=sys.stderr)


def _make_param():
    """Return the param path to drive. Priority: an explicit param-file arg
    (used as-is) > an SFE override (writes a temp param over simple_cluster) >
    the base simple_cluster param."""
    global _PARAM_FILE
    if _CONFIG_PATH is not None:
        _PARAM_FILE = _CONFIG_PATH
        return _CONFIG_PATH
    if _SFE is None:
        _PARAM_FILE = BASE_PARAM
        return BASE_PARAM
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".param", delete=False)
    tmp.write(f"mCloud    1e5\nsfe    {_SFE}\n")
    tmp.close()
    _PARAM_FILE = Path(tmp.name)
    return _PARAM_FILE


def _drive_host_run():
    import logging
    logging.disable(logging.CRITICAL)
    from trinity._input import read_param
    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    from trinity import main
    params = read_param.read_param(str(_make_param()))
    gmc_check = validate_gmc_from_params(params)
    if not gmc_check.valid:
        raise RuntimeError("GMC validation failed: " + "; ".join(gmc_check.errors))
    main.start_expansion(params)


def main():
    global _SFE, _CONFIG_PATH
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if Path(arg).exists():
            _CONFIG_PATH = Path(arg)
        else:
            _SFE = float(arg)  # bare number -> sfe override on simple_cluster
    if _CONFIG_PATH is not None:
        tag = _CONFIG_PATH.stem
    elif _SFE is not None:
        tag = f"sfe{_SFE:g}"
    else:
        tag = "sfe0.3"
    if _FROM_PHASE:
        tag = f"{tag}_{_FROM_PHASE}"
    if _MATRIX:
        tag = f"matrix_{tag}"
    csv_path = DATA_DIR / f"replay_variants_{tag}.csv"

    # If an outer `timeout` kills us mid-gated-run, still flush what we captured.
    import signal

    def _on_term(signum, frame):
        _write_csv(csv_path)
        os._exit(143)
    signal.signal(signal.SIGTERM, _on_term)

    print("=" * 70, file=sys.stderr)
    print(f"shell-solver VARIANT + TIMING + EVENT  (config={tag})", file=sys.stderr)
    print(f"  python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"scipy {scipy.__version__}  timing_reps={TIMING_REPS}", file=sys.stderr)
    print(f"  variants: {', '.join(_VARIANTS)}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    scipy.integrate.odeint = _patched_odeint
    try:
        _drive_host_run()
        print(f"Host run finished early ({len({r['call_idx'] for r in _rows})} captured).",
              file=sys.stderr)
    except _CaptureDone:
        target = dict(_PHASE_N) if _MATRIX else MAX_CAPTURES
        print(f"Capture targets met ({target}); host aborted cleanly.", file=sys.stderr)
    except _HostTimeout as exc:
        print(f"WARNING: {exc}. Writing what we have.", file=sys.stderr)
    except SystemExit as exc:
        print(f"Host run sys.exit({exc.code}); "
              f"{len({r['call_idx'] for r in _rows})} captured.", file=sys.stderr)
    finally:
        scipy.integrate.odeint = _REAL_ODEINT
        if _SFE is not None and _PARAM_FILE.exists():
            try:
                _PARAM_FILE.unlink()
            except OSError:
                pass
        _write_csv(csv_path)

    if _rows:
        import logging
        logging.disable(logging.NOTSET)
        n_calls = len({r["call_idx"] for r in _rows})
        n_ion = len({r["call_idx"] for r in _rows if r["is_ionised"]})
        base = [r["baseline_odeint_time_ms"] for r in _rows
                if r["variant"] == "V_lsoda_teval"]
        base_med = sorted(base)[len(base) // 2] if base else float("nan")
        idxp = [r["idx_phi"] for r in _rows if r["variant"] == "V_lsoda_teval"]
        print("\n" + "=" * 70, file=sys.stderr)
        print(f"SUMMARY {tag}  ({n_calls} calls: ion={n_ion}, neu={n_calls - n_ion})",
              file=sys.stderr)
        print(f"  baseline odeint time: median {base_med:.3f} ms/call", file=sys.stderr)
        print(f"  idx_phi (phi-depletion row): min={min(idxp)} max={max(idxp)} "
              f"(of ~1000; -1=never)", file=sys.stderr)
        if _MATRIX:
            cap_calls = {r["call_idx"]: r["phase"] for r in _rows}
            per = Counter(cap_calls.values())
            print(f"  captures per phase: {dict(per)} (targets {_PHASE_N})",
                  file=sys.stderr)
        for v in _VARIANTS:
            vr = [r for r in _rows if r["variant"] == v]
            nok = sum(r["success"] for r in vr)
            rels = [r["max_rel_diff_n"] for r in vr
                    if r["success"] and not np.isnan(r["max_rel_diff_n"])]
            worst = f"{max(rels):.2e}" if rels else "n/a"
            sp = [r["speedup_vs_odeint"] for r in vr
                  if not np.isnan(r["speedup_vs_odeint"])]
            sp_med = f"{sorted(sp)[len(sp)//2]:.2f}x" if sp else "n/a"
            evf = sum(r["event_fired"] for r in vr)
            print(f"  {v:16s} ok={nok:2d}/{len(vr)}  speedup~{sp_med:>6s}  "
                  f"worst_rel_n={worst:>9s}  event_fired={evf}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)


if __name__ == "__main__":
    main()
