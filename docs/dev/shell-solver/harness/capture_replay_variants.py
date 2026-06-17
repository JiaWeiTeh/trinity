#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variant capture-and-replay: which scipy integrator configuration (if any) is a
safe drop-in for ``scipy.integrate.odeint`` in the TRINITY shell-structure solver?

WHY THIS EXISTS
---------------
The first harness (``capture_replay.py``) showed that the bubble-precedent call
``solve_ivp(method='LSODA', t_eval=grid, dense_output=True)`` raised
``ValueError: `ts` must be strictly increasing or decreasing`` on EVERY ionized
shell solve. That error is raised by scipy's GLOBAL dense-output container
(``OdeSolution``), which is only built when ``dense_output=True`` and which
rejects the duplicate internal breakpoints LSODA produces when its step collapses
in the stiff ionization-front layer (the "t + h = t" regime).

That left the production-faithful configurations untested. This harness captures
each real in-run shell solve ONCE and replays it through several integrator
configurations, so we learn WHICH (if any) reproduces odeint without failing:

  V_lsoda_teval   solve_ivp('LSODA', t_eval=grid)                 # no dense_output
  V_lsoda_dense   solve_ivp('LSODA', dense_output=True); sol.sol(grid)   # bubble style
  V_radau_teval   solve_ivp('Radau', t_eval=grid)
  V_bdf_teval     solve_ivp('BDF',   t_eval=grid)
  V_odeint_hi     odeint(..., mxstep=50000)                       # Option-B evidence

All solve_ivp variants use rtol=atol=1.49012e-8 (odeint's defaults). Every variant
is compared against the DEFAULT odeint result (the production baseline) on the same
(func, y0, grid, args): max relative difference per state variable + the physically
used last grid point. Fortran LSODA fd-chatter and Python warnings are counted per
variant.

REPRODUCE
---------
    cd /home/user/trinity
    python docs/dev/shell-solver/harness/capture_replay_variants.py

Writes docs/dev/shell-solver/data/replay_variants.csv  (long format: one row per
captured call x variant). Authored env: python 3.11.15, numpy 1.26.4, scipy 1.17.1.
"""

import os
import sys
import csv
import time
import warnings
import tempfile
import contextlib
from pathlib import Path

import numpy as np
import scipy.integrate

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

DATA_DIR = TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "replay_variants.csv"
PARAM_FILE = TRINITY_ROOT / "param" / "simple_cluster.param"

MAX_CAPTURES = 40        # a few more than the first run, to try to reach the neutral region
HARD_TIMEOUT_S = 300.0
RTOL = 1.49012e-8        # scipy.integrate.odeint defaults
ATOL = 1.49012e-8

_LSODA_MARKERS = ("lsoda", "t + h = t", "t+h=t", "excess work", "intdy")
_REAL_ODEINT = scipy.integrate.odeint
_rows = []
_start_time = None


class _CaptureDone(Exception):
    pass


class _HostTimeout(Exception):
    pass


@contextlib.contextmanager
def _fd_capture():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
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
    """Index of the physically-USED prefix of a shell-slice solution.

    The fixed slice grid runs far past the ionization front, where the n^2
    stiffness makes n blow up to inf/nan (this is what makes odeint report
    "excess work"). Production discards everything past the first index where
    phi depletes (<=1e-9) or mass is swept; here we approximate that physical
    truncation with the phi-depletion crossing (ionized) and the first
    non-finite row, whichever comes first. Only this prefix is meaningful to
    compare. Returns at least 2."""
    od = np.asarray(od, dtype=float)
    finite_rows = np.isfinite(od).all(axis=1)
    cut = len(od) if finite_rows.all() else int(np.argmax(~finite_rows))
    if n_state == 3:  # ionized: phi is column 1
        below = np.where(od[:, 1] <= 1e-9)[0]
        if below.size:
            cut = min(cut, int(below[0]) + 1)
    return max(cut, 2)


def _compare(od, y, n_state):
    """Compare a variant solution y against the baseline odeint od on the
    physically-used prefix only, masking any non-finite rows. Returns
    (rel dict, endpoint-rel dict, shapes_match, cutoff_idx, n_common_finite)."""
    rel = {"n": np.nan, "phi": np.nan, "tau": np.nan}
    endp = {"n": np.nan, "phi": np.nan, "tau": np.nan}
    if y is None or np.asarray(y).shape != np.asarray(od).shape:
        return rel, endp, False, -1, 0
    od = np.asarray(od, dtype=float)
    y = np.asarray(y, dtype=float)
    cut = _phys_cutoff(od, n_state)
    odc, yc = od[:cut], y[:cut]
    common = np.isfinite(odc).all(axis=1) & np.isfinite(yc).all(axis=1)
    cols = ("n", "phi", "tau") if n_state == 3 else ("n", "tau")
    for j, name in enumerate(cols):
        a, b = odc[common, j], yc[common, j]
        if a.size:
            rel[name] = _max_rel_diff(a, b)
            endp[name] = _max_rel_diff(a[-1:], b[-1:])
    return rel, endp, True, cut, int(common.sum())


def _run_variant(name, func, y0, t, args, od_ref, n_state):
    """Run one integrator configuration; return a result row dict (without the
    per-call identity columns, which the caller adds)."""
    fun = lambda r, y: np.asarray(func(y, r, *args))  # noqa: E731  odeint(y,t)->ivp(t,y)
    t0, t1 = float(t[0]), float(t[-1])
    y0 = np.asarray(y0, dtype=float)
    success = False
    status = -99
    message = ""
    error = ""
    y = None
    n_pts_out = -1

    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            with _fd_capture() as chatter:
                if name == "V_odeint_hi":
                    y = _REAL_ODEINT(func, y0, t, args=args, mxstep=50000)
                    success = True
                    status = 0
                    n_pts_out = len(y)
                elif name == "V_lsoda_dense":
                    sol = scipy.integrate.solve_ivp(
                        fun, (t0, t1), y0, method="LSODA",
                        dense_output=True, rtol=RTOL, atol=ATOL)
                    success = bool(sol.success)
                    status = int(sol.status)
                    message = str(sol.message)
                    y = sol.sol(np.asarray(t, dtype=float)).T if sol.sol is not None else None
                    n_pts_out = -1 if y is None else len(y)
                else:
                    method = {"V_lsoda_teval": "LSODA",
                              "V_radau_teval": "Radau",
                              "V_bdf_teval": "BDF"}[name]
                    sol = scipy.integrate.solve_ivp(
                        fun, (t0, t1), y0, method=method,
                        t_eval=np.asarray(t, dtype=float),
                        rtol=RTOL, atol=ATOL)
                    success = bool(sol.success)
                    status = int(sol.status)
                    message = str(sol.message)
                    y = sol.y.T
                    n_pts_out = len(y)
        lsoda_lines = _count_lsoda_lines(chatter())
        py_warns = len(wlist)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        lsoda_lines = -1
        py_warns = -1

    rel, endp, shapes_match, cutoff, n_common = _compare(od_ref, y, n_state)
    return {
        "variant": name,
        "success": int(bool(success)),
        "status": status,
        "shapes_match": int(shapes_match),
        "cutoff_idx": cutoff,
        "n_common_finite": n_common,
        "n_pts_out": n_pts_out,
        "lsoda_warns": lsoda_lines,
        "py_warns": py_warns,
        "max_rel_diff_n": rel["n"],
        "max_rel_diff_phi": rel["phi"],
        "max_rel_diff_tau": rel["tau"],
        "endpoint_rel_diff_n": endp["n"],
        "endpoint_rel_diff_phi": endp["phi"],
        "endpoint_rel_diff_tau": endp["tau"],
        "message": message,
        "error": error,
    }


_VARIANTS = ("V_lsoda_teval", "V_lsoda_dense", "V_radau_teval",
             "V_bdf_teval", "V_odeint_hi")


def _patched_odeint(func, y0, t, args=(), **kwargs):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    n_calls = len({r["call_idx"] for r in _rows})
    if (time.time() - _start_time) > HARD_TIMEOUT_S and n_calls < MAX_CAPTURES:
        raise _HostTimeout(f"Stalled: {n_calls} captures after {HARD_TIMEOUT_S:.0f}s")
    if n_calls >= MAX_CAPTURES:
        raise _CaptureDone()

    call_idx = n_calls
    # Baseline: the REAL production odeint result (default mxstep), with warning capture.
    with warnings.catch_warnings(record=True) as base_warns:
        warnings.simplefilter("always")
        with _fd_capture() as base_chatter:
            od_ref = _REAL_ODEINT(func, y0, t, args=args, **kwargs)
    base_lsoda = _count_lsoda_lines(base_chatter())
    base_py = len(base_warns)

    n_state = np.asarray(od_ref).shape[1]
    is_ionised = bool(args[1]) if len(args) >= 2 else (n_state == 3)

    for vname in _VARIANTS:
        res = _run_variant(vname, func, y0, t, args, od_ref, n_state)
        res.update({
            "call_idx": call_idx,
            "is_ionised": int(is_ionised),
            "n_state": n_state,
            "n_pts": int(len(t)),
            "r_start": float(t[0]),
            "r_stop": float(t[-1]),
            "baseline_odeint_lsoda_warns": base_lsoda,
            "baseline_odeint_py_warns": base_py,
        })
        _rows.append(res)

    ok = {r["variant"]: r["success"] for r in _rows if r["call_idx"] == call_idx}
    print(f"[capture {call_idx + 1}/{MAX_CAPTURES}] ion={int(is_ionised)} "
          f"npts={len(t)} base_pywarn={base_py} "
          f"ok=" + ",".join(f"{k.split('_',1)[1]}:{v}" for k, v in ok.items()),
          file=sys.stderr, flush=True)
    return od_ref


def _write_csv():
    if not _rows:
        print("No captures; nothing written.", file=sys.stderr)
        return
    cols = ["call_idx", "is_ionised", "n_state", "n_pts", "r_start", "r_stop",
            "variant", "success", "status", "shapes_match", "cutoff_idx",
            "n_common_finite", "n_pts_out",
            "baseline_odeint_lsoda_warns", "baseline_odeint_py_warns",
            "lsoda_warns", "py_warns",
            "max_rel_diff_n", "max_rel_diff_phi", "max_rel_diff_tau",
            "endpoint_rel_diff_n", "endpoint_rel_diff_phi", "endpoint_rel_diff_tau",
            "message", "error"]
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in _rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nWrote {len(_rows)} rows -> {CSV_PATH}", file=sys.stderr)


def _drive_host_run():
    import logging
    logging.disable(logging.CRITICAL)
    from trinity._input import read_param
    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    from trinity import main
    params = read_param.read_param(str(PARAM_FILE))
    gmc_check = validate_gmc_from_params(params)
    if not gmc_check.valid:
        raise RuntimeError("GMC validation failed: " + "; ".join(gmc_check.errors))
    main.start_expansion(params)


def main():
    print("=" * 70, file=sys.stderr)
    print("shell-solver VARIANT capture-and-replay", file=sys.stderr)
    print(f"  python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"scipy {scipy.__version__}", file=sys.stderr)
    print(f"  variants: {', '.join(_VARIANTS)}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    scipy.integrate.odeint = _patched_odeint
    try:
        _drive_host_run()
        print(f"Host run finished early ({len({r['call_idx'] for r in _rows})} captured).",
              file=sys.stderr)
    except _CaptureDone:
        print(f"Reached {MAX_CAPTURES} captures; host aborted cleanly.", file=sys.stderr)
    except _HostTimeout as exc:
        print(f"WARNING: {exc}. Writing what we have.", file=sys.stderr)
    except SystemExit as exc:
        print(f"Host run sys.exit({exc.code}); "
              f"{len({r['call_idx'] for r in _rows})} captured.", file=sys.stderr)
    finally:
        scipy.integrate.odeint = _REAL_ODEINT
        _write_csv()

    if _rows:
        import logging
        logging.disable(logging.NOTSET)
        n_calls = len({r["call_idx"] for r in _rows})
        n_ion = len({r["call_idx"] for r in _rows if r["is_ionised"]})
        print("\n" + "=" * 70, file=sys.stderr)
        print(f"SUMMARY  ({n_calls} calls: ionised={n_ion}, neutral={n_calls - n_ion})",
              file=sys.stderr)
        for v in _VARIANTS:
            vr = [r for r in _rows if r["variant"] == v]
            nok = sum(r["success"] for r in vr)
            rels = [r["max_rel_diff_n"] for r in vr
                    if r["success"] and not np.isnan(r["max_rel_diff_n"])]
            worst = f"{max(rels):.2e}" if rels else "n/a"
            lsoda = sum(r["lsoda_warns"] for r in vr if r["lsoda_warns"] >= 0)
            pyw = sum(r["py_warns"] for r in vr if r["py_warns"] >= 0)
            print(f"  {v:16s} ok={nok}/{len(vr)}  worst_rel_n={worst:>9s}  "
                  f"lsoda_lines={lsoda}  py_warns={pyw}", file=sys.stderr)
        base_py = sum({r["call_idx"]: r["baseline_odeint_py_warns"]
                       for r in _rows}.values())
        print(f"  baseline odeint py-warns (excess-work), summed over calls: {base_py}",
              file=sys.stderr)
        print("=" * 70, file=sys.stderr)


if __name__ == "__main__":
    main()
