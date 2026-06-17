#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capture-and-replay harness: is scipy.integrate.solve_ivp(LSODA) a drop-in for
scipy.integrate.odeint in the TRINITY shell-structure solver?

WHAT IT DOES
------------
1. Monkeypatches ``scipy.integrate.odeint`` with a wrapper (the shell solver in
   ``trinity/shell_structure/shell_structure.py`` calls ``scipy.integrate.odeint``
   at module scope, so patching the attribute on the ``scipy.integrate`` module
   intercepts every shell solve with the REAL in-run y0 / radius grid / params).
2. For the first MAX_CAPTURES shell solves, the wrapper:
     - runs the REAL odeint (capturing Fortran LSODA fd-1/2 chatter + py warnings),
     - runs solve_ivp(fun, (t0,t1), y0, method='LSODA', t_eval=t, dense_output=True,
       rtol=1.49012e-8, atol=1.49012e-8) on the same problem (same fd/py capture),
     - compares solve_ivp.y.T against odeint elementwise (max abs / max rel diff
       per state variable, plus the physically-used LAST grid point),
     - records a row,
     - returns the REAL odeint result unchanged so the host run is unperturbed.
   After MAX_CAPTURES rows it raises _CaptureDone to abort the (slow) host run.
3. Drives a real run of ``param/simple_cluster.param`` by replicating run.py's
   single-run setup (read_param -> logging -> GMC validate -> main.start_expansion)
   so the captured (y0, grid, params) are genuine in-run values, NOT synthetic.
4. Writes docs/dev/shell-solver/data/replay_comparison.csv (one row per capture).

REPRODUCE
---------
    cd /home/user/trinity
    python docs/dev/shell-solver/harness/capture_replay.py

ENVIRONMENT (captured at authoring time; printed at runtime for the record)
    python 3.11.15   numpy 1.26.4   scipy 1.17.1
    odeint default tolerances rtol=atol=1.49012e-8 are reproduced for solve_ivp.

NOTES / CAVEATS
---------------
- odeint's RHS is func(y, r, *args); solve_ivp's is fun(t, y). The wrapper builds
  fun = lambda r, y: np.asarray(func(y, r, *args)) to bridge the signatures.
- The ionized region integrates 3 states (n, phi, tau); the neutral region 2
  states (n, tau). The CSV columns for phi are NaN on neutral-region captures.
- params holds live Python objects (.value attributes); everything stays in ONE
  process. Nothing is pickled.
- A wall-clock safety abort (HARD_TIMEOUT_S) guards against the host run getting
  stuck before MAX_CAPTURES is reached.
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

# --- make 'trinity' importable regardless of cwd -----------------------------
TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

DATA_DIR = TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "replay_comparison.csv"
PARAM_FILE = TRINITY_ROOT / "param" / "simple_cluster.param"

MAX_CAPTURES = 30
HARD_TIMEOUT_S = 300.0  # safety: abort host run if we stall before MAX_CAPTURES
ODEINT_RTOL = 1.49012e-8  # scipy.integrate.odeint default
ODEINT_ATOL = 1.49012e-8  # scipy.integrate.odeint default

_LSODA_MARKERS = ("lsoda", "t + h = t", "t+h=t", "excess work", "intdy")

_REAL_ODEINT = scipy.integrate.odeint
_captures = []
_start_time = None


class _CaptureDone(Exception):
    """Raised once MAX_CAPTURES rows are collected to abort the host run fast."""


class _HostTimeout(Exception):
    """Raised if the host run stalls before MAX_CAPTURES is reached."""


@contextlib.contextmanager
def _fd_capture():
    """Redirect OS-level fd 1 and 2 to a temp file so we catch Fortran LSODA
    chatter printed by compiled code (it bypasses Python's sys.stdout). Yields
    a callable returning the captured text."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    captured = {"text": ""}
    try:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)

        def _read():
            return captured["text"]

        yield _read
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
    """Max relative difference between arrays a (reference, odeint) and b
    (solve_ivp), elementwise, with a small floor on the denominator so that
    near-zero reference values don't blow the ratio up artificially."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(a), 1e-300)
    rel = np.abs(a - b) / denom
    # Where both are ~0, treat as no error.
    both_tiny = (np.abs(a) < 1e-300) & (np.abs(b) < 1e-300)
    rel = np.where(both_tiny, 0.0, rel)
    return float(np.max(rel))


def _max_abs_diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)))


def _patched_odeint(func, y0, t, args=(), **kwargs):
    """Drop-in wrapper around the real odeint. For the first MAX_CAPTURES
    invocations it ALSO runs solve_ivp(LSODA) on the same problem, records a
    comparison row, then returns the REAL odeint result unchanged."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    # Safety: don't let the slow host run go forever if we stall.
    if (time.time() - _start_time) > HARD_TIMEOUT_S and len(_captures) < MAX_CAPTURES:
        raise _HostTimeout(
            f"Stalled: {len(_captures)} captures after {HARD_TIMEOUT_S:.0f}s"
        )

    if len(_captures) >= MAX_CAPTURES:
        # We have enough; abort the host run immediately.
        raise _CaptureDone()

    call_idx = len(_captures)

    # --- run REAL odeint, capturing fd-level LSODA chatter + py warnings -----
    with warnings.catch_warnings(record=True) as od_warns:
        warnings.simplefilter("always")
        with _fd_capture() as od_chatter:
            sol_odeint = _REAL_ODEINT(func, y0, t, args=args, **kwargs)
    od_lsoda_lines = _count_lsoda_lines(od_chatter())
    od_py_warns = len(od_warns)
    # odeint surfaces LSODA "Excess work" / "t+h=t" via ODEintWarning (python),
    # NOT via fd-1/2, so count those warning messages too.
    od_excess_work = sum(
        1 for w in od_warns
        if any(m in str(w.message).lower()
               for m in ("excess work", "t + h = t", "t+h=t"))
    )

    od = np.asarray(sol_odeint, dtype=float)
    n_state = od.shape[1]
    is_ionised = bool(args[1]) if len(args) >= 2 else (n_state == 3)
    cols = ("n", "phi", "tau") if n_state == 3 else ("n", "tau")

    # odeint health: did its own profile stay finite, or underflow/NaN?
    od_finite = bool(np.all(np.isfinite(od)))
    od_endpoint_underflow = bool(
        np.any(np.abs(od[-1]) < 1e-300) and np.all(od[-1] >= 0)
    )

    # --- physical truncation index (the ONLY part the host actually uses) ----
    # shell_structure.py truncates each slice at idx = first point where the
    # cumulative swept-up shell mass >= shell_mass (both regions) OR phi<=1e-9
    # (ionised region only). Rows past idx are discarded, so solver fidelity
    # only matters on [0:idx+1]. Replicate that here from the odeint profile.
    params = args[2] if len(args) >= 3 else None
    trunc_idx = len(t) - 1
    if params is not None:
        try:
            rstep = float(t[1] - t[0]) if len(t) > 1 else 0.0
            mu_H = params['mu_convert'].value
            mShell_end = params['shell_mass'].value
            nprof = od[:, 0]
            mloc = np.empty_like(t)
            mloc[0] = 0.0
            mloc[1:] = nprof[1:] * mu_H * 4 * np.pi * t[1:] ** 2 * rstep
            mcum = np.cumsum(mloc)
            cond = mcum >= mShell_end
            if is_ionised:
                cond = cond | (od[:, 1] <= 1e-9)
            nz = np.nonzero(cond)[0]
            trunc_idx = int(nz[0]) if len(nz) else len(t) - 1
        except Exception:  # noqa: BLE001 - diagnostic only
            trunc_idx = len(t) - 1

    # --- build solve_ivp problem on the SAME (func, y0, t, args) --------------
    def fun(r, y):
        return np.asarray(func(y, r, *args))

    def _run_ivp(dense):
        """Returns (success, status, message, y_T_or_None, lsoda_lines,
        py_warns, error_str)."""
        try:
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                with _fd_capture() as chatter:
                    sol = scipy.integrate.solve_ivp(
                        fun,
                        (float(t[0]), float(t[-1])),
                        np.asarray(y0, dtype=float),
                        method="LSODA",
                        t_eval=np.asarray(t, dtype=float),
                        dense_output=dense,
                        rtol=ODEINT_RTOL,
                        atol=ODEINT_ATOL,
                    )
            yT = sol.y.T if sol.y.size else None
            return (bool(sol.success), int(sol.status), str(sol.message),
                    yT, _count_lsoda_lines(chatter()), len(wl), "")
        except Exception as exc:  # noqa: BLE001 - record any solver failure
            return (False, -99, "", None, -1, -1, f"{type(exc).__name__}: {exc}")

    # Proposed config: t_eval + dense_output.
    (ivp_success, ivp_status, ivp_message, sol_ivp_y,
     ivp_lsoda_lines, ivp_py_warns, ivp_error) = _run_ivp(dense=True)
    # Fallback config: t_eval only (no dense_output).
    (ivp2_success, ivp2_status, ivp2_message, sol_ivp2_y,
     ivp2_lsoda_lines, ivp2_py_warns, ivp2_error) = _run_ivp(dense=False)

    def _compare(iv_y, hi=None):
        """Compare odeint vs solve_ivp over rows [0:hi] (hi=None -> full grid).
        Returns (shapes_match, rel{}, absd{}, end_od{}, end_iv{}). The endpoint
        is taken at row hi-1 (the physically-used last point when hi=trunc+1)."""
        rel = {"n": np.nan, "phi": np.nan, "tau": np.nan}
        absd = {"n": np.nan, "phi": np.nan, "tau": np.nan}
        e_od = {"n": np.nan, "phi": np.nan, "tau": np.nan}
        e_iv = {"n": np.nan, "phi": np.nan, "tau": np.nan}
        ok = iv_y is not None and iv_y.shape == od.shape
        if ok:
            iv = np.asarray(iv_y, dtype=float)
            sl = slice(0, hi) if hi is not None else slice(None)
            last = (hi - 1) if hi is not None else -1
            for j, name in enumerate(cols):
                rel[name] = _max_rel_diff(od[sl, j], iv[sl, j])
                absd[name] = _max_abs_diff(od[sl, j], iv[sl, j])
                e_od[name] = float(od[last, j])
                e_iv[name] = float(iv[last, j])
        return ok, rel, absd, e_od, e_iv

    # --- FULL-GRID comparison (raw, includes float-overflow tail) ------------
    shapes_match, rel, absd, end_odeint, end_ivp = _compare(sol_ivp_y)
    # If dense-output variant crashed, fall back to the t_eval-only profile for
    # the numerical-agreement comparison (so we still learn whether the VALUES
    # agree even when only the dense interpolant construction failed).
    if not shapes_match:
        shapes_match2, rel, absd, end_odeint, end_ivp = _compare(sol_ivp2_y)
        compared_against = "t_eval_only" if shapes_match2 else "none"
        shapes_match = shapes_match2
    else:
        compared_against = "t_eval+dense"

    # --- TRUNCATION-REGION comparison (the physically-used part only) --------
    # Compare over [0 : trunc_idx+1]; this is what the host actually consumes.
    # Use whichever ivp variant gave a profile (prefer dense, else t_eval-only).
    iv_for_trunc = sol_ivp_y if sol_ivp_y is not None else sol_ivp2_y
    hi = trunc_idx + 1
    (trunc_ok, trel, tabsd, tend_od, tend_iv) = _compare(iv_for_trunc, hi=hi)

    def _endpoint_rel(name):
        a, b = tend_od[name], tend_iv[name]
        if np.isnan(a) or np.isnan(b):
            return np.nan
        return _max_rel_diff(np.array([a]), np.array([b]))

    row = {
        "call_idx": call_idx,
        "is_ionised": int(is_ionised),
        "n_state": n_state,
        "n_pts": int(len(t)),
        "r_start": float(t[0]),
        "r_stop": float(t[-1]),
        "y0_n": float(np.asarray(y0, float)[0]),
        # --- odeint health ---
        "odeint_finite": int(od_finite),
        "odeint_endpoint_underflow": int(od_endpoint_underflow),
        "odeint_lsoda_warns": od_lsoda_lines + od_excess_work,
        "odeint_excess_work": od_excess_work,
        "odeint_py_warns": od_py_warns,
        # --- solve_ivp(LSODA), t_eval + dense_output (PROPOSED config) ---
        "ivp_success": int(ivp_success),
        "ivp_status": ivp_status,
        "ivp_message": ivp_message,
        "ivp_error": ivp_error,
        "ivp_lsoda_warns": ivp_lsoda_lines,
        "ivp_py_warns": ivp_py_warns,
        # --- solve_ivp(LSODA), t_eval only (FALLBACK config) ---
        "ivp_teval_success": int(ivp2_success),
        "ivp_teval_status": ivp2_status,
        "ivp_teval_error": ivp2_error,
        # --- agreement (vs whichever ivp variant produced a profile) ---
        "compared_against": compared_against,
        "shapes_match": int(shapes_match),
        # full-grid (raw) agreement: includes the discarded float-overflow tail,
        # so these are EXPECTED to be huge/nan in the degenerate regime.
        "fullgrid_max_rel_diff_n": rel["n"],
        "fullgrid_max_rel_diff_phi": rel["phi"],
        "fullgrid_max_rel_diff_tau": rel["tau"],
        # truncation index actually used by shell_structure.py for this slice.
        "trunc_idx": int(trunc_idx),
        "trunc_ok": int(trunc_ok),
        # PHYSICALLY-USED agreement: over rows [0 : trunc_idx+1] only.
        "max_rel_diff_n": trel["n"],
        "max_rel_diff_phi": trel["phi"],
        "max_rel_diff_tau": trel["tau"],
        "max_abs_diff_n": tabsd["n"],
        "max_abs_diff_phi": tabsd["phi"],
        "max_abs_diff_tau": tabsd["tau"],
        "endpoint_rel_diff_n": _endpoint_rel("n"),
        "endpoint_rel_diff_phi": _endpoint_rel("phi"),
        "endpoint_rel_diff_tau": _endpoint_rel("tau"),
        "endpoint_n_odeint": tend_od["n"],
        "endpoint_n_ivp": tend_iv["n"],
        "endpoint_phi_odeint": tend_od["phi"],
        "endpoint_phi_ivp": tend_iv["phi"],
        "endpoint_tau_odeint": tend_od["tau"],
        "endpoint_tau_ivp": tend_iv["tau"],
    }
    _captures.append(row)

    # Progress to real stderr (not the redirected fd, which is restored now).
    print(f"[capture {call_idx + 1}/{MAX_CAPTURES}] "
          f"ion={int(is_ionised)} npts={len(t)} y0_n={float(np.asarray(y0,float)[0]):.2e} "
          f"trunc_idx={trunc_idx} od_excess={od_excess_work} "
          f"ivp_dense_ok={int(ivp_success)} ivp_teval_ok={int(ivp2_success)} "
          f"trunc_rel_n={trel['n']:.2e}",
          file=sys.stderr, flush=True)

    return sol_odeint


def _write_csv():
    if not _captures:
        print("No captures recorded; nothing to write.", file=sys.stderr)
        return
    fieldnames = list(_captures[0].keys())
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in _captures:
            w.writerow(r)
    print(f"\nWrote {len(_captures)} rows -> {CSV_PATH}", file=sys.stderr)


def _drive_host_run():
    """Replicate run.py's single-run setup, with quiet logging, then start the
    expansion. The patched odeint aborts via _CaptureDone after MAX_CAPTURES."""
    import logging
    logging.disable(logging.CRITICAL)  # keep the host run quiet & fast

    from trinity._input import read_param
    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    from trinity import main

    params = read_param.read_param(str(PARAM_FILE))

    gmc_check = validate_gmc_from_params(params)
    if not gmc_check.valid:
        raise RuntimeError(
            "GMC validation failed for the harness param file; cannot drive a "
            "real run. Errors: " + "; ".join(gmc_check.errors)
        )

    main.start_expansion(params)


def main():
    print("=" * 70, file=sys.stderr)
    print("shell-solver capture-and-replay harness", file=sys.stderr)
    print(f"  python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"scipy {scipy.__version__}", file=sys.stderr)
    print(f"  param file: {PARAM_FILE}", file=sys.stderr)
    print(f"  target captures: {MAX_CAPTURES}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Install the patch on the module attribute the shell solver looks up.
    scipy.integrate.odeint = _patched_odeint
    try:
        _drive_host_run()
        print("Host run finished before reaching MAX_CAPTURES "
              f"({len(_captures)} captured).", file=sys.stderr)
    except _CaptureDone:
        print(f"Reached {MAX_CAPTURES} captures; host run aborted cleanly.",
              file=sys.stderr)
    except _HostTimeout as exc:
        print(f"WARNING: {exc}. Writing what we have.", file=sys.stderr)
    except SystemExit as exc:
        print(f"Host run called sys.exit({exc.code}); "
              f"{len(_captures)} captured.", file=sys.stderr)
    finally:
        scipy.integrate.odeint = _REAL_ODEINT
        _write_csv()

    # --- console summary ------------------------------------------------------
    if _captures:
        import logging
        logging.disable(logging.NOTSET)

        def _vals(key):
            return [r[key] for r in _captures if not np.isnan(r[key])]

        rel_n, rel_phi, rel_tau = _vals("max_rel_diff_n"), _vals("max_rel_diff_phi"), _vals("max_rel_diff_tau")
        n_ion = sum(r["is_ionised"] for r in _captures)
        n_dense_fail = sum(1 for r in _captures if not r["ivp_success"])
        n_teval_fail = sum(1 for r in _captures if not r["ivp_teval_success"])
        n_od_excess = sum(1 for r in _captures if r["odeint_excess_work"] > 0)
        n_od_nonfinite = sum(1 for r in _captures if not r["odeint_finite"])
        n_od_underflow = sum(1 for r in _captures if r["odeint_endpoint_underflow"])
        od_lsoda = sum(r["odeint_lsoda_warns"] for r in _captures)
        iv_lsoda = sum(r["ivp_lsoda_warns"] for r in _captures if r["ivp_lsoda_warns"] >= 0)
        print("\n" + "=" * 70, file=sys.stderr)
        print("SUMMARY", file=sys.stderr)
        print(f"  captures: {len(_captures)} (ionised={n_ion}, "
              f"neutral={len(_captures) - n_ion})", file=sys.stderr)
        print(f"  --- ODEINT health (the reference solver, same problems) ---", file=sys.stderr)
        print(f"  odeint calls with Excess-work/t+h=t warning : {n_od_excess}", file=sys.stderr)
        print(f"  odeint calls with non-finite profile         : {n_od_nonfinite}", file=sys.stderr)
        print(f"  odeint calls whose endpoint underflowed ~0   : {n_od_underflow}", file=sys.stderr)
        print(f"  --- solve_ivp(LSODA) ---", file=sys.stderr)
        print(f"  t_eval+dense_output failures : {n_dense_fail}/{len(_captures)}", file=sys.stderr)
        print(f"  t_eval-only        failures  : {n_teval_fail}/{len(_captures)}", file=sys.stderr)
        trunc_idxs = [r["trunc_idx"] for r in _captures]
        print(f"  --- physical truncation index (rows actually consumed) ---", file=sys.stderr)
        print(f"  trunc_idx: min={min(trunc_idxs)} max={max(trunc_idxs)} "
              f"(slice has {_captures[0]['n_pts']} pts; rows past trunc are discarded)",
              file=sys.stderr)
        print(f"  --- value agreement over PHYSICALLY-USED rows [0:trunc+1] ---", file=sys.stderr)
        print(f"  WORST max_rel_diff n   : {max(rel_n):.3e}" if rel_n else
              "  WORST max_rel_diff n   : (no comparable calls)", file=sys.stderr)
        print(f"  WORST max_rel_diff phi : {max(rel_phi):.3e}" if rel_phi else
              "  WORST max_rel_diff phi : (no comparable calls)", file=sys.stderr)
        print(f"  WORST max_rel_diff tau : {max(rel_tau):.3e}" if rel_tau else
              "  WORST max_rel_diff tau : (no comparable calls)", file=sys.stderr)
        ep_n = _vals("endpoint_rel_diff_n")
        print(f"  WORST endpoint_rel_diff n : {max(ep_n):.3e}" if ep_n else
              "  WORST endpoint_rel_diff n : (none)", file=sys.stderr)
        print(f"  --- LSODA chatter lines (fd-level) ---", file=sys.stderr)
        print(f"  odeint={od_lsoda}  solve_ivp={iv_lsoda}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)


if __name__ == "__main__":
    main()
