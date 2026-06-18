#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantify the V2 phi-guard shell-ODE fix across regimes (production untouched).

WHAT THIS DOES
--------------
The shell-ODE RHS (``trinity/shell_structure/get_shellODE.py``) carries a
finite-radius POLE in its ``+nShell**2`` recombination term (lines 97,100). Past
the ionization front the integrated state runs to ``inf`` -- but shell_structure
truncates each 1000-pt slice at the first ``phi<=1e-9`` / mass-limited row
(``shell_structure.py:182``), so the overflow lives entirely in the DISCARDED
post-front tail while LSODA still floods ``t+h=t`` warnings.

V2 phi-guard (``get_shellODE_phiguard`` in get_shellODE_variants.py) freezes the
derivatives once the integrated ``phi<=0`` so the integrator never grinds into the
pole. This harness, per config:

  * MONKEYPATCHES the production RHS to phi-guard so the HOST run consumes the fix
    (``god.get_shellODE = get_shellODE_phiguard``) -- this is what clears the
    real-run flood.
  * MONKEYPATCHES ``scipy.integrate.odeint`` to capture each shell solve ONCE and
    run, side by side, the BASELINE (production RHS) and the PHI-GUARD solve, then
    compare n/phi/tau over the physically-USED region (``_phys_cutoff``: up to the
    first ``phi<=1e-9`` / first non-finite row -- everything production keeps).

PER CONFIG it records (aggregated): overflow_warns_total (host-run flood, target
0), nonfinite_tail_solves (target 0), used-region max rel diff of n/phi/tau vs
baseline (the phi-guard is NOT an identity -- this is the front shift), and
per-solve wall time phi-guard vs baseline odeint.

CONFIGS (degenerate -> realistic): simple_cluster, sfe0.6, probe_typical_hybr,
steep, dense_flat, mock_hybr.

REPRODUCE
---------
    cd /home/user/trinity
    python docs/dev/shell-solver/harness/eval_phi_guard.py            # all configs
    python docs/dev/shell-solver/harness/eval_phi_guard.py sfe0.6     # one config

Writes docs/dev/shell-solver/data/eval_phi_guard.csv (one row per config). Per
config we cap captures at CAP_N (default 40) ionized solves to keep wall time
bounded; flood/overflow totals are aggregated over the captured solves (the early
energy phase, where the pole is hottest). Authored 2026-06-18; python 3.11,
numpy 1.26.4, scipy 1.17.1.
"""

import os
import sys
import csv
import time
import tempfile
import warnings
import importlib.util
import contextlib
from pathlib import Path

import numpy as np
import scipy.integrate

HARNESS_DIR = Path(__file__).resolve().parent
TRINITY_ROOT = HARNESS_DIR.parents[3]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

DATA_DIR = TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "eval_phi_guard.csv"

# Load the canonical variant (hyphenated dir -> import by path, not package).
_spec = importlib.util.spec_from_file_location(
    "get_shellODE_variants", HARNESS_DIR / "get_shellODE_variants.py")
_variants = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_variants)

# True production RHS, grabbed ONCE before any monkeypatch. The canonical
# get_shellODE_phiguard re-imports get_shellODE.get_shellODE at call time, which
# becomes the phi-guard itself once we patch the module attr -> infinite
# recursion. We bind the unpatched production function here and wrap it so the
# guard's logic is identical to the variant but recursion-safe.
import trinity.shell_structure.get_shellODE as _god  # noqa: E402
_PROD_RHS = _god.get_shellODE


def get_shellODE_phiguard(y, r, f_cover, is_ionised, params):
    """V2 phi-guard (recursion-safe binding of the canonical variant): freeze the
    derivatives once integrated phi<=0; else delegate to the true production RHS."""
    if is_ionised and y[1] <= 0.0:
        return 0.0, 0.0, 0.0
    return _PROD_RHS(y, r, f_cover, is_ionised, params)

CAP_N = int(os.environ.get("CAP_N", "40"))      # ionized solves captured per config
HARD_TIMEOUT_S = float(os.environ.get("HARD_TIMEOUT_S", "240"))
TIMING_REPS = 5
PHI_THRESH = 1e-9                                # shell_structure.py:182 phiCondition

# Configs in degenerate -> realistic order. Value is either an SFE override (float)
# or an explicit .param path.
CONFIGS = [
    ("simple_cluster", None),
    ("sfe0.6", 0.6),
    ("probe_typical_hybr",
     TRINITY_ROOT / "docs/dev/archive/betadelta/diagnostics/probe_typical_hybr.param"),
    ("steep", TRINITY_ROOT / "docs/dev/transition/harness/steep.param"),
    ("dense_flat", TRINITY_ROOT / "docs/dev/transition/harness/dense_flat.param"),
    ("mock_hybr", TRINITY_ROOT / "docs/dev/transition/harness/mock_hybr.param"),
]

_REAL_ODEINT = scipy.integrate.odeint
_OVERFLOW_MARKERS = ("overflow", "t + h = t", "t+h=t", "excess work", "lsoda")


class _CaptureDone(Exception):
    pass


class _HostTimeout(Exception):
    pass


@contextlib.contextmanager
def _fd_capture():
    """Capture C-level stdout/stderr (LSODA writes there, not via warnings)."""
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


def _count_overflow_lines(text):
    if not text:
        return 0
    low = text.lower()
    return sum(1 for line in low.splitlines()
               if any(m in line for m in _OVERFLOW_MARKERS))


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


def _max_rel_diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(a), 1e-300)
    rel = np.abs(a - b) / denom
    both_tiny = (np.abs(a) < 1e-300) & (np.abs(b) < 1e-300)
    return float(np.max(np.where(both_tiny, 0.0, rel))) if a.size else np.nan


def _used_rel(od_base, y_guard, n_state):
    """Max rel diff of n/phi/tau over the USED prefix (baseline odeint vs phi-guard
    odeint), comparing only rows finite in both."""
    out = {"n": np.nan, "phi": np.nan, "tau": np.nan}
    od = np.asarray(od_base, dtype=float)
    yg = np.asarray(y_guard, dtype=float)
    m = min(len(od), len(yg))
    if m < 2 or yg.shape[1] != od.shape[1]:
        return out
    cut = min(_phys_cutoff(od, n_state), m)
    odc, ygc = od[:cut], yg[:cut]
    common = np.isfinite(odc).all(axis=1) & np.isfinite(ygc).all(axis=1)
    cols = ("n", "phi", "tau") if n_state == 3 else ("n", "tau")
    for j, name in enumerate(cols):
        a, b = odc[common, j], ygc[common, j]
        if a.size:
            out[name] = _max_rel_diff(a, b)
    return out


def _nonfinite_tail(y, n_state):
    """True if the solve output has ANY non-finite row. The phi-guard should
    produce none (it freezes the state at the front), whereas the baseline RHS
    overflows the discarded tail. n_state kept for signature symmetry."""
    del n_state
    return not np.isfinite(np.asarray(y, dtype=float)).all()


def _time_call(thunk):
    best = np.inf
    for _ in range(TIMING_REPS):
        t0 = time.perf_counter()
        try:
            thunk()
        except Exception:  # noqa: BLE001
            return np.nan
        best = min(best, time.perf_counter() - t0)
    return best


class _Evaluator:
    """Per-config capture-and-compare state."""

    def __init__(self):
        self.n_solves = 0
        self.rel_n, self.rel_phi, self.rel_tau = 0.0, 0.0, 0.0
        self.overflow_warns_total = 0
        self.nonfinite_tail_solves = 0
        self.ms_guard = []
        self.ms_base = []
        self.start = None

    def capture(self, y0, t, args, kwargs):
        """Run baseline (prod RHS) + phi-guard on one solve; tally. Returns the
        phi-guard solution so the HOST run consumes the fix (prod RHS is patched)."""
        prod_rhs = _PROD_RHS  # unpatched production function, grabbed at import

        # BASELINE: production RHS via real odeint (this floods + overflows the tail).
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with _fd_capture() as base_chatter:
                np.seterr(over="warn")
                od_base = _REAL_ODEINT(prod_rhs, y0, t, args=args, **kwargs)
        base_floods = _count_overflow_lines(base_chatter())

        # PHI-GUARD: the fix, via real odeint (what the host run actually integrates).
        with _fd_capture() as guard_chatter:
            np.seterr(over="warn")
            od_guard = _REAL_ODEINT(get_shellODE_phiguard, y0, t, args=args, **kwargs)
        guard_floods = _count_overflow_lines(guard_chatter())

        n_state = np.asarray(od_base).shape[1]
        rel = _used_rel(od_base, od_guard, n_state)
        for k, attr in (("n", "rel_n"), ("phi", "rel_phi"), ("tau", "rel_tau")):
            v = rel[k]
            if v is not None and np.isfinite(v):
                setattr(self, attr, max(getattr(self, attr), v))

        # The HOST run uses phi-guard, so the flood count that matters is guard's.
        self.overflow_warns_total += guard_floods
        if _nonfinite_tail(od_guard, n_state):
            self.nonfinite_tail_solves += 1

        self.ms_base.append(
            _time_call(lambda: _REAL_ODEINT(prod_rhs, y0, t, args=args, **kwargs)) * 1e3)
        self.ms_guard.append(
            _time_call(lambda: _REAL_ODEINT(get_shellODE_phiguard, y0, t,
                                            args=args, **kwargs)) * 1e3)
        self.n_solves += 1
        _ = base_floods  # baseline flood is the problem being fixed; not summed
        return od_guard


_EVAL = None


def _patched_odeint(func, y0, t, args=(), **kwargs):
    """Capture each ionized solve, run baseline+guard, and FEED the guard solution
    back so the host integration uses the fix. Neutral solves pass through guard."""
    global _EVAL
    if _EVAL.start is None:
        _EVAL.start = time.time()
    is_ionised = bool(args[1]) if len(args) >= 2 else False
    # Only the ionized branch carries the pole; capture those.
    if is_ionised and _EVAL.n_solves < CAP_N:
        if (time.time() - _EVAL.start) > HARD_TIMEOUT_S:
            raise _HostTimeout(
                f"{_EVAL.n_solves} captures after {HARD_TIMEOUT_S:.0f}s")
        return _EVAL.capture(y0, t, args, kwargs)
    if is_ionised and _EVAL.n_solves >= CAP_N:
        raise _CaptureDone()
    # Neutral region: just run the (patched, = phi-guard) RHS faithfully.
    return _REAL_ODEINT(func, y0, t, args=args, **kwargs)


def _make_param(value):
    """Return (param_path, is_temp). value: None=base, float=sfe override, Path=file."""
    if value is None:
        return TRINITY_ROOT / "param" / "simple_cluster.param", False
    if isinstance(value, (int, float)):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".param", delete=False)
        tmp.write(f"mCloud    1e5\nsfe    {value}\n")
        tmp.close()
        return Path(tmp.name), True
    return Path(value), False


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


def run_config(name, value):
    """Patch prod RHS -> phi-guard, drive the host run, capture+compare. Returns a
    result dict for the CSV row."""
    global _EVAL
    _EVAL = _Evaluator()
    param_path, is_temp = _make_param(value)
    notes = ""

    import trinity.shell_structure.get_shellODE as god
    god_orig = god.get_shellODE
    god.get_shellODE = get_shellODE_phiguard       # HOST run consumes the fix
    scipy.integrate.odeint = _patched_odeint
    try:
        _drive_host_run(param_path)
        notes = "host finished"
    except _CaptureDone:
        notes = f"cap {CAP_N} met"
    except _HostTimeout as exc:
        notes = f"timeout: {exc}"
    except SystemExit as exc:
        notes = f"sys.exit({exc.code})"
    except Exception as exc:  # noqa: BLE001
        notes = f"ERR {type(exc).__name__}: {str(exc)[:60]}"
    finally:
        scipy.integrate.odeint = _REAL_ODEINT
        god.get_shellODE = god_orig
        if is_temp:
            try:
                param_path.unlink()
            except OSError:
                pass

    e = _EVAL
    ms_guard = float(np.nanmean(e.ms_guard)) if e.ms_guard else np.nan
    ms_base = float(np.nanmean(e.ms_base)) if e.ms_base else np.nan
    return {
        "config": name,
        "idea": "phi_guard_v2",
        "n_solves": e.n_solves,
        "used_rel_n_max": e.rel_n,
        "used_rel_phi_max": e.rel_phi,
        "used_rel_tau_max": e.rel_tau,
        "overflow_warns_total": e.overflow_warns_total,
        "nonfinite_tail_solves": e.nonfinite_tail_solves,
        "ms_per_solve_mean": ms_guard,
        "ms_per_solve_baseline": ms_base,
        "notes": notes,
    }


_COLS = ["config", "idea", "n_solves", "used_rel_n_max", "used_rel_phi_max",
         "used_rel_tau_max", "overflow_warns_total", "nonfinite_tail_solves",
         "ms_per_solve_mean", "ms_per_solve_baseline", "notes"]


def _fmt(v):
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return f"{v:.6g}"
    return v


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else None
    configs = [(n, v) for n, v in CONFIGS if sel is None or n == sel]
    if not configs:
        print(f"unknown config '{sel}'; choices: {[n for n, _ in CONFIGS]}",
              file=sys.stderr)
        sys.exit(2)

    print("=" * 72, file=sys.stderr)
    print(f"phi-guard (V2) evaluation  python {sys.version.split()[0]} "
          f"numpy {np.__version__} scipy {scipy.__version__}", file=sys.stderr)
    print(f"  configs: {[n for n, _ in configs]}  CAP_N={CAP_N}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    rows = []
    for name, value in configs:
        t0 = time.time()
        row = run_config(name, value)
        dt = time.time() - t0
        rows.append(row)
        print(f"[{name:20s}] solves={row['n_solves']:3d} "
              f"floods={row['overflow_warns_total']:4d} "
              f"nf_tail={row['nonfinite_tail_solves']:3d} "
              f"rel_n={row['used_rel_n_max']:.2e} rel_tau={row['used_rel_tau_max']:.2e} "
              f"ms {row['ms_per_solve_mean']:.2f}/{row['ms_per_solve_baseline']:.2f} "
              f"({dt:.0f}s) [{row['notes']}]", file=sys.stderr, flush=True)

    # Merge into the CSV (overwrite rows for configs we just ran; keep others).
    existing = {}
    if OUT_CSV.exists():
        with open(OUT_CSV) as fh:
            for r in csv.DictReader(fh):
                existing[r["config"]] = r
    for r in rows:
        existing[r["config"]] = {k: _fmt(r[k]) for k in _COLS}
    order = [n for n, _ in CONFIGS]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        for n in order:
            if n in existing:
                w.writerow({k: existing[n].get(k, "") for k in _COLS})
    print(f"\nWrote {len(rows)} config rows -> {OUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
