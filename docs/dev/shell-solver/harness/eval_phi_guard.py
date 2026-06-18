#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantify the V2 phi-guard shell-ODE fix across regimes AND phases (production untouched).

The shell-ODE RHS (``trinity/shell_structure/get_shellODE.py``) carries a finite-radius
POLE in its ``+nShell**2`` recombination term (lines 97,100). Past the ionization front
the integrated state runs to ``inf`` -- but shell_structure truncates each 1000-pt slice
at the first ``phi<=1e-9`` / mass-limited row (``shell_structure.py:181-183``), so the
overflow lives entirely in the DISCARDED post-front tail while LSODA floods ``t+h=t``.

V2 phi-guard (``get_shellODE_phiguard``) freezes the derivatives once integrated ``phi<=0``
so the integrator never grinds into the pole. This harness MONKEYPATCHES the production RHS
to phi-guard (so the HOST run consumes the fix) and captures shell solves PER PHASE, running
baseline (prod RHS) + phi-guard side by side and comparing n/phi/tau over the physically
USED region (``_phys_cutoff``).

PHASE COVERAGE (the point of this rewrite): captures up to ``N_ENERGY`` ionised solves in the
energy phase AND ``N_IMPLICIT`` in the implicit phase (defaults 20 / 100, matching
capture_replay_variants.py's matrix mode), keying off ``params['current_phase'].value``. It
does NOT stop at a flat cap in the energy phase. Per-phase accuracy/speed are reported so the
implicit phase (>=50 solves) is covered, not just the early energy phase.

REPRODUCE
---------
    cd /home/user/trinity
    N_ENERGY=20 N_IMPLICIT=100 python docs/dev/shell-solver/harness/eval_phi_guard.py
    python docs/dev/shell-solver/harness/eval_phi_guard.py sfe0.6     # one config

Writes docs/dev/shell-solver/data/eval_phi_guard.csv (one row per config). Authored
2026-06-18; python 3.11, numpy 1.26.4, scipy 1.17.1.
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
from collections import Counter

import numpy as np
import scipy.integrate

HARNESS_DIR = Path(__file__).resolve().parent
TRINITY_ROOT = HARNESS_DIR.parents[3]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

DATA_DIR = TRINITY_ROOT / "docs" / "dev" / "shell-solver" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "eval_phi_guard.csv"

# Load the canonical variant module (hyphenated dir -> import by path).
_spec = importlib.util.spec_from_file_location(
    "get_shellODE_variants", HARNESS_DIR / "get_shellODE_variants.py")
_variants = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_variants)

# True production RHS, grabbed ONCE before any monkeypatch (the canonical phi-guard
# re-imports get_shellODE at call time -> recursion once we patch the module attr;
# bind the unpatched function here and wrap it recursion-safe).
import trinity.shell_structure.get_shellODE as _god  # noqa: E402
_PROD_RHS = _god.get_shellODE


def get_shellODE_phiguard(y, r, f_cover, is_ionised, params):
    """V2 phi-guard (recursion-safe): freeze derivatives once integrated phi<=0."""
    if is_ionised and y[1] <= 0.0:
        return 0.0, 0.0, 0.0
    return _PROD_RHS(y, r, f_cover, is_ionised, params)


# Per-phase capture targets (matches capture_replay_variants.py matrix mode).
N_ENERGY = int(os.environ.get("N_ENERGY", "20"))
N_IMPLICIT = int(os.environ.get("N_IMPLICIT", "100"))
_PHASE_N = {"energy": N_ENERGY, "implicit": N_IMPLICIT}
_TARGET_PHASES = [p for p, n in _PHASE_N.items() if n > 0]
HARD_TIMEOUT_S = float(os.environ.get("HARD_TIMEOUT_S", "900"))
TIMING_REPS = int(os.environ.get("TIMING_REPS", "3"))
PHI_THRESH = 1e-9                                # shell_structure.py:182 phiCondition

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
    """Index of the physically-USED prefix (phi-depletion / first non-finite row)."""
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
    """Max rel diff of n/phi/tau over the USED prefix (baseline vs phi-guard)."""
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


def _nonfinite_tail(y):
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


def _current_phase(args):
    """The host run's evolution phase (energy/implicit/transition/momentum)."""
    params = args[2] if len(args) >= 3 else None
    if params is not None:
        try:
            return params["current_phase"].value
        except Exception:
            return ""
    return ""


class _Evaluator:
    """Per-config, PER-PHASE capture-and-compare state."""

    def __init__(self):
        self.phase_counts = Counter()
        self.rel = {p: {"n": 0.0, "phi": 0.0, "tau": 0.0} for p in _PHASE_N}
        self.overflow_warns_total = 0
        self.nonfinite_tail_solves = 0
        self.ms_guard = {p: [] for p in _PHASE_N}
        self.ms_base = {p: [] for p in _PHASE_N}
        self.start = None

    @property
    def n_solves(self):
        return int(sum(self.phase_counts.values()))

    def all_targets_met(self):
        return all(self.phase_counts[p] >= _PHASE_N[p] for p in _TARGET_PHASES)

    def capture(self, y0, t, args, kwargs, phase):
        """Baseline (prod RHS) + phi-guard on one solve; tally under `phase`. Returns
        the phi-guard solution so the HOST run consumes the fix."""
        # PHI-GUARD: what the host run integrates (capture its flood + tail state).
        with _fd_capture() as guard_chatter:
            np.seterr(over="warn")
            od_guard = _REAL_ODEINT(get_shellODE_phiguard, y0, t, args=args, **kwargs)
        self.overflow_warns_total += _count_overflow_lines(guard_chatter())

        # BASELINE: production RHS, for the used-region accuracy reference.
        with _fd_capture():
            np.seterr(over="warn")
            od_base = _REAL_ODEINT(_PROD_RHS, y0, t, args=args, **kwargs)

        n_state = np.asarray(od_base).shape[1]
        rel = _used_rel(od_base, od_guard, n_state)
        for k in ("n", "phi", "tau"):
            v = rel.get(k)
            if v is not None and np.isfinite(v):
                self.rel[phase][k] = max(self.rel[phase][k], v)
        if _nonfinite_tail(od_guard):
            self.nonfinite_tail_solves += 1

        self.ms_base[phase].append(
            _time_call(lambda: _REAL_ODEINT(_PROD_RHS, y0, t, args=args, **kwargs)) * 1e3)
        self.ms_guard[phase].append(
            _time_call(lambda: _REAL_ODEINT(get_shellODE_phiguard, y0, t,
                                            args=args, **kwargs)) * 1e3)
        self.phase_counts[phase] += 1
        return od_guard


_EVAL = None


def _patched_odeint(func, y0, t, args=(), **kwargs):
    """Per-phase capture; FEED the phi-guard solution back so the host run uses the fix
    and progresses through energy -> implicit. Neutral solves pass through guard."""
    global _EVAL
    if _EVAL.start is None:
        _EVAL.start = time.time()
    is_ionised = bool(args[1]) if len(args) >= 2 else False
    if not is_ionised:
        return _REAL_ODEINT(get_shellODE_phiguard, y0, t, args=args, **kwargs)

    if _EVAL.all_targets_met():
        raise _CaptureDone()
    if (time.time() - _EVAL.start) > HARD_TIMEOUT_S:
        raise _HostTimeout(f"{_EVAL.n_solves} captures after {HARD_TIMEOUT_S:.0f}s")

    phase = _current_phase(args)
    if phase in _PHASE_N and _EVAL.phase_counts[phase] < _PHASE_N[phase]:
        return _EVAL.capture(y0, t, args, kwargs, phase)
    # phase target met, or an untargeted phase (transition/momentum): run guard, no capture.
    return _REAL_ODEINT(get_shellODE_phiguard, y0, t, args=args, **kwargs)


def _make_param(value):
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


def _agg_rel(rel_by_phase, phases, key):
    vals = [rel_by_phase[p][key] for p in phases if p in rel_by_phase]
    return max(vals) if vals else 0.0


def _mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else np.nan


def run_config(name, value):
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
        notes = "targets met"
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
    n_en, n_im = e.phase_counts["energy"], e.phase_counts["implicit"]
    all_p = list(_PHASE_N)
    # overall (max over phases) + per-phase implicit accuracy (the previously-missing one)
    notes = (f"{notes}; phases en={n_en} im={n_im}; "
             f"implicit rel_n={e.rel['implicit']['n']:.2e} rel_tau={e.rel['implicit']['tau']:.2e}; "
             f"energy rel_n={e.rel['energy']['n']:.2e} rel_tau={e.rel['energy']['tau']:.2e}")
    return {
        "config": name,
        "idea": "phi_guard_v2",
        "n_solves": e.n_solves,
        "n_energy": n_en,
        "n_implicit": n_im,
        "used_rel_n_max": _agg_rel(e.rel, all_p, "n"),
        "used_rel_phi_max": _agg_rel(e.rel, all_p, "phi"),
        "used_rel_tau_max": _agg_rel(e.rel, all_p, "tau"),
        "implicit_rel_n_max": e.rel["implicit"]["n"],
        "implicit_rel_tau_max": e.rel["implicit"]["tau"],
        "overflow_warns_total": e.overflow_warns_total,
        "nonfinite_tail_solves": e.nonfinite_tail_solves,
        "ms_per_solve_mean": _mean([m for p in all_p for m in e.ms_guard[p]]),
        "ms_per_solve_baseline": _mean([m for p in all_p for m in e.ms_base[p]]),
        "notes": notes,
    }


_COLS = ["config", "idea", "n_solves", "n_energy", "n_implicit",
         "used_rel_n_max", "used_rel_phi_max", "used_rel_tau_max",
         "implicit_rel_n_max", "implicit_rel_tau_max",
         "overflow_warns_total", "nonfinite_tail_solves",
         "ms_per_solve_mean", "ms_per_solve_baseline", "notes"]


def _fmt(v):
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.6g}"
    return v


def main():
    sel = sys.argv[1] if len(sys.argv) > 1 else None
    configs = [(n, v) for n, v in CONFIGS if sel is None or n == sel]
    if not configs:
        print(f"unknown config '{sel}'; choices: {[n for n, _ in CONFIGS]}",
              file=sys.stderr)
        sys.exit(2)

    print("=" * 72, file=sys.stderr)
    print(f"phi-guard (V2) PER-PHASE eval  python {sys.version.split()[0]} "
          f"numpy {np.__version__} scipy {scipy.__version__}", file=sys.stderr)
    print(f"  configs: {[n for n, _ in configs]}  N_ENERGY={N_ENERGY} "
          f"N_IMPLICIT={N_IMPLICIT}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    rows = []
    for name, value in configs:
        t0 = time.time()
        row = run_config(name, value)
        dt = time.time() - t0
        rows.append(row)
        print(f"[{name:20s}] en={row['n_energy']:3d} im={row['n_implicit']:3d} "
              f"floods={row['overflow_warns_total']:4d} nf={row['nonfinite_tail_solves']:3d} "
              f"impl_rel_n={row['implicit_rel_n_max']:.2e} impl_rel_tau={row['implicit_rel_tau_max']:.2e} "
              f"ms {row['ms_per_solve_mean']:.2f}/{row['ms_per_solve_baseline']:.2f} "
              f"({dt:.0f}s)", file=sys.stderr, flush=True)

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
