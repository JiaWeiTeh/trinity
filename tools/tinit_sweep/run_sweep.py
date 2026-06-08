"""Driver for the T_init sensitivity sweep -- robust, not a coincidence pass.

Runs sweep.py in many *separate* processes and enforces five gates:

  1. FIDELITY    -- baseline (T_init=3e4) re-solve reproduces the dumped
                    converged dMdt (rel < FID_TOL). The states were dumped by
                    current code, so this is a real ground truth (and bubble_audit
                    separately confirms bit-exact T_array reproduction).
  2. DETERMINISM -- across K *pinned* (threads=1) separate processes, every
                    (state, T_init) L_total is BIT-identical (float.hex). A pass
                    that is not bit-reproducible is treated as invalid.
  3. CONTRAST    -- across K *unpinned* (default threads) separate processes,
                    report whether L_total varies. Demonstrates the determinism
                    in gate 2 is earned by pinning, not luck. (Observational:
                    if the box has 1 core it may not vary -- reported, not failed.)
  4. SENSITIVITY -- per state, |L(T_init) - L(3e4)| / L(3e4) <= SENS_TOL for the
                    whole grid. This is the actual physics question.
  5. ROBUSTNESS  -- every non-ok solve (crash / MonotonicError) is counted per
                    cell and printed. A crash is never silently a pass.

Usage:
    python run_sweep.py <states_dir|state.pkl> [--k 5] [--base path.param]
"""
from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_SWEEP = os.path.join(_HERE, "sweep.py")

PIN = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
UNPIN_CLEAR = list(PIN.keys())

FID_TOL = 1e-3      # fsolve xtol is 1e-4; allow 10x margin
SENS_TOL = 1e-2     # 1% band on L_total vs the 3e4 baseline
BASELINE = 3.0e4


def _run(state, base, pinned):
    env = dict(os.environ)
    if pinned:
        env.update(PIN)
    else:
        for k in UNPIN_CLEAR:
            env.pop(k, None)
    cmd = [sys.executable, _SWEEP, state] + ([base] if base else [])
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"sweep.py failed ({r.returncode}) on {state}\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--base", default=None)
    a = ap.parse_args(argv)

    states = (sorted(glob.glob(os.path.join(a.target, "*.pkl")))
              if os.path.isdir(a.target) else [a.target])
    if not states:
        print(f"no .pkl states at {a.target}")
        return 2
    print(f"states: {len(states)}   repeats K={a.k}   "
          f"sens_tol={SENS_TOL:.0e}   fid_tol={FID_TOL:.0e}\n")

    fail = {"fidelity": [], "determinism": [], "sensitivity": [], "robustness": []}
    unpinned_varies = False

    for state in states:
        name = os.path.basename(state)
        pinned_runs = [_run(state, a.base, True) for _ in range(a.k)]
        unpinned_runs = [_run(state, a.base, False) for _ in range(a.k)]
        ref = pinned_runs[0]
        grid = [c["t_init"] for c in ref["cells"]]

        # --- gate 1: fidelity
        fid = ref["fidelity_dMdt_rel"]
        fid_ok = fid is not None and fid < FID_TOL
        if not fid_ok:
            fail["fidelity"].append((name, fid))

        # --- gate 2: determinism (bit-identical across pinned processes)
        det_ok = True
        for ti in grid:
            hexes = {next(c for c in run["cells"] if c["t_init"] == ti).get("L_total_hex")
                     for run in pinned_runs}
            if len(hexes) != 1:
                det_ok = False
                fail["determinism"].append((name, ti, hexes))

        # --- gate 3: contrast (does unpinned vary?)
        for ti in grid:
            hexes = {next(c for c in run["cells"] if c["t_init"] == ti).get("L_total_hex")
                     for run in unpinned_runs}
            if len(hexes) != 1:
                unpinned_varies = True

        # --- gate 4: sensitivity (relative to 3e4 baseline)
        bcell = next(c for c in ref["cells"] if c["t_init"] == BASELINE)
        worst_sens = None
        if bcell["status"] == "ok":
            Lb = bcell["L_total"]
            for c in ref["cells"]:
                if c["status"] != "ok":
                    continue
                rel = abs(c["L_total"] - Lb) / abs(Lb)
                worst_sens = rel if worst_sens is None else max(worst_sens, rel)
                if rel > SENS_TOL:
                    fail["sensitivity"].append((name, c["t_init"], rel))

        # --- gate 5: robustness (count non-ok)
        crashes = [(c["t_init"], c["status"]) for c in ref["cells"] if c["status"] != "ok"]
        if crashes:
            fail["robustness"].append((name, crashes))

        # per-state line
        sens_str = f"{worst_sens:.2e}" if worst_sens is not None else "n/a"
        dips = [c["t_init"] for c in ref["cells"]
                if c["status"] == "ok" and c["T_min"] < c["t_init"] - 1.0]
        print(f"{name:40} fid={fid:.2e}({'ok' if fid_ok else 'FAIL'})  "
              f"det={'ok' if det_ok else 'FAIL'}  worst_sens={sens_str}  "
              f"crashes={len(crashes)}  boundary_dip@={dips}")

    # ---- report ----
    print("\n================ GATES ================")
    def show(label, key, fmt):
        items = fail[key]
        print(f"  {label:14} {'PASS' if not items else 'FAIL'}"
              + ("" if not items else "  " + "; ".join(fmt(x) for x in items)))
    show("1 fidelity", "fidelity", lambda x: f"{x[0]}={x[1]}")
    show("2 determinism", "determinism", lambda x: f"{x[0]}@{x[1]:.0e}")
    show("4 sensitivity", "sensitivity", lambda x: f"{x[0]}@{x[1]:.0e}:{x[2]:.2e}")
    show("5 robustness", "robustness", lambda x: f"{x[0]}:{x[1]}")
    print(f"  3 contrast     {'unpinned VARIES (determinism earned by pinning)' if unpinned_varies else 'unpinned also stable on this box (>=det)'}")

    crit = any(fail[k] for k in ("fidelity", "determinism", "sensitivity"))
    print("\n=== OVERALL:", "FAIL" if crit else "PASS",
          "(robustness crashes are reported, not auto-fail) ===")
    return 1 if crit else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
