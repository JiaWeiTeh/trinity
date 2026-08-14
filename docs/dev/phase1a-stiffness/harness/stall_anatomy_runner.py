#!/usr/bin/env python3
"""Batch 2 (docs/dev/phase1a-stiffness/PLAN.md §6): inside the stalling segment —
is it stiffness, or a singularity?

Batch 1 established *that* one phase-1a ``solve_ivp`` call never returns. This
samples the inside of that call, because the two candidate causes need opposite
remedies (PLAN §5's trap) and are distinguishable by measurement:

  stiffness   the RHS stays finite and smooth; the solver crawls because an
              explicit method's stability limit h < ~2.8/|lambda| is tiny.
              Integration time keeps advancing, |lambda| is large but roughly
              steady, and an implicit method (LSODA/Radau) would step over it.
  singularity the solution ceases to exist at some t* inside the span: the
              solver's reached time ASYMPTOTES to t* while |RHS| and |lambda|
              diverge. No method helps -- implicit solvers grind here too, so
              the remedy is to stop the segment, not to change the solver.

Recorded every ``--sample-every`` RHS evaluations, flushed per row so the
external ``timeout`` kill cannot lose the record:

  * reached time ``t_hi`` and progress fraction through the segment span,
  * a step-size estimate (spread of the last 7 evaluations = one RK45 attempt),
  * the state (R2, v2, Eb) and the RHS vector at that state,
  * eigenvalues of a finite-difference Jacobian of the 3-state RHS, giving
    |lambda|_max, the stiffness ratio, and the implied RK45 stability step,
  * the fraction of wall time spent inside the RHS (separates "millions of
    cheap evals" from "few evals that each got expensive" -- phase 1a's RHS
    calls ``solve_R1``, a root-find, so that distinction is not academic).

Production source is NOT modified; the RHS is wrapped at runtime. Always
wall-cap from outside -- by construction this run does not finish:

    timeout 900 python docs/dev/phase1a-stiffness/harness/stall_anatomy_runner.py \\
        --config f1edge_hidens --workdir <dir>

Writes ``stall_anatomy.csv`` into the workdir; copy it to ``data/`` to commit.
"""
import argparse
import csv
import importlib.util
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))

COLS = ["segment", "eval_n", "wall_s", "rhs_wall_frac", "t_now", "t_hi",
        "progress_frac", "h_est", "R2_pc", "v2", "Eb_au",
        "dR2dt", "dv2dt", "dEbdt", "abs_rhs",
        "lambda_max_abs", "lambda_max_re", "stiff_ratio", "h_stable_rk45"]


def load_screen():
    spec = importlib.util.spec_from_file_location(
        "screen", os.path.join(REPO, "docs", "dev", "screen", "screen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="f1edge_hidens")
    p.add_argument("--stop-t", type=float, default=0.003)
    p.add_argument("--sample-every", type=int, default=500,
                   help="RHS evaluations between samples")
    p.add_argument("--workdir", required=True)
    a = p.parse_args()

    screen = load_screen()
    workdir = os.path.abspath(a.workdir)
    os.makedirs(workdir, exist_ok=True)
    param = os.path.join(workdir, "p.param")
    screen.write_param(screen.CONFIGS[a.config], param, a.stop_t, "screen")
    os.chdir(workdir)

    sys.path.insert(0, REPO)
    from trinity._input import read_param
    from trinity._functions.logging_setup import setup_logging
    import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
    import trinity.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
    import scipy.integrate

    params = read_param.read_param(param)

    # --- the ablation that reproduces the stall (same as e8b / Batch 1) ---
    _orig_pressure = get_bubbleParams.get_effective_bubble_pressure

    def no_ramp(*args, **kw):
        kw["t"] = None
        return _orig_pressure(*args, **kw)

    get_bubbleParams.get_effective_bubble_pressure = no_ramp

    fh = open(os.path.join(workdir, "stall_anatomy.csv"), "w", buffering=1, newline="")
    w = csv.writer(fh)
    w.writerow(COLS)

    st = {"seg": 0, "n": 0, "t0": 0.0, "t1": 0.0, "t_hi": -1e300,
          "wall0": 0.0, "rhs_wall": 0.0, "recent": deque(maxlen=7), "busy": False}
    _orig_rhs = energy_phase_ODEs.get_ODE_Edot_pure

    def jacobian(t, y, snapshot, prm):
        """Finite-difference Jacobian of the 3-state RHS at (t, y)."""
        f0 = np.asarray(_orig_rhs(t, list(y), snapshot, prm), dtype=float)
        J = np.zeros((3, 3))
        for j in range(3):
            dy = 1e-6 * abs(y[j]) or 1e-12
            yp = list(y)
            yp[j] += dy
            J[:, j] = (np.asarray(_orig_rhs(t, yp, snapshot, prm), dtype=float) - f0) / dy
        return f0, J

    def rhs(t, y, snapshot, prm):
        if st["busy"]:                      # Jacobian probes must not recurse
            return _orig_rhs(t, y, snapshot, prm)
        t_rhs = time.monotonic()
        out = _orig_rhs(t, y, snapshot, prm)
        st["rhs_wall"] += time.monotonic() - t_rhs
        st["n"] += 1
        st["t_hi"] = max(st["t_hi"], float(t))
        st["recent"].append(float(t))
        if st["n"] % a.sample_every == 0:
            st["busy"] = True
            try:
                f0, J = jacobian(t, y, snapshot, prm)
                ev = np.linalg.eigvals(J)
                lam_abs = float(np.max(np.abs(ev)))
                lam_re = float(ev[int(np.argmax(np.abs(ev)))].real)
                nz = np.abs(ev)[np.abs(ev) > 0]
                ratio = float(nz.max() / nz.min()) if len(nz) else float("nan")
                wall = time.monotonic() - st["wall0"]
                span = st["t1"] - st["t0"]
                w.writerow([
                    st["seg"], st["n"], f"{wall:.2f}",
                    f"{st['rhs_wall'] / wall:.4f}" if wall > 0 else "",
                    repr(float(t)), repr(st["t_hi"]),
                    f"{(st['t_hi'] - st['t0']) / span:.6e}" if span else "",
                    repr(max(st["recent"]) - min(st["recent"])),
                    repr(float(y[0])), repr(float(y[1])), repr(float(y[2])),
                    repr(float(f0[0])), repr(float(f0[1])), repr(float(f0[2])),
                    repr(float(np.linalg.norm(f0))),
                    repr(lam_abs), repr(lam_re), repr(ratio),
                    repr(2.8 / lam_abs) if lam_abs > 0 else "",
                ])
            except Exception as e:                     # never kill the run to log
                w.writerow([st["seg"], st["n"], "", "", repr(float(t))]
                           + [""] * 13 + [f"jac-failed:{type(e).__name__}"])
            finally:
                st["busy"] = False
        return out

    energy_phase_ODEs.get_ODE_Edot_pure = rhs

    _orig_ivp = scipy.integrate.solve_ivp

    def wrapped_ivp(fun, t_span, y0, **kw):
        if sys._getframe(1).f_globals.get("__name__", "").endswith("run_energy_phase"):
            st.update(seg=st["seg"] + 1, n=0, rhs_wall=0.0,
                      t0=float(t_span[0]), t1=float(t_span[1]),
                      t_hi=-1e300, wall0=time.monotonic())
            st["recent"].clear()
        return _orig_ivp(fun, t_span, y0, **kw)

    scipy.integrate.solve_ivp = wrapped_ivp

    setup_logging(params["log_level"].value, False, params["path2output"].value,
                  log_file_name="trinity.log", use_colors=False)
    import logging
    logging.getLogger(__name__).info(
        f"stall_anatomy_runner: config={a.config} ramp ABLATED, sampling every "
        f"{a.sample_every} RHS evals; production source unmodified")

    from trinity import main as trinity_main
    trinity_main.start_expansion(params)


if __name__ == "__main__":
    sys.exit(main())
