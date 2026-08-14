#!/usr/bin/env python3
"""Batch 1 (docs/dev/phase1a-stiffness/PLAN.md §6): what does one phase-1a
segment actually cost the integrator, and how close does production get to the
stall?

Wraps ``scipy.integrate.solve_ivp`` and records one row per *phase-runner* call
(phase 1a's RK45 segment solve, and 1b/1c/2's LSODA solves for contrast) —
step count, ``nfev``, wall time, span and entry state. Calls from the bubble
solver are ignored: this measures the segment integrator, not the whole run.
Production source is NOT modified; the wrapper is installed at runtime.

Two rows are written per call, ``enter`` then ``exit``. That is deliberate: a
stalling call never returns, so an ``enter`` with no matching ``exit`` IS the
finding, and it survives an external ``timeout`` kill because the file is
line-buffered.

``--ablate-ramp`` disables the ``dt_switchon`` R1 ramp exactly as
``docs/dev/magic-numbers/harness/switchon_probe_runner.py`` does (forwards
``t=None``), which reproduces the stall — the **positive control** the
production numbers are measured against.

Configs come from ``docs/dev/screen/screen.py``'s ``CONFIGS`` table so the two
harnesses cannot drift apart.

    # one production config (ramp active); phase 1a ends at TFINAL = 3e-3 Myr
    python docs/dev/phase1a-stiffness/harness/seg_stepcount_runner.py \\
        --config simple_cluster --stop-t 0.003 --workdir <dir>

    # the positive control: same config, ramp ablated, wall-capped from outside
    timeout 600 python docs/dev/phase1a-stiffness/harness/seg_stepcount_runner.py \\
        --config f1edge_hidens --stop-t 0.003 --ablate-ramp --workdir <dir>

    # merge finished run dirs into the committed ledger
    python docs/dev/phase1a-stiffness/harness/seg_stepcount_runner.py --reduce <dir> [<dir> ...]
"""
import argparse
import csv
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
LEDGER = os.path.join(HERE, "..", "data", "seg_stepcount.csv")
SUMMARY = os.path.join(HERE, "..", "data", "seg_stepcount_summary.csv")

# Only these callers are segment integrators; bubble_luminosity also calls
# solve_ivp (once per bubble solve) and would swamp the record.
PHASES = {
    "trinity.phase1_energy.run_energy_phase": "1a",
    "trinity.phase1b_energy_implicit.run_energy_implicit_phase": "1b",
    "trinity.phase1c_transition.run_transition_phase": "1c",
    "trinity.phase2_momentum.run_momentum_phase": "2",
}

COLS = ["event", "phase", "call", "method", "t0_Myr", "t1_Myr", "dt_Myr",
        "R2_pc", "v2", "Eb_au", "steps", "nfev", "njev", "nlu", "wall_s",
        "success", "status", "message"]


def load_screen():
    spec = importlib.util.spec_from_file_location(
        "screen", os.path.join(REPO, "docs", "dev", "screen", "screen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install_wrapper(csv_fh):
    """Record every phase-runner solve_ivp call into csv_fh."""
    import scipy.integrate

    orig = scipy.integrate.solve_ivp
    w = csv.writer(csv_fh)
    w.writerow(COLS)
    state = {"n": 0}

    def wrapped(fun, t_span, y0, **kw):
        caller = sys._getframe(1).f_globals.get("__name__", "")
        phase = PHASES.get(caller)
        if phase is None:                      # bubble solver etc. -- not ours
            return orig(fun, t_span, y0, **kw)
        state["n"] += 1
        n = state["n"]
        t0, t1 = float(t_span[0]), float(t_span[1])
        head = [phase, n, kw.get("method", "RK45"), repr(t0), repr(t1),
                repr(t1 - t0)] + [repr(float(v)) for v in list(y0)[:3]]
        w.writerow(["enter"] + head + [""] * 8)
        csv_fh.flush()
        t_wall = time.monotonic()
        sol = orig(fun, t_span, y0, **kw)
        wall = time.monotonic() - t_wall
        w.writerow(["exit"] + head + [
            len(sol.t) - 1, sol.nfev, getattr(sol, "njev", ""),
            getattr(sol, "nlu", ""), f"{wall:.3f}", sol.success, sol.status,
            str(sol.message)[:60]])
        csv_fh.flush()
        return sol

    scipy.integrate.solve_ivp = wrapped


def run(args):
    screen = load_screen()
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    param = os.path.join(workdir, "p.param")
    screen.write_param(screen.CONFIGS[args.config], param, args.stop_t, "screen")
    os.chdir(workdir)

    sys.path.insert(0, REPO)
    from trinity._input import read_param
    from trinity._functions.logging_setup import setup_logging
    import trinity.bubble_structure.get_bubbleParams as get_bubbleParams

    params = read_param.read_param(param)

    if args.ablate_ramp:
        _orig = get_bubbleParams.get_effective_bubble_pressure

        def no_ramp(*a, **kw):
            kw["t"] = None                      # the ramp branch needs t
            return _orig(*a, **kw)

        get_bubbleParams.get_effective_bubble_pressure = no_ramp

    csv_fh = open(os.path.join(workdir, "seg_calls.csv"), "w", buffering=1, newline="")
    install_wrapper(csv_fh)

    setup_logging(params["log_level"].value, False, params["path2output"].value,
                  log_file_name="trinity.log", use_colors=False)
    import logging
    logging.getLogger(__name__).info(
        f"seg_stepcount_runner: config={args.config} stop_t={args.stop_t} "
        f"ablate_ramp={args.ablate_ramp}; production source unmodified")

    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
    check = validate_gmc_from_params(params)
    if not check.valid:
        sys.exit(f"GMC validation failed: {check.errors}")

    from trinity import main as trinity_main
    trinity_main.start_expansion(params)


def reduce(dirs):
    """Merge run dirs into the committed ledger, one row per phase-1a segment."""
    rows = []
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        path = os.path.join(d, "seg_calls.csv")
        if not os.path.exists(path):
            print(f"skip (no seg_calls.csv): {d}")
            continue
        with open(path) as fh:
            recs = list(csv.DictReader(fh))
        enters = [r for r in recs if r["event"] == "enter"]
        exits = {r["call"]: r for r in recs if r["event"] == "exit"}
        for e in enters:
            x = exits.get(e["call"])
            rows.append({
                "run": name, "phase": e["phase"], "call": e["call"],
                "method": e["method"], "t0_Myr": e["t0_Myr"], "dt_Myr": e["dt_Myr"],
                "Eb_au": e["Eb_au"], "R2_pc": e["R2_pc"],
                "steps": x["steps"] if x else "STALLED",
                "nfev": x["nfev"] if x else "STALLED",
                "wall_s": x["wall_s"] if x else "STALLED",
                "success": x["success"] if x else "",
            })
    out = os.path.normpath(LEDGER)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        fh.write("# phase1a-stiffness Batch 1 (PLAN.md par.6): per-segment cost of each phase's "
                 "solve_ivp call.\n")
        fh.write("# One row per call. steps/nfev/wall_s = STALLED means the call never returned "
                 "(the run was wall-capped) -- that IS the finding.\n")
        fh.write("# Command: python docs/dev/phase1a-stiffness/harness/seg_stepcount_runner.py "
                 "--config <name> --stop-t 0.003 [--ablate-ramp] --workdir <dir>, then --reduce "
                 "<dirs>. Configs from docs/dev/screen/screen.py CONFIGS.\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} rows from {len(dirs)} run(s)")
    _summarise(rows)


def _summarise(rows):
    """Per-run/per-phase aggregates — the table the write-up quotes, regenerated
    from the ledger rather than hand-copied."""
    import statistics as stat

    out, seen = [], []
    for r in rows:
        key = (r["run"], r["phase"])
        if key not in seen:
            seen.append(key)
    for run, phase in seen:
        grp = [r for r in rows if r["run"] == run and r["phase"] == phase]
        ok = [r for r in grp if r["steps"] != "STALLED"]
        stalled = len(grp) - len(ok)
        steps = [int(r["steps"]) for r in ok]
        wall = [float(r["wall_s"]) for r in ok]
        out.append({
            "run": run, "phase": phase, "calls": len(grp),
            "stalled_calls": stalled,
            "median_steps": f"{stat.median(steps):.0f}" if steps else "",
            "max_steps": max(steps) if steps else "",
            "max_nfev": max(int(r["nfev"]) for r in ok) if ok else "",
            "max_wall_s": f"{max(wall):.3f}" if wall else "",
            "total_wall_s": f"{sum(wall):.2f}" if wall else "",
        })
    path = os.path.normpath(SUMMARY)
    with open(path, "w", newline="") as fh:
        fh.write("# phase1a-stiffness Batch 1 summary, aggregated from seg_stepcount.csv by\n"
                 "# seg_stepcount_runner.py --reduce. One row per run+phase.\n"
                 "# stalled_calls counts solve_ivp calls that never returned (wall-capped run).\n")
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"{path}: {len(out)} rows")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="a key of docs/dev/screen/screen.py CONFIGS")
    p.add_argument("--stop-t", type=float, default=0.003,
                   help="Myr; 0.003 covers all of phase 1a (TFINAL_ENERGY_PHASE)")
    p.add_argument("--ablate-ramp", action="store_true",
                   help="disable the dt_switchon R1 ramp (the positive control)")
    p.add_argument("--workdir", default=os.path.join(REPO, "outputs", "seg_stepcount"))
    p.add_argument("--reduce", nargs="+", metavar="DIR",
                   help="merge finished run dirs into the committed ledger")
    a = p.parse_args()
    if a.reduce:
        return reduce(a.reduce)
    if not a.config:
        p.error("--config is required (or use --reduce)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
