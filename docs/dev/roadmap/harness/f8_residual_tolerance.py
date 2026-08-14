#!/usr/bin/env python3
"""solver-audit F8: is `RESIDUAL_THRESHOLD = 1e-4` calibrated, or just inherited?

Two modes, neither of which needs a new simulation to answer the first question.

``--survey`` reads finished run directories and reports the distribution of
`betadelta_total_residual` at accepted points (the field is already written to
every phase-1b snapshot). That answers "is the threshold binding or slack".

``--tighten F`` re-runs one config with the threshold scaled by F (and the
grid early-exit kept at threshold/10, the relation the source defines), so the
cost and trajectory effect of a tighter bar can be compared against a HEAD run
of the same config and `stop_t`.

What the constant means, read off the source rather than guessed: both
residuals are *relative* --
    Edot_residual = (Edot_from_beta - Edot_from_balance) / Edot_from_beta
    T_residual    = (T_bubble - T0) / T0
and acceptance is `Edot_residual**2 + T_residual**2 < RESIDUAL_THRESHOLD`. So
1e-4 is the square of 1e-2: **accept when the 2-norm of the relative-residual
vector is under 1%.** Dimensionless, and a statement about closure accuracy.

    python docs/dev/roadmap/harness/f8_residual_tolerance.py --survey <run_dir> ...
    python docs/dev/roadmap/harness/f8_residual_tolerance.py --tighten 0.01 \\
        --config simple_cluster --stop-t 0.02 --workdir <dir>
"""

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))


def load_screen():
    spec = importlib.util.spec_from_file_location(
        "screen", os.path.join(REPO, "docs", "dev", "screen", "screen.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def survey(run_dirs, threshold=1e-4):
    print(
        f"{'run':22s} {'1b':>4} {'conv':>5} {'worst':>10} {'median':>10} "
        f"{'worst/thr':>10} {'in[t/10,t)':>11}"
    )
    for d in run_dirs:
        path = os.path.join(d, "dictionary.jsonl")
        if not os.path.exists(path):
            print(f"{os.path.basename(d):22s} (missing)")
            continue
        rows = [json.loads(line) for line in open(path) if line.strip()]
        solves = [
            r
            for r in rows
            if r.get("current_phase") == "implicit"
            and r.get("betadelta_total_residual") is not None
        ]
        if not solves:
            print(f"{os.path.basename(d):22s} no phase-1b snapshots")
            continue
        res = sorted(r["betadelta_total_residual"] for r in solves)
        conv = sum(1 for r in solves if r.get("betadelta_converged"))
        band = sum(1 for x in res if threshold / 10 <= x < threshold)
        print(
            f"{os.path.basename(d):22s} {len(solves):4d} {conv:5d} {res[-1]:10.2e} "
            f"{res[len(res) // 2]:10.2e} {res[-1] / threshold:10.2f} {band:11d}"
        )


def tighten(factor, config, stop_t, workdir):
    screen = load_screen()
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    param = os.path.join(workdir, "p.param")
    screen.write_param(screen.CONFIGS[config], param, stop_t, "screen")
    os.chdir(workdir)

    sys.path.insert(0, REPO)
    import logging
    from trinity._input import read_param
    from trinity._functions.logging_setup import setup_logging
    import trinity.phase1b_energy_implicit.get_betadelta as gbd

    old = gbd.RESIDUAL_THRESHOLD
    gbd.RESIDUAL_THRESHOLD = old * factor
    # Keep the relation the source defines: the grid stops scanning once it finds
    # a point a decade inside the acceptance bar.
    gbd.GRID_EARLY_EXIT_RESIDUAL = gbd.RESIDUAL_THRESHOLD / 10

    params = read_param.read_param(param)
    setup_logging(
        params["log_level"].value,
        False,
        params["path2output"].value,
        log_file_name="trinity.log",
        use_colors=False,
    )
    logging.getLogger(__name__).info(
        f"f8_residual_tolerance: RESIDUAL_THRESHOLD {old:.1e} -> "
        f"{gbd.RESIDUAL_THRESHOLD:.1e}, GRID_EARLY_EXIT_RESIDUAL "
        f"{gbd.GRID_EARLY_EXIT_RESIDUAL:.1e}; production source unmodified"
    )

    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params

    check = validate_gmc_from_params(params)
    if not check.valid:
        sys.exit(f"GMC validation failed: {check.errors}")

    from trinity import main as trinity_main

    try:
        trinity_main.start_expansion(params)
    finally:
        out = params["path2output"].value
        meta = os.path.join(out, "metadata.json")
        term = json.load(open(meta)).get("termination") if os.path.exists(meta) else {}
        print(
            f"{config} thr={gbd.RESIDUAL_THRESHOLD:.1e}: "
            f"{(term or {}).get('exit_code')} {(term or {}).get('outcome')}"
        )
        survey([out], threshold=gbd.RESIDUAL_THRESHOLD)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--survey", nargs="*", help="finished run dirs to summarise")
    p.add_argument("--tighten", type=float, help="scale RESIDUAL_THRESHOLD by this")
    p.add_argument("--config")
    p.add_argument("--stop-t", type=float, default=0.02)
    p.add_argument("--workdir")
    a = p.parse_args()
    if a.survey:
        return survey(a.survey)
    if a.tighten:
        return tighten(a.tighten, a.config, a.stop_t, a.workdir)
    p.error("pass --survey or --tighten")


if __name__ == "__main__":
    sys.exit(main())
