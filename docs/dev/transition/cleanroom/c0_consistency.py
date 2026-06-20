#!/usr/bin/env python3
"""C0.2 substrate-certification gate: betadelta (beta, delta) <-> trajectory consistency.

Clean-room redo of the implicit->momentum transition study (see PLAN.md). This
harness certifies, independently of the transition trigger, that the solver's
(beta, delta) outputs are consistent with the integrated trajectory it produced,
via the *definitions* the code enforces (get_betadelta.py:248,294):

    beta  = -(t/Pb)(dPb/dt)   =>  predicted dPb/dt = -beta * Pb / t
    delta =  (t/T )(dT /dt)   =>  predicted dT0/dt =  delta * T0 / t

We finite-difference the stored Pb(t), T0(t) across consecutive implicit-phase
snapshots and compare to the predictions from the stored beta, delta. Pure
diagnostic: nothing in trinity/ is modified, production is untouched.

Usage:
    # analyze an existing run's snapshots:
    python c0_consistency.py path/to/dictionary.jsonl [--out data/foo.csv]
    # run a config (hybr) then analyze:
    python c0_consistency.py param/simple_cluster.param --stop-t 0.5 --out data/foo.csv

Pre-registered bars (PLAN.md S2 C0.2): median relative residual <= 5% for BOTH
dPb/dt and dT0/dt over implicit-phase rows, and the median shrinks under timestep
refinement (consistency error, not a systematic offset).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:
        return "unknown"


def load_implicit_rows(jsonl_path: str) -> list[dict]:
    """Load implicit-phase snapshots, sorted by t_now."""
    rows = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("current_phase") != "implicit":
                continue
            rows.append(d)
    rows.sort(key=lambda d: d["t_now"])
    return rows


def _finite(*vals) -> bool:
    return all(v is not None and v == v and abs(v) != float("inf") for v in vals)


def consistency(rows: list[dict]) -> tuple[list[dict], dict]:
    """Per-adjacent-pair relative residual of the beta<->Pb and delta<->T0 laws."""
    per_row = []
    for a, b in zip(rows, rows[1:]):
        t0, t1 = a["t_now"], b["t_now"]
        dt = t1 - t0
        if dt <= 0:
            continue
        rec = {"t_now": t0, "dt": dt}
        # beta law: dPb/dt = -beta*Pb/t  (evaluate prediction at the segment start)
        Pb0, Pb1, beta0 = a.get("Pb"), b.get("Pb"), a.get("cool_beta")
        if _finite(Pb0, Pb1, beta0, t0) and t0 > 0 and abs(Pb0) > 0:
            meas = (Pb1 - Pb0) / dt
            pred = -beta0 * Pb0 / t0
            rec["res_beta"] = abs(meas - pred) / (abs(pred) + abs(Pb0) / t0 * 1e-9 + 1e-300)
        # delta law: dT0/dt = delta*T0/t
        T0, T1, delta0 = a.get("T0"), b.get("T0"), a.get("cool_delta")
        if _finite(T0, T1, delta0, t0) and t0 > 0 and abs(T0) > 0:
            meas = (T1 - T0) / dt
            pred = delta0 * T0 / t0
            rec["res_delta"] = abs(meas - pred) / (abs(pred) + abs(T0) / t0 * 1e-9 + 1e-300)
        per_row.append(rec)

    def _stats(key):
        xs = [r[key] for r in per_row if key in r]
        if not xs:
            return None
        xs_sorted = sorted(xs)
        return {
            "n": len(xs),
            "median": statistics.median(xs),
            "p90": xs_sorted[min(len(xs) - 1, int(0.9 * len(xs)))],
            "max": max(xs),
        }

    summary = {"res_beta": _stats("res_beta"), "res_delta": _stats("res_delta"),
               "n_implicit_rows": len(rows)}
    return per_row, summary


def run_config(param_path: str, stop_t: float | None) -> str:
    """Run a config with betadelta_solver=hybr to a temp dir; return its dictionary.jsonl."""
    from trinity._input import read_param
    from trinity import main as trinity_main

    params = read_param.read_param(param_path)
    out_dir = tempfile.mkdtemp(prefix="c0_")
    params["path2output"].value = out_dir
    params["betadelta_solver"].value = "hybr"
    if stop_t is not None:
        params["stop_t"].value = stop_t
    try:
        trinity_main.start_expansion(params)
    except SystemExit:
        pass
    # find the dictionary.jsonl the run wrote
    hits = list(Path(out_dir).rglob("dictionary.jsonl"))
    if not hits:
        sys.exit(f"no dictionary.jsonl produced under {out_dir}")
    return str(hits[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help=".param config to run, or an existing dictionary.jsonl")
    ap.add_argument("--stop-t", type=float, default=None, help="override stop_t (Myr) when running a config")
    ap.add_argument("--out", default=None, help="write per-row residuals to this CSV")
    args = ap.parse_args()

    if args.target.endswith(".jsonl"):
        jsonl = args.target
        provenance = f"existing snapshots: {jsonl} (provenance not certified)"
    else:
        jsonl = run_config(args.target, args.stop_t)
        provenance = f"ran {args.target} (hybr, stop_t={args.stop_t}) @ {_git_sha()}"

    rows = load_implicit_rows(jsonl)
    per_row, summary = consistency(rows)

    print(f"# C0.2 beta/delta<->trajectory consistency")
    print(f"# {provenance}")
    print(f"# implicit-phase rows: {summary['n_implicit_rows']}")
    for key, label in [("res_beta", "beta<->dPb/dt"), ("res_delta", "delta<->dT0/dt")]:
        s = summary[key]
        if s is None:
            print(f"  {label:18s}: no valid pairs")
            continue
        bar = "PASS" if s["median"] <= 0.05 else "FAIL"
        print(f"  {label:18s}: n={s['n']:4d}  median={s['median']:.3%}  p90={s['p90']:.3%}  max={s['max']:.3%}  [median<=5%: {bar}]")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        cols = ["t_now", "dt", "res_beta", "res_delta"]
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in per_row:
                w.writerow({c: r.get(c) for c in cols})
        print(f"# wrote {len(per_row)} rows -> {out}")


if __name__ == "__main__":
    main()
