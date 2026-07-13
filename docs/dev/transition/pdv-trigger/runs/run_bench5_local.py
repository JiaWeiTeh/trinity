#!/usr/bin/env python3
"""In-container / no-HPC runner for the bench5 matrix (Phase 5; 60 arms).

Adapted from run_theta5s_local.py (same resumable, restart-surviving design — HPC was down, so the
Lancaster-2021b calibration runs in-container like the theta5s matrix did). Drives the 60 bench5
params with a small parallel pool, RESUMABLY (skips any arm already compliant in the committed
summary OR carrying a local .exit_code), writing the same .exit_code/.duration markers.

Order (completable-first, the theta5s lesson): PRODUCTION arms before DIAGNOSTIC arms — a production
arm fires on cooling_balance and terminates at the transition, while a diagnostic arm
(transition_trigger=blowout) runs until R2 > rCloud or stop_t=5 Myr, so it is the long pole. Within
each, dense+high-f_A first (fires/blows out earliest), diffuse+low-f_A last.

Compliance (📏 rule 2): an arm killed at --per-arm-timeout is NON-COMPLIANT (t_final<5, not a physics
end) — leave it in the todo list and re-run longer before quoting any θ.

    python runs/run_bench5_local.py --out $WS/bench5_out --workers 3 --per-arm-timeout 5400 \
        --summary runs/data/bench5_summary.csv
    # resumable: re-run the same command after a restart; done arms are skipped.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PARAMS = HERE / "params" / "bench5"

# bench config fast->slow by rCloud (dense small clouds blow out / fire earliest).
_BENCH_ORDER = ["bench5_m5e5_r2p5", "bench4_m1e5_r2p5", "bench3_m1e5_r5",
                "bench2_m1e5_r10", "bench1_m5e4_r20"]


def _fa_of(stem):
    core = stem[:-5] if stem.endswith("_diag") else stem
    tag = core.rsplit("__", 1)[1]  # 'none' or 'fa<v>'
    return 1 if tag == "none" else int(tag[2:])


def _order_key(p):
    stem = p.stem
    diag = 1 if stem.endswith("_diag") else 0          # production (0) before diagnostic (1)
    core = stem[:-5] if diag else stem
    bench = core.rsplit("__", 1)[0]
    bi = _BENCH_ORDER.index(bench) if bench in _BENCH_ORDER else len(_BENCH_ORDER)
    fa = _fa_of(stem)
    low = 1 if fa <= 1 else 0                           # __none baselines grind longest -> last
    return (diag, low, bi, -fa, stem)


def run_arm(param_path, out_root, timeout, env):
    name = param_path.stem
    outdir = out_root / name
    if (outdir / ".exit_code").exists():
        return name, "skip", 0
    outdir.mkdir(parents=True, exist_ok=True)
    text = param_path.read_text()
    text = "\n".join(
        (f"path2output            {outdir}" if ln.strip().startswith("path2output") else ln)
        for ln in text.splitlines()
    ) + "\n"
    run_param = outdir / "run.param"
    run_param.write_text(text)
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, str(REPO / "run.py"), str(run_param)],
            cwd=out_root, env=env, timeout=timeout,
            stdout=open(outdir / "run.log", "w"), stderr=subprocess.STDOUT,
        )
        code = r.returncode
    except subprocess.TimeoutExpired:
        code = 124  # wall-killed -> non-compliant, re-run longer before quoting θ
    dur = int(time.time() - t0)
    (outdir / ".exit_code").write_text(f"{code}\n")
    (outdir / ".duration").write_text(f"{dur}\n")
    return name, ("ok" if code == 0 else f"exit{code}"), dur


def _done_in_summary(summary):
    done = set()
    if summary and os.path.exists(summary):
        with open(summary) as fh:
            for r in csv.DictReader(x for x in fh if not x.lstrip().startswith("#")):
                try:
                    compliant = (float(r.get("t_final") or 0) >= 5.0
                                 or r.get("phase_final") not in (None, "", "implicit", "energy"))
                except (ValueError, TypeError):
                    compliant = False
                if compliant:
                    done.add(r["run_name"])
    return done


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--per-arm-timeout", type=int, default=5400, help="seconds (default 90 min)")
    ap.add_argument("--summary", help="committed bench5_summary.csv; skip arms already compliant there")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    params = sorted(PARAMS.glob("*.param"), key=_order_key)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MPLBACKEND="Agg")

    harvested = _done_in_summary(args.summary)
    todo = [p for p in params
            if not (out_root / p.stem / ".exit_code").exists() and p.stem not in harvested]
    print(f"[bench5-local] {len(params)} arms, {len(todo)} to run "
          f"({len(params) - len(todo)} already done), {args.workers} workers", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_arm, p, out_root, args.per_arm_timeout, env): p for p in todo}
        for fut in as_completed(futs):
            name, status, dur = fut.result()
            done += 1
            print(f"[bench5-local] {done}/{len(todo)}  {name:34s} {status:8s} {dur}s", flush=True)
    print(f"[bench5-local] finished: {done} arms ran this pass", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
