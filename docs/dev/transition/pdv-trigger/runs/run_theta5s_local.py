#!/usr/bin/env python3
"""In-container / no-HPC fallback runner for the theta5s matrix (81 arms).

The theta5s matrix is designed for Helix (run_theta5s.sbatch). This runner is the fallback when
HPC is unavailable: it drives the same 81 params locally with a small parallel pool, RESUMABLY
(skips any arm that already has a .exit_code), so it survives container restarts — re-launch and
it picks up where it left off. It writes the same .exit_code/.duration per arm as the sbatch, so
runs/harvest_theta_max.py reduces the outputs identically.

Order: fast-terminating configs first (dense collapse / PdV handoff), the §8d diffuse cliff
(large_diffuse, small_1e6) last, so the heartbeat harvest fills in the easy arms early.

Compliance (📏 protocol rule 2): an arm killed at --per-arm-timeout is NON-COMPLIANT (t_final<5
and not a physics end) — harvest_theta_max flags it; re-run at a longer limit before quoting theta.

    python runs/run_theta5s_local.py --out $WS/outputs/theta5s --workers 3 --per-arm-timeout 5400
    # resumable: re-run the same command after a restart; done arms are skipped.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PARAMS = HERE / "params" / "theta5s"

# fast -> slow (by expected terminate time): dense collapse & PdV handoff first, diffuse cliff last
_CONFIG_ORDER = [
    "fail_repro", "small_dense_highsfe", "normal_n1e3", "simple_cluster",
    "pl2_steep", "be_sphere", "midrange_pl0", "large_diffuse_lowsfe", "small_1e6",
]


def _fa_of(stem):
    tag = stem.rsplit("__", 1)[1]  # 'none' (fA=1 baseline) or 'fa<v>'
    return 1 if tag == "none" else int(tag[2:])


def _order_key(param_path):
    # config fast->slow, then HIGH-fA first: a strong boost fires (and terminates) early, so the
    # completable arms run before the slow low-fA / __none baselines (which grind the implicit
    # phase without early collapse and wall-kill in-container). Verified in-container: __none/__fa2
    # of dense configs hit exit 124 at 900s while __fa12+ complete in minutes. Baselines are the
    # normalization reference but are un-completable in-container regardless of order -> HPC.
    cfg = param_path.stem.rsplit("__", 1)[0]
    ci = _CONFIG_ORDER.index(cfg) if cfg in _CONFIG_ORDER else len(_CONFIG_ORDER)
    fa = _fa_of(param_path.stem)
    low = 1 if fa <= 2 else 0  # fA<=2 baselines grind the implicit phase -> run them LAST, globally,
    #                            so no slow baseline of an early config starves a completable high-fA
    #                            arm of a later one. Baselines are un-completable in-container anyway.
    return (low, ci, -fa, param_path.stem)


def run_arm(param_path, out_root, timeout, env):
    name = param_path.stem
    outdir = out_root / name
    if (outdir / ".exit_code").exists():
        return name, "skip", 0
    outdir.mkdir(parents=True, exist_ok=True)
    # rewrite path2output to the chosen out_root (params ship with a relative outputs/theta5s/…)
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
        code = 124  # wall-killed -> non-compliant, re-run longer before quoting theta
    dur = int(time.time() - t0)
    (outdir / ".exit_code").write_text(f"{code}\n")
    (outdir / ".duration").write_text(f"{dur}\n")
    return name, ("ok" if code == 0 else f"exit{code}"), dur


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output root (e.g. $WS/outputs/theta5s)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--per-arm-timeout", type=int, default=5400, help="seconds (default 90 min)")
    ap.add_argument("--summary", help="a committed theta5s_summary.csv; skip arms already in it "
                    "(restart resilience: /tmp outputs are wiped on restart but the summary "
                    "survives in git, so re-launching only runs the missing arms)")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    params = sorted(PARAMS.glob("*.param"), key=_order_key)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MPLBACKEND="Agg")

    harvested = set()
    if args.summary and os.path.exists(args.summary):
        import csv
        with open(args.summary) as fh:
            for r in csv.DictReader(x for x in fh if not x.lstrip().startswith("#")):
                # only a COMPLIANT arm counts as done (t_final>=5 or a physics end); a wall-killed
                # arm (exit 124) must re-run, so leave it in the todo list.
                try:
                    compliant = float(r.get("t_final") or 0) >= 5.0 or r.get("phase_final") not in (None, "", "implicit")
                except (ValueError, TypeError):
                    compliant = False
                if compliant:
                    harvested.add(r["run_name"])

    todo = [p for p in params
            if not (out_root / p.stem / ".exit_code").exists() and p.stem not in harvested]
    print(f"[theta5s-local] {len(params)} arms, {len(todo)} to run "
          f"({len(params) - len(todo)} already done), {args.workers} workers", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_arm, p, out_root, args.per_arm_timeout, env): p for p in todo}
        for fut in as_completed(futs):
            name, status, dur = fut.result()
            done += 1
            print(f"[theta5s-local] {done}/{len(todo)}  {name:36s} {status:8s} {dur}s", flush=True)
    print(f"[theta5s-local] finished: {done} arms ran this pass", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
