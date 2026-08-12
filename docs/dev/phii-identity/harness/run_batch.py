#!/usr/bin/env python3
"""Run a phii-identity config matrix, one run per process, with wall times.

Implements the execution half of `docs/dev/phii-identity/PLAN.md` batches 0/1
under the contamination rules in PLAN §0:

  C-3  one run per process (never two solves in one interpreter)
  C-4  single param source — variants are materialised here from the committed
       base param plus an explicit override dict, and the overrides are written
       into the run's `_overrides.txt` and echoed into the wall-time CSV
  C-7  output dirs embed arm + code SHA, so a dictionary.jsonl can never be
       silently reused across a code change

Usage (from the repo root):
    python docs/dev/phii-identity/harness/run_batch.py --arm b0 --tier core
    python docs/dev/phii-identity/harness/run_batch.py --arm b1 --tier core
    python docs/dev/phii-identity/harness/run_batch.py --arm b0 --configs SC,PRB

`--arm` is a free label for the code state being measured (b0 = base, b1 = with
the shadow diagnostic). Runs land in `outputs/phii/<arm>__<sha>/<config>/`.
Already-complete runs are skipped unless --force, so an interrupted batch
resumes without redoing work.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stamp import code_version, stamp  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parents[1] / "data"

# PLAN §4. base = committed param, never edited; over = the ONLY deltas applied.
P_PERF = "docs/dev/performance"
P_BENCH = "docs/dev/transition/pdv-trigger/runs/params/bench5"
P_1A = "docs/dev/phase1a-init/harness/params"
P_CR = "docs/dev/transition/cleanroom/configs"

MATRIX = {
    # id:    (tier,   base param,                              overrides)
    "SC": ("core", "param/simple_cluster.param", {}),
    "F1LO": ("core", f"{P_PERF}/f1edge_lowdens_himass_hisfe.param", {}),
    "F1HI": ("core", f"{P_PERF}/f1edge_hidens_himass_losfe.param", {}),
    "B3M": ("core", f"{P_BENCH}/bench3_m1e5_r5__none_diag.param", {}),
    "PRB": ("core", f"{P_1A}/probe.param", {}),
    "WW": ("core", "param/simple_cluster.param", {"FB_thermCoeffWind": "0.1"}),
    "B1M": ("full", f"{P_BENCH}/bench1_m5e4_r20__none_diag.param", {}),
    "B2M": ("full", f"{P_BENCH}/bench2_m1e5_r10__none_diag.param", {}),
    "GMC": ("full", f"{P_1A}/gmc_control.param", {}),
    "BE": ("full", f"{P_CR}/be_sphere.param", {}),
    "PL2": ("full", f"{P_CR}/pl2_steep.param", {}),
    "LDLS": ("full", f"{P_CR}/large_diffuse_lowsfe.param", {}),
    "SDHS": ("full", f"{P_CR}/small_dense_highsfe.param", {}),
}


def select(tier, names):
    if names:
        missing = [n for n in names if n not in MATRIX]
        if missing:
            sys.exit(f"unknown config id(s): {missing}. Known: {sorted(MATRIX)}")
        return list(names)
    if tier == "all":
        return list(MATRIX)
    return [k for k, v in MATRIX.items() if v[0] == tier]


def materialise(cfg, out_dir, extra):
    """Write the run param: committed base + overrides. Returns (path, overrides)."""
    _, base_rel, over = MATRIX[cfg]
    over = {**over, **extra}
    base = (REPO / base_rel).read_text()
    # path2output/model_name are plumbing, not physics: force them so C-7 holds.
    forced = {"path2output": str(out_dir), "model_name": cfg}
    if "stop_t" in over:
        forced["stop_t"] = over.pop("stop_t")
    lines = []
    for line in base.splitlines():
        key = line.split("#", 1)[0].split()
        if key and key[0] in {**forced, **over}:
            continue  # drop the base's value; we re-emit below
        lines.append(line)
    lines.append(f"\n# --- materialised by run_batch.py from {base_rel} ---")
    for k, v in {**over, **forced}.items():
        lines.append(f"{k}    {v}")
    out_dir.mkdir(parents=True, exist_ok=True)
    param = out_dir / f"{cfg}.param"
    param.write_text("\n".join(lines) + "\n")
    (out_dir / "_overrides.txt").write_text(
        f"base: {base_rel}\noverrides: {over}\nforced: {forced}\n"
    )
    return param, over


def done(run_dir):
    d = run_dir / "dictionary.jsonl"
    return d.exists() and d.stat().st_size > 0 and (run_dir / "metadata.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="label for the code state, e.g. b0 / b1")
    ap.add_argument("--tier", default="core", choices=["core", "full", "all"])
    ap.add_argument("--configs", help="comma-separated ids, overrides --tier")
    ap.add_argument("--stop-t", help="override stop_t on every config (documented in the CSV)")
    ap.add_argument("--timeout", type=int, default=7200, help="per-run seconds")
    ap.add_argument("--force", action="store_true", help="re-run even if output exists")
    args = ap.parse_args()

    sha = code_version()
    root = REPO / "outputs" / "phii" / f"{args.arm}__{sha.replace('+', '_')}"
    names = select(args.tier, args.configs.split(",") if args.configs else None)
    extra = {"stop_t": args.stop_t} if args.stop_t else {}

    print(f"arm={args.arm}  code={sha}  configs={names}")
    print(f"root={root}\n")

    results = []
    for cfg in names:
        run_dir = root / cfg
        if done(run_dir) and not args.force:
            print(f"[skip] {cfg} — already complete")
            results.append((cfg, "skipped", "", ""))
            continue
        param, over = materialise(cfg, run_dir, extra)
        print(f"[run ] {cfg} ({MATRIX[cfg][1]}) overrides={over or '{}'}")
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "run.py", str(param), "--yes"],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            status = "ok" if proc.returncode == 0 else f"exit{proc.returncode}"
            if proc.returncode != 0:
                (run_dir / "_stderr.txt").write_text(proc.stderr[-20000:])
        except subprocess.TimeoutExpired:
            status = f"timeout>{args.timeout}s"
        wall = time.monotonic() - t0
        n_rows = 0
        d = run_dir / "dictionary.jsonl"
        if d.exists():
            n_rows = sum(1 for line in d.open() if line.strip())
        print(f"       -> {status} in {wall:.0f}s, {n_rows} snapshots")
        results.append((cfg, status, f"{wall:.1f}", str(n_rows)))

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"{args.arm}_walltimes.csv"
    with out.open("w") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write(f"# arm={args.arm} tier={args.tier} stop_t_override={args.stop_t or 'none'}\n")
        fh.write(f"# run root: {root.relative_to(REPO)}\n")
        fh.write("config,status,wall_s,n_snapshots\n")
        for row in results:
            fh.write(",".join(row) + "\n")
    print(f"\nwrote {out}")
    return 0 if all(r[1] in ("ok", "skipped") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
