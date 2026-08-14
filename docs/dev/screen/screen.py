#!/usr/bin/env python3
"""Multi-config scheme screen: two git refs, N configs, one pass/fail ledger.

Answers the question every scheme change has to answer and currently hand-rolls
from scratch: *does this change behave on configs other than the one config the
whole test suite uses?*  Every end-to-end test in `test/` runs `mCloud=1e5,
sfe=0.3`; the multi-config coverage exists only as `.param` files driven by HPC
campaigns.  See `docs/dev/phase1a-init/PLAN.md` §9 for how that gap was felt.

Both arms run in **separate processes** (trinity leaks module-level global
state) and are compared at **matched simulation time** by interpolating both
onto a common grid -- never nearest-snapshot, which is the error this harness
exists partly to stop people repeating (CLAUDE.md rule 5).

    python docs/dev/screen/screen.py --before HEAD~1 --after WORKTREE \\
        --configs simple_cluster,f1edge_hidens --stop-t 0.02 --bar 5

`WORKTREE` as a ref means "the current working tree", so uncommitted work can be
screened without committing it.  Any other value is resolved with
`git worktree add`, left in place afterwards, and the cleanup command printed --
this harness does not delete things on your behalf.

Output: per-run CSVs plus a ledger CSV in the `data/gate_results.csv` schema
(gate,config,quantity,reference,reference_source,measured,rel_diff,verdict), and
a pass/fail table on stdout.  Exit code is 1 if any config fails its bar, so it
can gate a merge.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
MYR2YR = 1e6

# The screen set: spans the axes that have actually broken things -- core
# density over four decades, and both feedback extremes -- plus the sub-GMC
# scale no config in `test/` covers.
CONFIGS = {
    "simple_cluster": "param/simple_cluster.param",
    "f1edge_lowdens": "docs/dev/performance/f1edge_lowdens_himass_hisfe.param",
    "f1edge_hidens": "docs/dev/performance/f1edge_hidens_himass_losfe.param",
    "m43_probe": "docs/dev/phase1a-init/harness/params/probe.param",
    "gmc_control": "docs/dev/phase1a-init/harness/params/gmc_control.param",
}

# Times (yr) the arms are compared at, when both arms reach them. The last time
# both arms share is always added -- that is the "or the end of the run if it
# terminates earlier" clause any sane bar needs, since configs stop at wildly
# different t (a collapse at 0.04 Myr vs a 2 Myr stopping time).
GRID = (3e3, 1e4, 3e4, 1e5, 3e5, 1e6)


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def interp(xs, ys, x):
    """Linear interpolation; None outside the sampled range (never extrapolate)."""
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i] >= x:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return None


def resolve_ref(ref, workdir):
    """A run.py path for `ref`. WORKTREE means the live tree; else a git worktree."""
    if ref == "WORKTREE":
        return os.path.join(REPO, "run.py"), None
    wt = os.path.join(workdir, f"tree-{ref.replace('/', '_').replace('~', '-')}")
    if not os.path.isdir(wt):
        sh(["git", "worktree", "add", "--detach", wt, ref], cwd=REPO)
    return os.path.join(wt, "run.py"), wt


def write_param(src, dst, stop_t, name):
    """Copy a .param, overriding only what the screen controls."""
    drop = {"stop_t", "path2output", "model_name", "log_console"}
    keep = []
    with open(os.path.join(REPO, src)) as fh:
        for line in fh:
            key = line.strip().split()[0] if line.strip() and not line.startswith("#") else None
            if key not in drop:
                keep.append(line.rstrip("\n"))
    with open(dst, "w") as fh:
        fh.write("\n".join(keep) + "\n")
        fh.write(f"stop_t {stop_t}\nmodel_name {name}\nlog_console False\n")


def run_arm(run_py, config, arm, stop_t, workdir):
    """One full run.py in its own cwd and process. Returns the snapshot rows."""
    cwd = os.path.join(workdir, f"{config}-{arm}")
    os.makedirs(cwd, exist_ok=True)
    param = os.path.join(cwd, "p.param")
    write_param(CONFIGS[config], param, stop_t, "screen")
    with open(os.path.join(cwd, "run.log"), "w") as log:
        proc = subprocess.run([sys.executable, run_py, param], cwd=cwd,
                              stdout=log, stderr=subprocess.STDOUT, text=True)
    jsonl = os.path.join(cwd, "outputs", "screen", "dictionary.jsonl")
    if proc.returncode != 0 or not os.path.exists(jsonl):
        return None, cwd
    with open(jsonl) as fh:
        return [json.loads(line) for line in fh if line.strip()], cwd


def fate(rows, run_dir=None):
    """The run's stopping fate, as 'code outcome'.

    The end record lands in ``metadata.json[termination]`` (exit_code,
    outcome), NOT in the snapshot rows: a run that stops on ``stop_t`` flushes
    its last snapshot before ``main.py`` stamps the code, so the jsonl tail
    carries ``SimulationEndCode: None`` even for a clean STOPPING_TIME end
    (verified 2026-08-06 on f1edge_hidens -- the first screen run in anger).
    Fall back to the row fields for outputs that do carry them; only a run
    with neither reports ``(no stop condition reached)``. Two arms that both
    stop short still compare equal -- but the fate check is then vacuous, so
    make that visible instead of dressing it up as a pass.
    """
    if run_dir:
        meta = os.path.join(run_dir, "metadata.json")
        if os.path.exists(meta):
            try:
                with open(meta) as fh:
                    term = json.load(fh).get("termination") or {}
            except ValueError:
                term = {}
            if term.get("exit_code") is not None or term.get("outcome"):
                return f"{term.get('exit_code')} {term.get('outcome', '')}".strip()
    last = rows[-1]
    code, reason = last.get("SimulationEndCode"), last.get("SimulationEndReason")
    if code is None and not reason:
        return "(no stop condition reached)"
    return f"{code} {reason}".strip()


def compare(config, before, after, bar, before_dir=None, after_dir=None):
    """Ledger rows + a verdict for one config, at matched t."""
    tb = [r["t_now"] * MYR2YR for r in before]
    ta = [r["t_now"] * MYR2YR for r in after]
    rb = [r["R2"] for r in before]
    ra = [r["R2"] for r in after]
    last = min(tb[-1], ta[-1])
    times = sorted({t for t in GRID if max(tb[0], ta[0]) <= t <= last} | {last})

    out, worst = [], 0.0
    for t in times:
        b, a = interp(tb, rb, t), interp(ta, ra, t)
        if b is None or a is None or b == 0:
            continue
        pct = 100 * (a - b) / b
        worst = max(worst, abs(pct))
        out.append(("screen", config, f"dR2_at_{t:.4g}yr_pct", f"<{bar}", "before arm",
                    f"{pct:+.3f}", f"{abs(pct) / 100:.2e}",
                    "PASS" if abs(pct) < bar else "FAIL"))

    fb, fa = fate(before, before_dir), fate(after, after_dir)
    same_fate = fb == fa
    out.append(("screen", config, "stopping_fate", fb, "before arm", fa,
                "0.00e+00" if same_fate else "NA", "PASS" if same_fate else "FAIL"))
    return out, (worst < bar and same_fate), worst, last, fb, fa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True, help="git ref for the baseline arm, or WORKTREE")
    p.add_argument("--after", default="WORKTREE", help="git ref for the candidate arm")
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--stop-t", type=float, default=0.02, help="Myr; overrides each config")
    p.add_argument("--bar", type=float, default=5.0, help="max |dR2| %% at any compared time")
    p.add_argument("--workdir", default=os.path.join(REPO, "outputs", "screen"))
    p.add_argument("--out", default=os.path.join(HERE, "data", "screen_results.csv"))
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown config(s): {', '.join(unknown)}. known: {', '.join(CONFIGS)}")

    os.makedirs(a.workdir, exist_ok=True)
    before_py, before_wt = resolve_ref(a.before, a.workdir)
    after_py, after_wt = resolve_ref(a.after, a.workdir)

    jobs = [(cfg, arm) for cfg in configs for arm in ("before", "after")]
    runner = {"before": before_py, "after": after_py}
    print(f"screening {len(configs)} config(s) x 2 arms, stop_t={a.stop_t} Myr, "
          f"bar |dR2| < {a.bar}%  ({a.workers} at a time)\n")

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(
            lambda j: (j, run_arm(runner[j[1]], j[0], j[1], a.stop_t, a.workdir)), jobs))
    runs = {job: res for job, res in results}

    ledger, failures = [], []
    print(f"{'config':<18}{'worst |dR2|':>13}{'at t (yr)':>13}  fate                verdict")
    print("-" * 78)
    for cfg in configs:
        (rows_b, cwd_b), (rows_a, cwd_a) = runs[(cfg, "before")], runs[(cfg, "after")]
        if not rows_b or not rows_a:
            dead = cwd_b if not rows_b else cwd_a
            ledger.append(("screen", cfg, "run_completes", "yes", "before arm",
                           "NO - see run.log", "NA", "FAIL"))
            failures.append(cfg)
            print(f"{cfg:<18}{'—':>13}{'—':>13}  run failed          FAIL  ({dead})")
            continue
        rows, ok, worst, last, fb, fa = compare(
            cfg, rows_b, rows_a, a.bar,
            os.path.join(cwd_b, "outputs", "screen"),
            os.path.join(cwd_a, "outputs", "screen"))
        ledger += rows
        if not ok:
            failures.append(cfg)
        note = fb if fb == fa else f"{fb} -> {fa}"
        print(f"{cfg:<18}{worst:>12.3f}%{last:>13.4g}  {note:<20}{'PASS' if ok else 'FAIL'}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        fh.write(f"# scheme screen: before={a.before} after={a.after} stop_t={a.stop_t} "
                 f"bar={a.bar}%; separate processes, matched t by interpolation\n")
        w = csv.writer(fh)
        w.writerow(["gate", "config", "quantity", "reference", "reference_source",
                    "measured", "rel_diff", "verdict"])
        w.writerows(ledger)
    print(f"\nledger: {os.path.normpath(a.out)}")
    for wt in (before_wt, after_wt):
        if wt:
            print(f"worktree left in place: git worktree remove {wt}")
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nall configs pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
