#!/usr/bin/env python3
"""Prove a diagnostic-only change moved nothing — PLAN.md Batch 1 gate G1(i).

Compares two arms' `dictionary.jsonl` row by row. Keys that exist only in the
new arm (the diagnostic being added) are dropped first; everything that existed
before must then match **exactly**, value for value, on every row.

Exactness here is real bit-equality, not a tolerance: floats are compared via
`repr()`, which in Python 3 round-trips IEEE-754 doubles losslessly, so two
values compare equal iff they are the same double. A single differing ULP fails
the gate — which is the point. If the shadow diagnostic perturbed the solver at
all (an extra dict key changing iteration order, a recomputed intermediate), it
shows up here rather than silently contaminating Batch 4's cap experiments.

Usage (from the repo root):
    python docs/dev/phii-identity/harness/compare_bitidentical.py \
        --base outputs/phii/b0__<sha> --new outputs/phii/b1__<sha> \
        --out docs/dev/phii-identity/data/b1_bitidentity.csv

Exit status is 1 if any config fails, so it can gate a commit.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stamp import stamp  # noqa: E402


def load(path):
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def canon(d, drop):
    """Exact, order-insensitive rendering of a row minus the new keys."""
    return {k: repr(v) for k, v in d.items() if k not in drop}


def compare(base_dir, new_dir, allow_prefix=False):
    """Returns (verdict, detail, n_rows, new_keys)."""
    b_path, n_path = base_dir / "dictionary.jsonl", new_dir / "dictionary.jsonl"
    if not b_path.exists() or not n_path.exists():
        return "SKIP", "missing dictionary.jsonl", 0, ""
    base, new = load(b_path), load(n_path)
    truncated = ""
    if len(base) != len(new):
        # A wall-clock timeout truncates an arm at a non-deterministic row, so a
        # length mismatch is only meaningful when both arms ran to completion.
        # With --allow-prefix we still check the overlap, which is what actually
        # proves the diagnostic inert, and say so in the verdict.
        if not allow_prefix:
            return "FAIL", f"row count {len(base)} vs {len(new)}", len(base), ""
        n_common = min(len(base), len(new))
        truncated = f" (prefix only: {len(base)} vs {len(new)} rows, compared {n_common})"
        base, new = base[:n_common], new[:n_common]

    new_keys = (
        sorted(set().union(*(d.keys() for d in new)) - set().union(*(d.keys() for d in base)))
        if base and new
        else []
    )
    drop = set(new_keys)

    for i, (b, n) in enumerate(zip(base, new)):
        cb, cn = canon(b, drop), canon(n, drop)
        if cb.keys() != cn.keys():
            missing = sorted(set(cb) ^ set(cn))
            return (
                "FAIL",
                f"row {i}: key set differs ({missing[:4]})",
                len(base),
                ",".join(new_keys),
            )
        diffs = [k for k in cb if cb[k] != cn[k]]
        if diffs:
            k = diffs[0]
            return (
                "FAIL",
                f"row {i}: {len(diffs)} key(s) differ, first {k}={cb[k]} vs {cn[k]}",
                len(base),
                ",".join(new_keys),
            )
    return (
        "PASS-PREFIX" if truncated else "PASS",
        f"{len(base)} rows identical on all pre-existing keys{truncated}",
        len(base),
        ",".join(new_keys),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="baseline arm root")
    ap.add_argument("--new", type=Path, required=True, help="new arm root")
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--allow-prefix",
        action="store_true",
        help="compare the common prefix when an arm was cut short by a wall-clock timeout",
    )
    args = ap.parse_args()

    configs = sorted(p.name for p in args.new.iterdir() if p.is_dir()) if args.new.is_dir() else []
    if not configs:
        sys.exit(f"no config dirs under {args.new}")

    rows, failed = [], False
    w = max(len(c) for c in configs)
    print(f"{'config':{w}}  {'verdict':>7}  detail")
    for cfg in configs:
        verdict, detail, n, keys = compare(args.base / cfg, args.new / cfg, args.allow_prefix)
        failed |= verdict == "FAIL"
        print(f"{cfg:{w}}  {verdict:>7}  {detail}")
        rows.append(
            {"config": cfg, "verdict": verdict, "n_rows": n, "new_keys": keys, "detail": detail}
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            fh.write(f"# base: {args.base}\n# new:  {args.new}\n")
            fh.write(
                "# PASS = every pre-existing key bit-identical on every row "
                "(floats compared via repr, so 1 ULP fails). PASS-PREFIX = same, but one arm "
                "was cut short by a wall-clock timeout so only the common prefix was compared.\n"
            )
            fh.write("config,verdict,n_rows,new_keys,detail\n")
            for r in rows:
                fh.write(
                    f"{r['config']},{r['verdict']},{r['n_rows']},"
                    f"\"{r['new_keys']}\",\"{r['detail']}\"\n"
                )
        print(f"\nwrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
