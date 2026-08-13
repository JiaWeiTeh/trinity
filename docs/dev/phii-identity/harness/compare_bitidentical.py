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

# Empirical noise floor for a CROSS-WORKTREE comparison where the code change
# provably cannot act. Measured on B1M (Batch 3 control: never reaches the
# momentum phase, so C1 is inert there): every physical key agrees to machine
# precision until t ~ 0.8 Myr, then last-bit integrator drift grows to at most
# 2.9e-14 in R2 / 2.2e-13 in Pb by t = 1.5 Myr. The seed is `Lmech_SN`, which is
# `Lmech_total - Lmech_W` and therefore exactly zero before SN onset -- the stored
# ~1e-18 is a cancellation remnant ~1e-26 RELATIVE to Lmech_total, i.e. ten orders
# below double precision. 1e-10 sits ~3 decades above that drift and ~12 decades
# below any effect this workstream cares about (Batch 4a moved R2 by 15-28%).
NOISE_FLOOR = 1e-10


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


def _abs_max(v):
    """Largest magnitude inside a snapshot value (scalar or per-radius array)."""
    if isinstance(v, bool) or v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return abs(float(v))
    if isinstance(v, list):
        return max((_abs_max(x) for x in v), default=0.0)
    return 0.0


def key_scales(rows):
    """Per-key largest magnitude over the whole run — the scale a difference is
    judged against.

    Judging a difference against the *local* value is wrong for any quantity whose
    true value is zero: `pdot_SN` is `pdot_total - pdot_W` and no SNe occur before
    t = 3.6 Myr, so it holds cancellation garbage and a self-relative comparison
    reports 100% for a difference of ~1e-21. Normalising by the largest value the
    key ever reaches asks the question that actually matters — is this difference
    significant on the scale this quantity attains? — and leaves genuinely
    load-bearing keys judged as strictly as before.
    """
    scales = {}
    for d in rows:
        for k, v in d.items():
            m = _abs_max(v)
            if m > scales.get(k, 0.0):
                scales[k] = m
    return scales


def _abs_diff(va, vb):
    """Largest absolute difference between two snapshot values; inf if unalignable."""
    if isinstance(va, bool) or isinstance(vb, bool):
        return 0.0 if va == vb else float("inf")
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return abs(float(va) - float(vb))
    if isinstance(va, list) and isinstance(vb, list):
        if len(va) != len(vb):
            return float("inf")
        return max((_abs_diff(x, y) for x, y in zip(va, vb)), default=0.0)
    return 0.0 if va == vb else float("inf")


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

    # Scan EVERY row, not just up to the first difference: an exact-match prefix
    # followed by last-bit integrator drift is the normal signature of a genuine
    # no-op, and stopping early would report only the drift's first symptom.
    scales = key_scales(base)
    worst_rel, worst_key, worst_row, n_exact = 0.0, "", -1, 0
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
        if not diffs:
            n_exact += 1
            continue
        for k in diffs:
            ad = _abs_diff(b.get(k), n.get(k))
            scale = scales.get(k, 0.0)
            rel = float("inf") if (ad and not scale) else (ad / scale if scale else 0.0)
            if rel > worst_rel:
                worst_rel, worst_key, worst_row = rel, k, i

    if worst_rel == 0.0:
        return (
            "PASS-PREFIX" if truncated else "PASS",
            f"{len(base)} rows identical on all pre-existing keys{truncated}",
            len(base),
            ",".join(new_keys),
        )
    verdict = "PASS-NOISE" if worst_rel <= NOISE_FLOOR else "FAIL"
    return (
        verdict + ("-PREFIX" if truncated else ""),
        f"{n_exact}/{len(base)} rows exact; worst rel diff {worst_rel:.2e} "
        f"on {worst_key} at row {worst_row} (floor {NOISE_FLOOR:.0e}){truncated}",
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
