"""Scan a TRINITY run's dictionary.jsonl for things physics forbids.

Static review cannot tell you the trajectory is right; this can. Reports rather
than asserts, because several of these are judgement calls — the audit decides
which reported item is a finding.

    python docs/dev/code-audit/harness/check_invariants.py outputs/<run>/dictionary.jsonl
"""

import json
import math
import sys
from collections import defaultdict

# Quantities that are positive-definite in every regime this code integrates.
POSITIVE = [
    "R2", "R1", "Eb", "Pb", "T0", "shell_mass", "bubble_mass", "shell_thickness",
    "rShell", "c_sound", "bubble_Tavg", "shell_n0", "shell_nMax",
]
NONNEGATIVE = ["t_now", "n_IF", "shell_fAbsorbedIon", "shell_fAbsorbedNeu"]


def scalars(row):
    return {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def report(title, items):
    print(f"\n## {title}")
    print("  (none)" if not items else "\n".join(f"  {i}" for i in items))


def main(path):
    rows = [json.loads(line) for line in open(path)]
    print(f"# invariant scan — {path}\n\n{len(rows)} snapshots, "
          f"t = {rows[0].get('t_now'):.6g} .. {rows[-1].get('t_now'):.6g} Myr")

    # --- non-finite values, first occurrence only ---
    bad = {}
    for i, row in enumerate(rows):
        for k, v in scalars(row).items():
            if not math.isfinite(v) and k not in bad:
                bad[k] = (i, row.get("t_now"), v)
    report("non-finite values (first occurrence)",
           [f"{k}: snapshot {i} (t={t:.6g}) = {v}" for k, (i, t, v) in sorted(bad.items())])

    # --- sign violations ---
    viol = []
    for name, keys, ok in (("must be > 0", POSITIVE, lambda v: v > 0),
                           ("must be >= 0", NONNEGATIVE, lambda v: v >= 0)):
        for k in keys:
            hits = [(i, r[k]) for i, r in enumerate(rows)
                    if isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) and not ok(r[k])]
            if hits:
                viol.append(f"{k} ({name}): {len(hits)} snapshots, first at {hits[0][0]} = {hits[0][1]:.6g}")
    report("sign violations", viol)

    # --- monotonicity ---
    mono = []
    for k, want in (("t_now", "increasing"), ("R2", "increasing")):
        vals = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
        drops = [i for i in range(1, len(vals)) if vals[i] < vals[i - 1]]
        if drops:
            mono.append(f"{k} ({want}): {len(drops)} decreases, first at snapshot {drops[0]} "
                        f"({vals[drops[0] - 1]:.6g} -> {vals[drops[0]]:.6g})")
    report("monotonicity", mono)

    # --- distinct keys carrying bit-identical values in every snapshot ---
    # Either a deliberate alias or two names for one quantity that have silently
    # collapsed onto each other; both are worth a verdict.
    series = defaultdict(list)
    common = set.intersection(*(set(scalars(r)) for r in rows))
    for k in sorted(common):
        series[tuple(rows[i][k] for i in range(len(rows)))].append(k)
    report("distinct keys with bit-identical series",
           [" == ".join(ks) for vals, ks in series.items()
            if len(ks) > 1 and any(v != 0 for v in vals)])

    # --- constant-forever keys (written once, never updated) ---
    report("scalar keys constant across the whole run",
           [f"{ks[0]} = {vals[0]:.6g}" for vals, ks in series.items()
            if len(set(vals)) == 1 and len(ks) == 1 and vals[0] not in (0, 0.0)])

    # --- self-similar expansion exponent, per phase ---
    # Energy-driven (Weaver) in a uniform medium: R ~ t^(3/5); momentum: R ~ t^(1/2).
    print("\n## expansion exponent  d(log R2)/d(log t), fitted per phase")
    by_phase = defaultdict(list)
    for r in rows:
        if isinstance(r.get("R2"), (int, float)) and isinstance(r.get("t_now"), (int, float)):
            by_phase[r.get("current_phase")].append((r["t_now"], r["R2"]))
    for phase, pts in by_phase.items():
        pts = [(t, R) for t, R in pts if t > 0 and R > 0]
        if len(pts) < 5:
            print(f"  {phase}: {len(pts)} usable points — too few to fit")
            continue
        n = len(pts)
        xs = [math.log(t) for t, _ in pts]
        ys = [math.log(R) for _, R in pts]
        mx, my = sum(xs) / n, sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else float("nan")
        print(f"  {phase:12s} n={n:4d}  t=[{pts[0][0]:.4g}, {pts[-1][0]:.4g}]  "
              f"alpha={slope:.4f}")
    print("\n  reference: energy-driven 0.6, momentum-driven 0.5 "
          "(uniform density, constant mechanical luminosity)")


if __name__ == "__main__":
    main(sys.argv[1])
