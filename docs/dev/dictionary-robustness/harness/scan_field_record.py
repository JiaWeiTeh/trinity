#!/usr/bin/env python3
"""Scan a real run's on-disk record against the plan's robustness invariants.

Battery H of ``docs/dev/dictionary-robustness/PLAN.md``. Stdlib-only and not a
package — ``test/test_dictionary_stress_process.py`` imports it by path (the
convention used by ``test/test_rosette_cf_harness.py``), so the invariant logic
has exactly one home and is covered by the default test suite.

Usage — scan one or more completed run directories and append a CSV row each:

    python docs/dev/dictionary-robustness/harness/scan_field_record.py \
        --label smoke --commit 030b658 \
        --csv docs/dev/dictionary-robustness/data/field_scan.csv \
        <run_dir> [<run_dir> ...]

A run directory is one containing ``dictionary.jsonl`` (e.g.
``outputs/<model_name>/``). The CSV gets a provenance header on creation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# The writer's flush cadence (``DescribedDict.snapshot_interval``). Duplicated
# here on purpose: the harness is stdlib-only and must not import trinity, and
# finding F16 is precisely that this number is a hardcoded constant with no
# .param key. Re-check it if that ever changes.
SNAPSHOT_INTERVAL = 10

FIELDS = [
    "label", "commit", "run_dir", "snapshots", "unparsable_lines",
    "distinct_keysets", "t_non_decreasing", "adjacent_dup_indices",
    "adjacent_dups_off_boundary", "near_dup_indices",
    "phase_boundary_indices", "phases", "nan_bearing_lines",
    "inf_bearing_lines",
]


def scan_run_record(run_dir: Path) -> dict:
    """Return one row of invariant findings for the run at ``run_dir``.

    Invariant IDs refer to ``PLAN.md`` §2: I1 (every line parses), I3
    (t non-decreasing; duplicates only at flush boundaries), I4 (stable
    per-line key-set). F11 is the NaN/Infinity literal count.
    """
    raw = (Path(run_dir) / "dictionary.jsonl").read_text().splitlines()
    lines, unparsable = [], 0
    for ln in raw:
        if not ln.strip():
            continue
        try:
            lines.append(json.loads(ln))
        except json.JSONDecodeError:
            unparsable += 1

    ts = [ln.get("t_now") for ln in lines]
    phases = [ln.get("current_phase") for ln in lines]

    # I3: adjacent duplicate guard-keys. F1 predicts these only where the
    # buffer was just cleared, i.e. index % SNAPSHOT_INTERVAL == 0; anything
    # elsewhere means the in-window guard missed a duplicate (a new finding).
    dups = [i for i in range(1, len(lines))
            if (lines[i].get("t_now"), lines[i].get("R2"))
            == (lines[i - 1].get("t_now"), lines[i - 1].get("R2"))]
    # Near-duplicates the exact-equality guard cannot see.
    near_dups = [i for i in range(1, len(lines))
                 if ts[i] is not None and ts[i - 1] is not None
                 and i not in dups and abs(ts[i] - ts[i - 1]) < 1e-12]

    return {
        "snapshots": len(lines),
        "unparsable_lines": unparsable,
        "distinct_keysets": len({frozenset(ln) for ln in lines}),
        "t_non_decreasing": all(
            a <= b for a, b in zip(ts, ts[1:]) if a is not None and b is not None
        ),
        "adjacent_dup_indices": dups,
        "adjacent_dups_off_boundary": [i for i in dups if i % SNAPSHOT_INTERVAL != 0],
        "near_dup_indices": near_dups,
        "phase_boundary_indices": [i for i in range(1, len(lines))
                                   if phases[i] != phases[i - 1]],
        "phases": sorted({p for p in phases if p is not None}),
        "nan_bearing_lines": sum(1 for ln in raw if "NaN" in ln),
        "inf_bearing_lines": sum(1 for ln in raw if "Infinity" in ln),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--label", required=True, help="config name, e.g. smoke")
    ap.add_argument("--commit", required=True, help="commit the run was made at")
    args = ap.parse_args()

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fresh = not args.csv.exists()
    with args.csv.open("a", newline="", encoding="utf-8") as fh:
        if fresh:
            fh.write(
                "# Battery-H field scan of dictionary.jsonl invariants.\n"
                "# Produced by docs/dev/dictionary-robustness/harness/scan_field_record.py\n"
                "# One row per (config, commit). See PLAN.md §2 for invariant IDs.\n"
            )
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        for run_dir in args.run_dirs:
            row = scan_run_record(run_dir)
            row.update(label=args.label, commit=args.commit, run_dir=str(run_dir))
            writer.writerow(row)
            print(f"{args.label}: {row['snapshots']} snapshots, "
                  f"dups={row['adjacent_dup_indices']}, "
                  f"off-boundary={row['adjacent_dups_off_boundary']}, "
                  f"phases={row['phases']}")


if __name__ == "__main__":
    main()
