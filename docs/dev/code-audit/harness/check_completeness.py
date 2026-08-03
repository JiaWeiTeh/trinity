"""The audit's own bookkeeping: which phases are *actually* complete.

Written because Phase 3 was reported complete while 5 of its 9 sweeps existed —
a directory listing was eyeballed instead of counted. Every phase declares the
artifacts that constitute "done"; this script is the only thing entitled to say
a phase is complete.

    python docs/dev/code-audit/harness/check_completeness.py

Exit status is 0 only when every phase is complete. When a phase gains a
deliverable, add it to PHASES *first*, so the checklist can never lag the plan.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from slices import SLICES, check_partition

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _slice_reports():
    """Phase 2 expects A/B/C + reconciled per physics slice, A/B + reconciled per infra."""
    for sid, (_, tier, _files) in SLICES.items():
        lenses = ("lensA", "lensB", "lensC") if tier == "physics" else ("lensA", "lensB")
        for suffix in lenses + ("reconciled",):
            yield f"slices/{sid}_{suffix}.md"


# PLAN.md Phase 3 numbers these (1)-(9); keep both in sync.
SWEEPS = [
    ("(1) units & dimensions", "slices/sweep_units.md"),
    ("(2) signs, factors, exponents", "slices/sweep_signs.md"),
    ("(3) silent failure", "slices/sweep_silentfail.md"),
    ("(4) duplicate divergence", "slices/sweep_duplicates.md"),
    ("(5) dead code & unused contracts", "slices/sweep_deadcode.md"),
    ("(6) magic numbers & provenance", "slices/sweep_magic.md"),
    ("(7) table bounds", "slices/sweep_tablebounds.md"),
    ("(8) state mutation & aliasing", "slices/sweep_state.md"),
    ("(9) numerical hygiene", "slices/sweep_numerical.md"),
]

PHASES = {
    "0 ground truth": [
        "data/baseline.md",
        "data/calibration.md",
        "data/claims_prose.csv",
        "data/claims_literals.csv",
        "data/claims_guards.csv",
        "data/claims_params.csv",
        "reference/PHYSICS_SPEC.md",
        "reference/STRUCTURE_MAP.md",
    ],
    "1 slice partition": [],  # asserted by check_partition(), not by file existence
    "2 blind-lens triangulation": sorted(_slice_reports()),
    "3 cross-cutting sweeps": [path for _, path in SWEEPS],
    "4 test-suite audit": ["slices/test_suite_audit.md"],
    "5 verification gate": ["data/skeptics.md", "UNVERIFIED.md"],
    "6 dynamic verification": ["data/dynamic_verification.md"],
    "7 deliverables": ["FINDINGS.md"],
}


def main():
    incomplete = []
    print(f"audit root: {ROOT}\n")
    for phase, required in PHASES.items():
        missing = [p for p in required if not (ROOT / p).exists()]
        have = len(required) - len(missing)
        if phase.startswith("1 "):
            n = check_partition()  # raises if the partition has a gap or overlap
            print(f"  [OK]   {phase}: partition asserted, {n} files in {len(SLICES)} slices")
            continue
        mark = "OK" if not missing else "--"
        print(f"  [{mark}]   {phase}: {have}/{len(required)}")
        for p in missing:
            print(f"           missing: {p}")
        if missing:
            incomplete.append(phase)

    print()
    if incomplete:
        print(f"INCOMPLETE: {len(incomplete)} phase(s) -> {', '.join(incomplete)}")
        return 1
    print("ALL PHASES COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
