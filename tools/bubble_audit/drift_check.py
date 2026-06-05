"""Silent-drift guard for the bubble-luminosity solver.

For EVERY change to production code, prove the SUCCESS PATH is unchanged: run
``get_bubbleproperties_pure`` on fixed dumped states (see TRINITY_BUBBLE_STATE_DUMP
and audit.py) and compare every BubbleProperties field against a baseline
computed from the *original* code. A clean change reports worst rel diff 0
(byte-identical success path); any non-zero diff is silent drift to explain.

Workflow -- baseline against a git ref (e.g. the pre-change commit):

    # 1. current code
    python tools/bubble_audit/drift_check.py compute <states_dir> /tmp/cur
    # 2. baseline: swap production files to the ref, recompute, restore
    git show <ref>:trinity/_functions/operations.py \
        > trinity/_functions/operations.py
    git show <ref>:trinity/bubble_structure/bubble_luminosity.py \
        > trinity/bubble_structure/bubble_luminosity.py
    python tools/bubble_audit/drift_check.py compute <states_dir> /tmp/base
    git checkout HEAD -- trinity/_functions/operations.py \
        trinity/bubble_structure/bubble_luminosity.py
    # 3. compare
    python tools/bubble_audit/drift_check.py compare /tmp/cur /tmp/base

Validated 2026-06 on the guard-A + Phase-0 changes vs ef35ab2: 0.00e+00.
"""
from __future__ import annotations

import os
import sys
import glob
import dataclasses

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from audit import load_state  # noqa: E402


def compute(states_dir: str, out_dir: str, base_param: str | None = None) -> None:
    """Run get_bubbleproperties_pure on each state; save all fields to npz."""
    import trinity.bubble_structure.bubble_luminosity as bl  # late: may be swapped
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(states_dir, "*.pkl")))
    for f in files:
        kwargs = {} if base_param is None else {"base_param": base_param}
        params, inputs, ref, meta = load_state(f, **kwargs)
        props = bl.get_bubbleproperties_pure(params)
        out = {fld.name: np.asarray(getattr(props, fld.name), dtype=float)
               for fld in dataclasses.fields(props)}
        name = os.path.splitext(os.path.basename(f))[0]
        np.savez(os.path.join(out_dir, name + ".npz"), **out)
        print(f"  computed {name}")


def compare(dir_a: str, dir_b: str) -> float:
    """Field-by-field rel diff of two compute() output dirs. Returns worst rel."""
    worst = 0.0
    files = sorted(glob.glob(os.path.join(dir_a, "*.npz")))
    for fa in files:
        name = os.path.basename(fa)
        fb = os.path.join(dir_b, name)
        if not os.path.exists(fb):
            print(f"{name}: MISSING in baseline")
            worst = float("inf")
            continue
        a_npz, b_npz = np.load(fa), np.load(fb)
        print(f"\n{name}:")
        for k in sorted(a_npz.files):
            a, b = a_npz[k], b_npz[k]
            if a.shape != b.shape:
                print(f"  {k:22} SHAPE DIFF {a.shape} vs {b.shape}")
                worst = float("inf")
                continue
            rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))) if a.size else 0.0
            worst = max(worst, rel)
            tag = "exact" if np.array_equal(a, b) else f"max_rel={rel:.2e}"
            flag = "" if (np.array_equal(a, b) or rel < 1e-13) else "  <-- DRIFT"
            print(f"  {k:22} {tag}{flag}")
    print(f"\n=== worst rel diff = {worst:.2e} "
          f"({'NO DRIFT' if worst < 1e-13 else 'DRIFT DETECTED'}) ===")
    return worst


def main(argv):
    if len(argv) >= 3 and argv[0] == "compute":
        compute(argv[1], argv[2], argv[3] if len(argv) > 3 else None)
        return 0
    if len(argv) >= 3 and argv[0] == "compare":
        return 0 if compare(argv[1], argv[2]) < 1e-13 else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
