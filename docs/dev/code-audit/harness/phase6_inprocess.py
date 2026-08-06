"""In-process determinism: does a second run in the same interpreter match the first?

CLAUDE.md states trinity leaks module-level global state in-process, which is why
every baseline in this audit is launched as its own process. That is a documented
hazard nothing had measured. This probe measures it.

    python docs/dev/code-audit/harness/phase6_inprocess.py <param file>

Runs the same config twice inside one interpreter and diffs the two
dictionary.jsonl outputs. Identical => no leak on this path. Different => the leak
is real and this prints the first key that diverges.

Complements the separate-process probe (which passed, byte-identical): that one
tests reproducibility across processes, this one tests contamination within one.
"""

import json
import math
import pathlib
import shutil
import sys


def read_rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def first_divergence(a, b):
    """(snapshot index, key, value_a, value_b) of the first difference, or None."""
    for i, (ra, rb) in enumerate(zip(a, b)):
        for k in sorted(set(ra) | set(rb)):
            va, vb = ra.get(k), rb.get(k)
            if isinstance(va, float) and isinstance(vb, float):
                # NaN in both is agreement for our purposes, so isnan-guard it.
                if va != vb and not (math.isnan(va) and math.isnan(vb)):
                    return i, k, va, vb
            elif va != vb:
                return i, k, va, vb
    if len(a) != len(b):
        return min(len(a), len(b)), "<row count>", len(a), len(b)
    return None


def main(param_path):
    # Imported here so the import itself is inside the measured process.
    from trinity._input.read_param import read_param
    from trinity.main import start_expansion

    outputs = []
    for run in ("A", "B"):
        params = read_param(param_path)
        start_expansion(params)
        src = pathlib.Path(str(params["path2output"].value)) / "dictionary.jsonl"
        dst = src.with_name(f"dictionary_inproc_{run}.jsonl")
        shutil.copy(src, dst)
        outputs.append(dst)
        print(f"  run {run}: {len(read_rows(dst))} snapshots -> {dst}")

    a, b = (read_rows(p) for p in outputs)
    print(f"\nrun A: {len(a)} snapshots\nrun B: {len(b)} snapshots")
    div = first_divergence(a, b)
    if div is None:
        print("\nRESULT: identical — no in-process state leak on this path")
        return 0
    i, k, va, vb = div
    print(f"\nRESULT: DIVERGE at snapshot {i}, key '{k}'\n  A = {va!r}\n  B = {vb!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
