#!/usr/bin/env python3
"""Read a bundled trajectory CSV back into the per-arm shape the analysis scripts expect.

``harvest_bench5.py --traj-bundle`` writes every arm's θ(t) trajectory into ONE long CSV with a
leading ``run_name`` column, so a 500+-arm campaign (bench8/f_area) comes down from the cluster as
a couple of files instead of one per arm. This splits it back apart in memory:

    from read_bundle import load
    for run_name, rows in load(RDATA / "bench8_traj.csv").items():
        theta_cum_prefire(rows)          # rows are the same dicts _read_csv yields per-arm

The rows are byte-for-byte the values the per-arm files would have carried (same writer, same
float repr), so every existing consumer — theta_cum_prefire, decompose, the track plots — works on
them unchanged. Comment ('#') lines are skipped, matching make_bench5_analysis._read_csv.
"""
import csv
from pathlib import Path


def load(path):
    """-> {run_name: [row dicts]}, each row keyed by the bundle's columns minus ``run_name``.

    Insertion order is the bundle's own arm order (harvest writes arms in argv order); rows keep
    their in-file order, which harvest already sorted by t_now.
    """
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(x for x in fh if not x.lstrip().startswith("#")):
            out.setdefault(row.pop("run_name"), []).append(row)
    return out


if __name__ == "__main__":                      # tiny CLI: how many arms, how many rows
    import sys
    for p in sys.argv[1:]:
        d = load(Path(p))
        print(f"{p}: {len(d)} arms, {sum(len(v) for v in d.values())} rows")
