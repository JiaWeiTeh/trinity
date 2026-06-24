"""Reduce a Trinity parameter-sweep directory to one summary row per run.

The point: a full sweep is ~tens of GB of ``dictionary.jsonl`` trajectories that
must stay on the cluster. Aggregate "one-point-per-simulation" figures (escape
fraction, dominant feedback, allowed-GMC, ...) only need a handful of scalars per
run. This walks the sweep, extracts those scalars into a single ``summary.csv``
(a few MB for thousands of runs), and that tiny table is what you rsync to your
laptop to plot — the jsonl never crosses the wire.

Data sources per run (both cheap — we never parse the whole jsonl):
  - ``metadata.json``      : run constants (mCloud, sfe, nCore, rCloud, profile, ...)
  - last line of jsonl     : the final snapshot (R2, v2, forces, end reason, ...).
    Read by seeking to the end of the file, NOT by loading every timestep.

Usage:
    python tools/reduce_sweep.py outputs/my_sweep                 # -> outputs/my_sweep/summary.csv
    python tools/reduce_sweep.py outputs/my_sweep -o summary.csv  # explicit output path
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trinity._output.trinity_reader import find_all_simulations, parse_simulation_params

METADATA_KEYS = [
    "mCloud", "mCloud_input", "mCluster", "sfe", "ZCloud", "coverFraction",
    "dens_profile", "densPL_alpha", "nCore", "nISM", "nEdge", "rCore",
    "rCloud", "rCloud_max", "include_PHII", "betadelta_solver", "tSF", "PISM",
]

_SCALAR = (int, float, bool, str, type(None))


def _last_jsonl_record(path: Path) -> dict:
    """Return the final snapshot dict from a run's data file.

    For .jsonl (the modern format every sweep produces) this tails the last
    non-empty line without reading the rest of the file. For the legacy .json
    format it falls back to the full reader (rare; not worth optimising).
    """
    if path.suffix == ".jsonl":
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            pos = fh.tell()
            buf = b""
            while pos > 0:
                step = min(4096, pos)
                pos -= step
                fh.seek(pos)
                buf = fh.read(step) + buf
                if buf.count(b"\n") >= 2:
                    break
        lines = [ln for ln in buf.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("empty jsonl")
        return json.loads(lines[-1])

    from trinity._output.trinity_reader import load_output

    return dict(load_output(path)[-1].data)


def reduce_run(data_path: Path) -> dict:
    """Build one summary row for a single run directory."""
    run_dir = data_path.parent
    row = {"run_name": run_dir.name, "run_path": str(run_dir)}

    parsed = parse_simulation_params(run_dir.name)
    if parsed:
        row.update({f"dir_{k}": v for k, v in parsed.items()})

    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for k in METADATA_KEYS:
            if k in meta:
                row[k] = meta[k]

    final = _last_jsonl_record(data_path)
    for k, v in final.items():
        if isinstance(v, _SCALAR):
            row[k] = v

    if "shell_fAbsorbedIon" in final:
        row["escape_fraction"] = 1.0 - final["shell_fAbsorbedIon"]

    return row


def write_csv(rows: list[dict], out_path: Path) -> int:
    """Write rows to CSV, unioning keys so heterogeneous runs all fit. Returns column count."""
    preferred = [
        "run_name", "run_path", "dir_mCloud", "dir_sfe", "dir_ndens",
    ] + METADATA_KEYS + [
        "t_now", "R2", "v2", "rShell", "current_phase",
        "SimulationEndReason", "SimulationEndCode", "isCollapse",
        "escape_fraction",
    ]
    all_keys = set()
    for r in rows:
        all_keys.update(r)

    fieldnames = [k for k in preferred if k in all_keys]
    fieldnames += sorted(all_keys - set(fieldnames))

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return len(fieldnames)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("sweep_dir", help="directory containing run subdirectories")
    ap.add_argument(
        "-o", "--output", default=None,
        help="output CSV (default: <sweep_dir>/summary.csv)",
    )
    args = ap.parse_args(argv)

    sweep = Path(args.sweep_dir)
    out_path = Path(args.output) if args.output else sweep / "summary.csv"

    sims = find_all_simulations(sweep)
    if not sims:
        print(f"No simulations found under {sweep}", file=sys.stderr)
        return 1

    rows, failed = [], 0
    for i, data_path in enumerate(sims, 1):
        try:
            rows.append(reduce_run(data_path))
        except Exception as exc:
            failed += 1
            print(f"WARN: skipped {data_path.parent.name}: {exc}", file=sys.stderr)
        if i % 100 == 0 or i == len(sims):
            print(f"  ...{i}/{len(sims)} runs", file=sys.stderr)

    if not rows:
        print("No runs could be reduced.", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = write_csv(rows, out_path)
    size_kb = out_path.stat().st_size / 1024
    print(
        f"Wrote {len(rows)} runs x {ncols} cols -> {out_path} ({size_kb:.0f} KB)"
        + (f"  [{failed} skipped]" if failed else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
