#!/usr/bin/env python3
"""Cf-scan harvest + checkpoint — summary CSV + a compact, restart-durable trajectory per arm.

Both outputs are committed so matching runs OFFLINE from git, never re-running the sims (the
theta5s lesson: raw arms lost to a /tmp wipe). Doubles as the restart checkpoint: it MERGES the
current container's arms into the committed summary (union by run_name, prefer exit_code==0 rows)
and refreshes each arm's trajectory CSV — call it repeatedly (the autocommit heartbeat does).

1. --csv <summary>: one row per arm — axes (parsed from the run-folder name, which encodes all six;
   run.py leaves no .param in the output dir), exit code, duration, t_final, phase_final, final
   radii, quotable flag (exit_code==0; 📏 never quote a 124 arm).
2. --traj-dir <dir>: per arm, <arm>.csv with every snapshot's (t_now, R2, v2, rShell,
   current_phase) — a lightweight index for quick offline matching. Capped at 4000 rows by stride
   downsample keeping endpoints.
3. --dicts-dir <dir>: per FINISHED arm, <arm>.jsonl.gz — the gzipped RAW dictionary.jsonl. This
   is the actual Rosette deliverable: the maintainer reduces these later with their own tools. Raw
   dicts are large (~10 MB/arm, ~26 KB/snapshot because each snapshot carries the full shell-
   density arrays), so gzip (~2.7x here) is what keeps 72 of them committable. gunzip before
   reducing. Idempotent (skips arms whose .gz is already current).

    python docs/dev/rosette-cf/harness/harvest_cf_scan.py "$WS"/outputs/rosette_cf_PISM1e5/* \
        --csv docs/dev/rosette-cf/data/cf_scan_PISM1e5_summary.csv \
        --traj-dir docs/dev/rosette-cf/data/cf_scan_PISM1e5_traj \
        --dicts-dir docs/dev/rosette-cf/data/cf_scan_PISM1e5_dicts
"""

import argparse
import csv
import gzip
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _stamp import stamp  # noqa: E402

# The six swept axes, all encoded in the run-folder name by the sweep_parser naming convention
# (PISM=1e5 and stop_t=3 are constant across the scan — stated in HEADER, not per-row columns).
AXES = [
    "mCloud",
    "sfe",
    "nCore",
    "coverFraction",
    "cooling_boost_fmix",
    "include_PHII",
]
COLUMNS = (
    ["run_name"]
    + AXES
    + [
        "exit_code",
        "duration_s",
        "n_snap",
        "t_final",
        "phase_final",
        "R2_final",
        "rShell_final",
        "quotable",
    ]
)
TRAJ_COLS = ["t_now", "R2", "v2", "rShell", "current_phase"]
TRAJ_CAP = 4000
HEADER = (
    "# Rosette Cf scan (PISM=1e5, stop_t=3 Myr) cumulative summary, merged across container\n"
    "# restarts by harvest_cf_scan.py. quotable = exit_code==0 (a 124 arm was wall-killed: its\n"
    "# t_final is not a physics end — 📏 never quote its Cf). Per-arm trajectory CSVs in the\n"
    "# sibling traj dir carry (t_now, R2, v2, rShell, phase) for offline matching.\n"
    "# ⚠️ PROVISIONAL / IN-CONTAINER — NOT HPC (HPC down; same caveat as the theta5s/bench5\n"
    "# campaigns). Re-confirm on HPC before any paper number.\n"
)


def _finite(v):
    return (
        v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
    )


def parse_run_name(name):
    """Recover the six swept axes from a run-folder name (or <name>.jsonl.gz / .csv).

    The sweep_parser naming convention encodes every axis, e.g.
    ``1e5_sfe001_n5e2_PL0_yesPHII_coolingBoostFmix1p0_coverFraction0p7`` — so this is the reliable
    source (run.py does NOT leave a .param in the output dir, and after a container reclaim the raw
    output dirs are gone but the committed .jsonl.gz keep the name). sfe is the *100 zero-padded
    integer (``sfe001`` -> 0.01); nCore/mCloud are compact scientific (``n5e2`` -> 500, ``n50`` ->
    50); ``p`` is the decimal point in the generic suffixes.
    """
    name = re.sub(r"\.(jsonl\.gz|csv|param)$", "", name)
    return {
        "mCloud": float(name.split("_", 1)[0]),
        "sfe": int(re.search(r"_sfe(\d+)", name).group(1)) / 100,
        "nCore": float(re.search(r"_n([0-9]+(?:e[0-9]+)?)_", name).group(1)),
        "coverFraction": float(
            re.search(r"coverFraction([0-9p]+)", name).group(1).replace("p", ".")
        ),
        "cooling_boost_fmix": float(
            re.search(r"coolingBoostFmix([0-9p]+)", name).group(1).replace("p", ".")
        ),
        "include_PHII": bool(re.search(r"_yesPHII", name)),
    }


def snapshots(run_dir):
    rows = []
    path = run_dir / "dictionary.jsonl"
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = _finite(d.get("t_now"))
            if t is None:
                continue
            rows.append(
                [
                    t,
                    _finite(d.get("R2")),
                    _finite(d.get("v2")),
                    _finite(d.get("rShell")),
                    d.get("current_phase"),
                ]
            )
    rows.sort(key=lambda r: r[0])
    return rows


def harvest(run_dir):
    run_dir = Path(run_dir)
    rows = snapshots(run_dir)
    last = rows[-1] if rows else [None] * 5

    def _sentinel(name):
        f = run_dir / name
        return f.read_text().strip() if f.exists() else ""

    exit_code = _sentinel(".exit_code")
    rec = {"run_name": run_dir.name, **parse_run_name(run_dir.name)}
    rec.update(
        exit_code=exit_code,
        duration_s=_sentinel(".duration"),
        n_snap=len(rows),
        t_final=last[0],
        phase_final=last[4],
        R2_final=last[1],
        rShell_final=last[3],
        quotable=(exit_code == "0"),
    )
    return rec


def write_traj(run_dir, traj_dir):
    rows = snapshots(Path(run_dir))
    if not rows:
        return 0
    if len(rows) > TRAJ_CAP:  # stride downsample, keep endpoints
        step = len(rows) / (TRAJ_CAP - 1)
        idx = sorted(
            {min(int(i * step), len(rows) - 1) for i in range(TRAJ_CAP - 1)} | {len(rows) - 1}
        )
        rows = [rows[i] for i in idx]
    traj_dir.mkdir(parents=True, exist_ok=True)
    with (traj_dir / f"{Path(run_dir).name}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRAJ_COLS)
        w.writerows(rows)
    return len(rows)


def write_dict_gz(run_dir, dicts_dir):
    """Gzip a FINISHED arm's raw dictionary.jsonl into <dicts_dir>/<arm>.jsonl.gz.

    The raw dict is the actual Rosette deliverable (maintainer reduces it later); saving it
    committed + gzipped is what survives the ephemeral container. Only arms with a .exit_code are
    written (a dict still being appended to would be truncated). Idempotent: skips if the .gz is
    already newer than the source, so the ~2-min heartbeat never re-gzips unchanged arms.
    Returns bytes written (0 if skipped/absent).
    """
    run_dir = Path(run_dir)
    src = run_dir / "dictionary.jsonl"
    if not src.exists() or not (run_dir / ".exit_code").exists():
        return 0
    dicts_dir.mkdir(parents=True, exist_ok=True)
    dst = dicts_dir / f"{run_dir.name}.jsonl.gz"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return 0
    data = src.read_bytes()
    # mtime=0 -> deterministic gzip header (no embedded timestamp), so an unchanged dict
    # produces a byte-identical .gz and does not create spurious git diffs.
    with gzip.GzipFile(dst, "wb", mtime=0) as fh:
        fh.write(data)
    return dst.stat().st_size


def _read_summary(path):
    if not path.exists():
        return {}
    with path.open() as fh:
        return {
            r["run_name"]: r
            for r in csv.DictReader(x for x in fh if not x.lstrip().startswith("#"))
        }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--csv", required=True, help="committed summary CSV (merged, not overwritten)")
    ap.add_argument("--traj-dir", help="committed per-arm trajectory dir")
    ap.add_argument("--dicts-dir", help="committed dir for gzipped raw dictionary.jsonl per arm")
    args = ap.parse_args(argv)

    csv_out = Path(args.csv)
    merged = _read_summary(csv_out)
    n_new = n_dicts = 0
    for a in args.run_dirs:
        run_dir = Path(a)
        if not (run_dir / "dictionary.jsonl").exists():
            continue
        rec = {k: ("" if v is None else v) for k, v in harvest(run_dir).items()}
        name = rec["run_name"]
        if name not in merged or (rec["exit_code"] == "0" and merged[name].get("exit_code") != "0"):
            merged[name] = rec
            n_new += 1
        if args.traj_dir:
            write_traj(run_dir, Path(args.traj_dir))
        if args.dicts_dir and write_dict_gz(run_dir, Path(args.dicts_dir)):
            n_dicts += 1

    # Backfill axes from the run name for every row — authoritative and always available, so rows
    # harvested by an older build (or whose raw output dir was wiped by a reclaim) get correct axes.
    for name, row in merged.items():
        row.update(parse_run_name(name))
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    rows = [merged[k] for k in sorted(merged)]
    with csv_out.open("w", newline="") as fh:
        fh.write(stamp(str(HERE / "harvest_cf_scan.py")) + "\n")
        fh.write(HEADER)
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_quot = sum(1 for r in rows if str(r.get("quotable")) == "True")
    print(
        f"{len(rows)} arms in summary ({n_quot} quotable, {n_new} added/upgraded, "
        f"{n_dicts} dict.gz written this pass)"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
