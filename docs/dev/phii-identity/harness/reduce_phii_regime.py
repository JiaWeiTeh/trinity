#!/usr/bin/env python3
"""Reduce a stock/C3c run pair to one small trajectory CSV for the regime figures.

House pattern (paper/CLAUDE.md): reduce once with the stdlib only, plot from the
table. This walks each arm's ``dictionary.jsonl`` and emits the raw per-row fields
the confinement figures need. It does NOT convert units and does NOT compute
P_C3a -- both belong in the plotting layer, which uses trinity's own constants
(``plot_phii_regime.py``).

Emitted per row: the confinement ingredients (R2, Qi, shell_fAbsorbedIon), the
confining pressure (Pb, P_ram), the scheme's answer (P_HII, P_drive) and the
force budget the feedback figure decomposes (F_grav/F_rad/F_HII/F_ram_*).

``n_IF_Str`` / ``n_IF_Str_raw`` / ``shell_n0`` come along so the stock identity
(capped Stromgren density <=> pressure equilibrium with Pb) can be re-checked
straight from the table rather than taken on trust.

Usage:
    python docs/dev/phii-identity/harness/reduce_phii_regime.py \
        --arm stock=<run_dir> --arm c3c=<run_dir> --out <path.csv>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stamp import stamp  # noqa: E402

# Raw fields copied straight through, no arithmetic.
FIELDS = [
    "t_now", "R2", "v2", "current_phase",
    "Qi", "shell_fAbsorbedIon",
    "Pb", "P_ram", "P_HII", "P_drive", "press_HII_in",
    "n_IF_Str", "n_IF_Str_raw", "shell_n0",
    "F_grav", "F_rad", "F_HII", "F_ram_wind", "F_ram_SN", "F_ion_in",
    "Lmech_total", "v_mech_total", "Eb", "R1",
]


def rows(run_dir):
    d = Path(run_dir) / "dictionary.jsonl"
    if not d.exists():
        raise SystemExit(f"no dictionary.jsonl under {run_dir}")
    with d.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="label=run_dir, repeatable (e.g. stock=outputs/.../B3M)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", help="one-line provenance note written into the header")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        for spec in args.arm:
            label, _, run_dir = spec.partition("=")
            fh.write(f"# arm {label}: {run_dir}\n")
        if args.note:
            fh.write(f"# {args.note}\n")
        w = csv.writer(fh)
        w.writerow(["arm"] + FIELDS)
        for spec in args.arm:
            label, _, run_dir = spec.partition("=")
            n = 0
            for r in rows(run_dir):
                # A missing key is written empty rather than 0.0: the two arms have
                # different schemas (n_IF_Str_raw is post-Batch-1 only) and a silent
                # zero would read as a measured value in the figure.
                w.writerow([label] + [r.get(k, "") for k in FIELDS])
                n += 1
            print(f"{label:6s} {n:5d} rows  <- {run_dir}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
