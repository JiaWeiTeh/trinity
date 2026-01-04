#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 18:41:55 2026

@author: Jia Wei Teh

Use: python3 generate_params.py --outdir param --pism 1e6
Use: python3 generate_params.py --outdir param
"""

from __future__ import annotations

from pathlib import Path
import argparse

# ---- EDIT YOUR GRID HERE ----
MCLOUD_LIST = ["1e5", "1e7", "1e8"]
NDENS_LIST  = ["1e2", "1e3", "1e4"]              # maps to nCore
SFE_TAGS    = ["001", "010", "020", "030", "050", "080"]

# Common (fixed) parameters for all runs
BASE_PARAMS = [
    ("# powerlaw, homogeneous", None),
    ("dens_profile", "densPL"),
    ("densPL_alpha", "0"),
    ("nISM", "0.1"),
    ("expansionBeyondCloud", "True"),
]
# ----------------------------


def sfe_tag_to_value(tag: str) -> str:
    # "020" -> "0.20"
    return f"{int(tag) / 100:.2f}"


def write_param_file(path: Path, mcloud: str, sfe_tag: str, ncore: str, extra: dict[str, str]):
    lines: list[str] = []
    lines.append(f"mCloud    {mcloud}")
    lines.append(f"sfe    {sfe_tag_to_value(sfe_tag)}")

    for key, val in BASE_PARAMS:
        if val is None:  # comment line
            lines.append(key)
        else:
            lines.append(f"{key}    {val}")

    lines.append(f"nCore    {ncore}")

    # add extra parameters at the end (e.g., PISM)
    for k, v in extra.items():
        lines.append(f"{k}    {v}")

    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Generate Trinity .param files for a parameter grid.")
    ap.add_argument("--outdir", default="params", help="Output directory for .param files")
    ap.add_argument("--pism", default=None, help="If set, append 'PISM <value>' to every file (e.g. 1e6)")
    ap.add_argument("--suffix", default="", help="Optional suffix to add to filenames (e.g. _PISM1e6)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extra: dict[str, str] = {}
    suffix = args.suffix
    if args.pism is not None:
        extra["PISM"] = str(args.pism)
        if not suffix:
            suffix = f"_PISM{args.pism}"

    count = 0
    for m in MCLOUD_LIST:
        for sfe_tag in SFE_TAGS:
            for n in NDENS_LIST:
                fname = f"{m}_sfe{sfe_tag}_n{n}{suffix}.param"
                write_param_file(outdir / fname, m, sfe_tag, n, extra)
                count += 1

    print(f"Wrote {count} param files to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
