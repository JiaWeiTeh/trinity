#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all (or selected) plotting scripts in src/_plots/.

Dispatches a folder path and shared flags to each registered paper_* script.
Scripts can be included or excluded via --only / --skip flags.

By default only a curated subset is enabled (paper_feedback,
paper_momentum, paper_thermalRegime).  Use --all to run every
registered script, or --only to pick specific ones.

Usage
-----
    # Run default set
    python run_all.py -F /path/to/sweep

    # Run with shared flags forwarded to every sub-script
    python run_all.py -F /path/to/sweep -n 1e4 -o ./figures

    # Run all registered scripts (not just the default set)
    python run_all.py -F /path/to/sweep --all

    # Run only specific scripts
    python run_all.py -F /path/to/sweep --only paper_feedback paper_ODIN

    # Run default set minus one
    python run_all.py -F /path/to/sweep --skip paper_thermalRegime

    # List available scripts without running
    python run_all.py --list
"""

import sys
import subprocess
import argparse
import textwrap
from pathlib import Path
from typing import List, Optional


# ======================================================================
# Registry
# ======================================================================
# (short_name, module_path, default_on)
#   default_on=True  → included unless --skip or --only overrides
#   default_on=False → only included with --all or --only

SCRIPTS = [
    ("paper_feedback",               "src/_plots/paper_feedback.py",               True),
    ("paper_momentum",               "src/_plots/paper_momentum.py",               True),
    ("paper_thermalRegime",          "src/_plots/paper_thermalRegime.py",           True),
    ("paper_accelerationDecomposition", "src/_plots/paper_accelerationDecomposition.py", False),
    ("paper_AllowedGMC",             "src/_plots/paper_AllowedGMC.py",              False),
    ("paper_BEprofile",              "src/_plots/paper_BEprofile.py",               False),
    ("paper_bestFitOrion",           "src/_plots/paper_bestFitOrion.py",            False),
    ("paper_betadelta",              "src/_plots/paper_betadelta.py",               False),
    ("paper_bubblePhase",            "src/_plots/paper_bubblePhase.py",             False),
    ("paper_cancellationMetric",     "src/_plots/paper_cancellationMetric.py",      False),
    ("paper_dominantFeedback",       "src/_plots/paper_dominantFeedback.py",        False),
    ("paper_escapeFraction",         "src/_plots/paper_escapeFraction.py",          False),
    ("paper_expansionVelocity",      "src/_plots/paper_expansionVelocity.py",       False),
    ("paper_forceFraction",          "src/_plots/paper_forceFraction.py",           False),
    ("paper_InitialCloudRadius",     "src/_plots/paper_InitialCloudRadius.py",      False),
    ("paper_LbolLWind",              "src/_plots/paper_LbolLWind.py",               False),
    ("paper_ODIN",                   "src/_plots/paper_ODIN.py",                    False),
    ("paper_PISM",                   "src/_plots/paper_PISM.py",                    False),
    ("paper_PPVtest",                "src/_plots/paper_PPVtest.py",                 False),
    ("paper_pressureEvolution",      "src/_plots/paper_pressureEvolution.py",       False),
    ("paper_radiusEvolution",        "src/_plots/paper_radiusEvolution.py",         False),
]

# Shared flags forwarded to sub-scripts.
# paper_* scripts use -n/--nCore, -o/--output-dir, etc.
SHARED_FLAGS = [
    ("-n",  "--nCore"),
    ("-o",  "--output-dir"),
    ("--mCloud",),
    ("--sfe",),
]


# ======================================================================
# Helpers
# ======================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve(rel_path: str) -> Path:
    return PROJECT_ROOT / rel_path


def _available_names() -> List[str]:
    return [name for name, _, _ in SCRIPTS]


def _default_names() -> List[str]:
    return [name for name, _, on in SCRIPTS if on]


def _print_list() -> None:
    print("Available plotting scripts:")
    print()
    for name, path, default_on in SCRIPTS:
        exists = _resolve(path).exists()
        tag = "DEFAULT" if default_on else "       "
        status = "OK" if exists else "MISSING"
        print(f"  {name:<35} {tag}  {status}")
    print()
    print(f"Default set: {', '.join(_default_names())}")
    print("Use --all to include non-default scripts, or --only to pick specific ones.")
    print()


# ======================================================================
# Runner
# ======================================================================

def run_scripts(
    folder: str,
    only: Optional[List[str]],
    skip: Optional[List[str]],
    run_all_flag: bool,
    extra_args: List[str],
    dry_run: bool = False,
) -> int:
    """
    Run selected scripts, forwarding ``-F folder`` and any extra flags.

    Returns the number of scripts that failed (0 = all OK).
    """
    names_available = _available_names()

    if only:
        unknown = set(only) - set(names_available)
        if unknown:
            print(f"ERROR: unknown script(s): {', '.join(sorted(unknown))}")
            print(f"       available: {', '.join(names_available)}")
            return 1
        to_run = [(n, p) for n, p, _ in SCRIPTS if n in only]
    elif run_all_flag:
        to_run = [(n, p) for n, p, _ in SCRIPTS]
    else:
        to_run = [(n, p) for n, p, on in SCRIPTS if on]

    if skip:
        to_run = [(n, p) for n, p in to_run if n not in skip]

    if not to_run:
        print("Nothing to run.")
        return 0

    print(f"Running {len(to_run)} plotting script(s) on: {folder}")
    print()

    n_fail = 0
    for name, rel_path in to_run:
        script = _resolve(rel_path)
        if not script.exists():
            print(f"  [{name}] SKIP — file not found: {script}")
            continue

        cmd = [sys.executable, str(script), "-F", folder] + extra_args

        if dry_run:
            print(f"  [{name}] DRY RUN: {' '.join(cmd)}")
            continue

        print(f"  [{name}] running ...", flush=True)
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"  [{name}] FAILED (exit code {result.returncode})")
            n_fail += 1
        else:
            print(f"  [{name}] done")
        print()

    n_ok = len(to_run) - n_fail
    if dry_run:
        print(f"Dry run complete: {len(to_run)} script(s) would be executed.")
    elif n_fail == 0:
        print(f"All {n_ok} script(s) completed successfully.")
    else:
        print(f"{n_ok}/{len(to_run)} succeeded, {n_fail} failed.")

    return n_fail


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    default_names = ", ".join(_default_names())
    parser = argparse.ArgumentParser(
        description="Run all (or selected) plotting scripts in src/_plots/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
        Default scripts: {default_names}

        Examples:
          python run_all.py -F /data/sweep
          python run_all.py -F /data/sweep -n 1e4 -o ./figures
          python run_all.py -F /data/sweep --all
          python run_all.py -F /data/sweep --only paper_feedback paper_ODIN
          python run_all.py -F /data/sweep --skip paper_thermalRegime
          python run_all.py --list
        """),
    )
    parser.add_argument(
        "-F", "--folder",
        help="Path to the sweep output directory tree (required unless --list).",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="Run only these scripts (space-separated short names).",
    )
    parser.add_argument(
        "--skip", nargs="+", metavar="NAME",
        help="Skip these scripts (space-separated short names).",
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all",
        help="Run all registered scripts, not just the default set.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available scripts and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )

    # Shared flags — forwarded to sub-scripts
    shared = parser.add_argument_group(
        "shared flags (forwarded to sub-scripts)")
    shared.add_argument("-n", "--nCore", default=None,
                        help="Filter by cloud density (e.g. 1e4).")
    shared.add_argument("-o", "--output-dir", default=None,
                        help="Output directory override.")
    shared.add_argument("--mCloud", nargs="+", default=None,
                        help="Filter by cloud mass (one or more values).")
    shared.add_argument("--sfe", nargs="+", default=None,
                        help="Filter by SFE (one or more values).")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        _print_list()
        return 0

    if not args.folder:
        parser.error("-F / --folder is required (or use --list)")

    # Build extra-args from shared flags
    extra: List[str] = []
    if args.nCore is not None:
        extra.extend(["-n", args.nCore])
    if args.output_dir is not None:
        extra.extend(["-o", args.output_dir])
    if args.mCloud is not None:
        extra.append("--mCloud")
        extra.extend(args.mCloud)
    if args.sfe is not None:
        extra.append("--sfe")
        extra.extend(args.sfe)

    n_fail = run_scripts(
        folder=args.folder,
        only=args.only,
        skip=args.skip,
        run_all_flag=args.run_all,
        extra_args=extra,
        dry_run=args.dry_run,
    )
    return min(n_fail, 1)


if __name__ == "__main__":
    raise SystemExit(main())
