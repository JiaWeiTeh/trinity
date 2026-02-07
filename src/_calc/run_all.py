#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all (or selected) analysis scripts in src/_calc/.

Dispatches a folder path and shared flags to each registered script.
Scripts can be included or excluded via --only / --skip flags.

Usage
-----
    # Run everything
    python run_all.py -F /path/to/sweep

    # Run with shared flags forwarded to every sub-script
    python run_all.py -F /path/to/sweep --nCore-ref 1e4 --fmt png

    # Run only two scripts
    python run_all.py -F /path/to/sweep --only scaling_phases energy_retention

    # Run everything except one
    python run_all.py -F /path/to/sweep --skip velocity_radius

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
# Each entry: (short_name, module_path relative to project root)
# Add or remove entries here to control which scripts are available.

SCRIPTS = [
    ("scaling_phases",      "src/_calc/scaling_phases.py"),
    ("collapse_criterion",  "src/_calc/collapse_criterion.py"),
    ("terminal_momentum",   "src/_calc/terminal_momentum.py"),
    ("velocity_radius",     "src/_calc/velocity_radius.py"),
    ("dispersal_timescale", "src/_calc/dispersal_timescale.py"),
    ("energy_retention",    "src/_calc/energy_retention.py"),
]

# Flags that are forwarded verbatim to every sub-script.
# Each sub-script's argparse silently ignores flags it doesn't recognise
# because we call them with parse_known_args—but here we simply forward
# the raw extra arguments, so unknown flags will cause the sub-script to
# error (which is the desired behaviour: fail loudly).
SHARED_FLAGS = [
    "--nCore-ref",
    "--mCloud-ref",
    "--sfe-ref",
    "--sigma-clip",
    "--fmt",
]


# ======================================================================
# Helpers
# ======================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve(rel_path: str) -> Path:
    return PROJECT_ROOT / rel_path


def _available_names() -> List[str]:
    return [name for name, _ in SCRIPTS]


def _print_list() -> None:
    print("Available analysis scripts:")
    print()
    for name, path in SCRIPTS:
        exists = _resolve(path).exists()
        status = "  OK" if exists else "  MISSING"
        print(f"  {name:<25} {path:<45} {status}")
    print()


# ======================================================================
# Runner
# ======================================================================

def run_scripts(
    folder: str,
    only: Optional[List[str]],
    skip: Optional[List[str]],
    extra_args: List[str],
    dry_run: bool = False,
) -> int:
    """
    Run selected scripts, forwarding ``-F folder`` and any extra flags.

    Returns the number of scripts that failed (0 = all OK).
    """
    # Resolve which scripts to run
    names_available = _available_names()

    if only:
        unknown = set(only) - set(names_available)
        if unknown:
            print(f"ERROR: unknown script(s): {', '.join(sorted(unknown))}")
            print(f"       available: {', '.join(names_available)}")
            return 1
        to_run = [(n, p) for n, p in SCRIPTS if n in only]
    else:
        to_run = list(SCRIPTS)

    if skip:
        to_run = [(n, p) for n, p in to_run if n not in skip]

    if not to_run:
        print("Nothing to run.")
        return 0

    print(f"Running {len(to_run)} script(s) on: {folder}")
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

    # Summary
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
    script_names = ", ".join(_available_names())
    parser = argparse.ArgumentParser(
        description="Run all (or selected) analysis scripts in src/_calc/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
        Available scripts: {script_names}

        Examples:
          python run_all.py -F /data/sweep
          python run_all.py -F /data/sweep --fmt png --sigma-clip 2.5
          python run_all.py -F /data/sweep --only scaling_phases energy_retention
          python run_all.py -F /data/sweep --skip velocity_radius
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
        "--list", action="store_true",
        help="List available scripts and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )

    # Shared flags — parsed here so they appear in --help,
    # then forwarded to sub-scripts.
    shared = parser.add_argument_group("shared flags (forwarded to sub-scripts)")
    shared.add_argument("--nCore-ref", type=float, default=None,
                        help="Reference nCore [cm^-3].")
    shared.add_argument("--mCloud-ref", type=float, default=None,
                        help="Reference mCloud [Msun].")
    shared.add_argument("--sfe-ref", type=float, default=None,
                        help="Reference SFE.")
    shared.add_argument("--sigma-clip", type=float, default=None,
                        help="Sigma-clipping threshold.")
    shared.add_argument("--fmt", type=str, default=None,
                        help="Output figure format (e.g. pdf, png).")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        _print_list()
        return 0

    if not args.folder:
        parser.error("-F / --folder is required (or use --list)")

    # Build the extra-args list from shared flags that were actually set
    extra: List[str] = []
    for flag in SHARED_FLAGS:
        attr = flag.lstrip("-").replace("-", "_")
        val = getattr(args, attr, None)
        if val is not None:
            extra.extend([flag, str(val)])

    n_fail = run_scripts(
        folder=args.folder,
        only=args.only,
        skip=args.skip,
        extra_args=extra,
        dry_run=args.dry_run,
    )
    return min(n_fail, 1)   # cap at 1 for exit code


if __name__ == "__main__":
    raise SystemExit(main())
