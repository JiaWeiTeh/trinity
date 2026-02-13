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
import json
import textwrap
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    ("bubble_distribution", "src/_calc/bubble_distribution.py"),
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
    "--t-end",
]

# Boolean flags forwarded verbatim (store_true: no value argument).
SHARED_BOOL_FLAGS = [
    "--diagnostics",
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
# Equation summary PDF
# ======================================================================

# LaTeX-friendly labels for parameter names
_PARAM_LATEX = {
    "nCore": r"n_c",
    "mCloud": r"M_{\rm cl}",
    "sfe": r"\varepsilon",
    "Sigma": r"\Sigma",
}

# Quality tiers  (label, colour, R² lower bound)
_TIERS = [
    ("Really good",  "#1b7837", 0.95),
    ("Good",         "#5ab4ac", 0.85),
    ("Average",      "#d8b365", 0.70),
    ("Bad",          "#c51b7d", -1.0),
]


def _build_latex_equation(entry: dict) -> str:
    """Build a LaTeX equation string from an equation-JSON entry."""
    A = entry["A"]
    exponents = entry["exponents"]
    refs = entry["refs"]
    is_linear = entry.get("linear_fit", False)

    if is_linear:
        # velocity_radius: alpha = a0 + sum(a_i * log(param/ref))
        parts = [f"{A:.3f}"]
        for pname, exp in exponents.items():
            lab = _PARAM_LATEX.get(pname, pname)
            ref = refs.get(pname, 1.0)
            ref_str = f"{ref:.0e}"
            parts.append(
                rf"{exp:+.2f}\,\log_{{10}}"
                rf"\!\left(\frac{{{lab}}}{{{ref_str}}}\right)"
            )
        return " ".join(parts)

    # Standard log-space power law:  Y = A * prod( (param/ref)^exp )
    parts = [f"{A:.3g}"]
    for pname, exp in exponents.items():
        if abs(exp) < 1e-6:
            continue
        lab = _PARAM_LATEX.get(pname, pname)
        ref = refs.get(pname, 1.0)
        ref_str = f"{ref:.0e}"
        parts.append(
            rf"\left(\frac{{{lab}}}{{{ref_str}}}\right)"
            rf"^{{{exp:+.2f}}}"
        )
    return r"\;\cdot\;".join(parts)


def generate_summary_pdf(output_dir: Path, fmt: str = "pdf") -> Optional[Path]:
    """
    Read all *_equations.json files in *output_dir* and produce a
    single-page summary with LaTeX equations grouped by fit quality.
    """
    # Collect all equation entries
    json_files = sorted(output_dir.glob("*_equations.json"))
    if not json_files:
        print("No equation JSON files found — skipping summary PDF.")
        return None

    all_entries = []
    for jf in json_files:
        try:
            with open(jf) as fh:
                entries = json.load(fh)
            all_entries.extend(entries)
        except Exception as exc:
            print(f"  Warning: could not read {jf.name}: {exc}")

    if not all_entries:
        print("No valid equations found — skipping summary PDF.")
        return None

    # Sort by R² descending
    all_entries.sort(key=lambda e: e.get("R2", 0), reverse=True)

    # Assign tiers
    tiered: dict = {label: [] for label, _, _ in _TIERS}
    for entry in all_entries:
        r2 = entry.get("R2", 0)
        for label, _, lb in _TIERS:
            if r2 >= lb:
                tiered[label].append(entry)
                break

    # --- Render figure ---
    # Use inch-based spacing so the layout scales with content.
    TITLE_IN = 0.50       # main title height
    TIER_HDR_IN = 0.35    # tier header height
    EQ_LINE_IN = 0.25     # per equation line
    TIER_GAP_IN = 0.12    # gap between tiers
    MARGIN_IN = 0.40      # top + bottom margin combined

    n_eqs = len(all_entries)
    n_tiers = sum(1 for label, _, _ in _TIERS if tiered[label])
    fig_height = max(4.0,
                     MARGIN_IN + TITLE_IN
                     + n_tiers * (TIER_HDR_IN + TIER_GAP_IN)
                     + n_eqs * EQ_LINE_IN)

    fig, ax = plt.subplots(figsize=(11, fig_height), dpi=150)
    ax.axis("off")

    # Convert inch spacings to figure fractions
    dy_title = TIER_HDR_IN / fig_height
    dy_eq = EQ_LINE_IN / fig_height
    dy_gap = TIER_GAP_IN / fig_height

    y = 1.0 - (MARGIN_IN * 0.5) / fig_height   # start below top margin

    # Title
    ax.text(0.5, y, "TRINITY Scaling-Relation Summary",
            transform=ax.transAxes, fontsize=14, ha="center", va="top",
            fontweight="bold")
    y -= TITLE_IN / fig_height

    for tier_label, tier_color, _ in _TIERS:
        entries = tiered[tier_label]
        if not entries:
            continue

        # Tier header
        r2_range = f"$R^2 \\geq {_TIERS[[l for l,_,_ in _TIERS].index(tier_label)][2]:.2f}$"
        if tier_label == "Bad":
            r2_range = "$R^2 < 0.70$"
        ax.text(0.02, y,
                f"{tier_label}  ({r2_range}, {len(entries)} fit{'s' if len(entries)!=1 else ''})",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                color=tier_color, va="top")
        y -= dy_title

        for entry in entries:
            latex_eq = _build_latex_equation(entry)
            label = entry["label"]
            r2 = entry.get("R2", float("nan"))
            rms = entry.get("rms_dex", float("nan"))
            n_used = entry.get("n_used", "?")
            script = entry.get("script", "")

            line = (f"[{script}]  {label}  $\\approx$  ${latex_eq}$"
                    f"     ($R^2={r2:.3f}$, RMS$={rms:.3f}$ dex, $N={n_used}$)")
            ax.text(0.04, y, line,
                    transform=ax.transAxes, fontsize=8, va="top",
                    family="monospace", color="0.15")
            y -= dy_eq

        y -= dy_gap  # gap between tiers

    fig.tight_layout()
    out_path = output_dir / f"scaling_relations_summary.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved scaling-relations summary: {out_path}")
    return out_path


# ======================================================================
# Sigma-clip rejection summary
# ======================================================================

def _run_key(info: dict) -> str:
    """Build a human-readable key for a rejected run from its identifying info."""
    parts = []
    if "folder" in info:
        return info["folder"]
    if "mCloud" in info:
        parts.append(f"M={info['mCloud']:.0e}")
    if "nCore" in info:
        parts.append(f"n={info['nCore']:.0e}")
    if "sfe" in info:
        parts.append(f"sfe={info['sfe']:.3f}")
    if "Sigma" in info:
        parts.append(f"Sigma={info['Sigma']:.1f}")
    return ", ".join(parts) if parts else "unknown"


def generate_rejection_summary(output_dir: Path) -> None:
    """
    Read all *_equations.json in *output_dir* and print a summary of
    runs rejected by sigma-clipping, grouped by run.
    """
    json_files = sorted(output_dir.glob("*_equations.json"))
    if not json_files:
        return

    # Collect: run_key -> list of (script, label) that rejected it
    from collections import defaultdict
    rejections_by_run = defaultdict(list)
    total_rejected = 0

    for jf in json_files:
        try:
            with open(jf) as fh:
                entries = json.load(fh)
        except Exception:
            continue
        for entry in entries:
            script = entry.get("script", jf.stem)
            label = entry.get("label", "?")
            n_rej = entry.get("n_rejected", 0)
            rejected = entry.get("rejected", [])
            for info in rejected:
                key = _run_key(info)
                rejections_by_run[key].append(f"[{script}] {label}")
                total_rejected += 1

    if total_rejected == 0:
        print("\nSigma-clip rejection summary: no runs were rejected.")
        return

    # Print summary
    print()
    print("=" * 90)
    print("SIGMA-CLIP REJECTION SUMMARY")
    print("=" * 90)
    print(f"  Total rejections: {total_rejected} (across all fits)")
    print(f"  Unique runs affected: {len(rejections_by_run)}")
    print()

    # Sort by number of rejections (most rejected first)
    for key in sorted(rejections_by_run, key=lambda k: -len(rejections_by_run[k])):
        fits = rejections_by_run[key]
        print(f"  {key}  ({len(fits)} rejection{'s' if len(fits) != 1 else ''}):")
        for fit_label in fits:
            print(f"      {fit_label}")
        print()

    print("=" * 90)


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
    shared.add_argument("--t-end", type=float, default=None,
                        help="Maximum time [Myr] to consider in calculations.")
    shared.add_argument("--diagnostics", action="store_true", default=False,
                        help="Generate diagnostic plots in sub-scripts.")

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
    for flag in SHARED_BOOL_FLAGS:
        attr = flag.lstrip("-").replace("-", "_")
        if getattr(args, attr, False):
            extra.append(flag)

    n_fail = run_scripts(
        folder=args.folder,
        only=args.only,
        skip=args.skip,
        extra_args=extra,
        dry_run=args.dry_run,
    )

    # Generate scaling-relations summary PDF (unless dry-run)
    if not args.dry_run:
        folder_name = Path(args.folder).name
        output_dir = PROJECT_ROOT / "fig" / folder_name
        fmt = args.fmt if args.fmt else "pdf"
        if output_dir.is_dir():
            generate_summary_pdf(output_dir, fmt=fmt)
            generate_rejection_summary(output_dir)

    return min(n_fail, 1)   # cap at 1 for exit code


if __name__ == "__main__":
    raise SystemExit(main())
