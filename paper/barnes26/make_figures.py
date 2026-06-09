#!/usr/bin/env python3
"""Regenerate every Barnes 2026 (PHANGS) comparison figure with one command.

Sibling of ``paper/methods/make_figures.py`` (the methods-paper driver), but pointed
at a folder of TRINITY runs rather than ``.npz`` bundles: the barnes26 scripts
read live simulation output via ``-F``, so this driver forwards a runs folder
to each of them and funnels the PDFs into ``paper/barnes26/plots/``.

    python paper/barnes26/make_figures.py -F outputs/<sweep>

To regenerate a single figure, pass its short name (or any unique prefix):

    python paper/barnes26/make_figures.py -F outputs/<sweep> PradSources

The figures use the scripts' default stellar ages (0.5, 1, 3 Myr). For other
ages, run the individual script directly with its ``--ages`` flag, e.g.::

    python -m paper.barnes26.paper_PradSources -F outputs/<sweep> --ages 0.5 1

Each run folder under ``-F`` must contain ``dictionary.jsonl``; see the
individual scripts for the per-figure data requirements (e.g. ``PISM > 0``
for the pressure-balance figure).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = REPO_ROOT / "paper" / "barnes26" / "plots"

sys.path.insert(0, str(REPO_ROOT))
from trinity._output.trinity_reader import find_all_simulations  # noqa: E402


# ``args(folder, outdir)`` builds the trailing arguments after
# ``python -m <module>``. ``folder`` is the runs directory passed via -F.
FIGURES = [
    dict(
        name="PradSources",
        description="P_rad vs L_bol / M_star / radius, one row per age (R2 + R_IF variants)",
        module="paper.barnes26.paper_PradSources",
        args=lambda folder, outdir: [
            "-F", str(folder), "-o", str(outdir), "--radius", "both",
        ],
    ),
    dict(
        name="PressureBalance",
        description="P_ISM vs P_tot and P_tot-P_ISM vs P_tot, one column per age",
        module="paper.barnes26.paper_PressureBalance",
        args=lambda folder, outdir: [
            "-F", str(folder), "-o", str(outdir), "--prad", "both",
        ],
    ),
]


def _select(requested):
    """Filter ``FIGURES`` by command-line names (unique prefix match)."""
    if not requested:
        return list(FIGURES)
    selected = []
    for token in requested:
        matches = [f for f in FIGURES if f["name"].startswith(token)]
        if not matches:
            sys.exit(f"Unknown figure '{token}'. Available: "
                     + ", ".join(f["name"] for f in FIGURES))
        if len(matches) > 1:
            sys.exit(f"Ambiguous prefix '{token}'. Matches: "
                     + ", ".join(m["name"] for m in matches))
        selected.append(matches[0])
    return selected


def _run_one(fig, folder):
    cmd = [sys.executable, "-m", fig["module"], *fig["args"](folder, PLOTS_DIR)]
    print(f"\n[{fig['name']}] {fig['description']}")
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate the Barnes 2026 comparison figures from a runs folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("names", nargs="*",
                        help="Figure name(s) to run (unique prefix); default all")
    parser.add_argument("-F", "--folder", required=True,
                        help="Folder of TRINITY run subfolders (each with dictionary.jsonl)")
    args = parser.parse_args(argv)

    if not find_all_simulations(args.folder):
        sys.exit(f"No TRINITY runs (dictionary.jsonl) found under: {args.folder}")

    figures = _select(args.names)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for fig in figures:
        rc = _run_one(fig, args.folder)
        if rc != 0:
            failed.append((fig, rc))

    print("\n" + "=" * 60)
    print(f"Ran {len(figures)} figure(s); output in "
          f"{PLOTS_DIR.relative_to(REPO_ROOT)}/")
    if failed:
        print("Failed:")
        for fig, rc in failed:
            print(f"  - {fig['name']:<18}  exit code {rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
