#!/usr/bin/env python3
"""Reproduce every paper figure from the bundled ``.npz`` files in ``paper/methods/data/``.

This is the entry point for paper reproducibility: after cloning the
repo, run

    python paper/methods/make_figures.py

and every figure whose bundle is present in ``paper/methods/data/`` will be
regenerated into ``paper/plots/`` — a single directory holding the
published paper figures, so readers don't have to hunt through the
per-run subfolders that standalone scripts write to. Bundles that
haven't been published yet are reported and skipped — the rest still run.

Each row in ``FIGURES`` below maps one published bundle to the plot
script that consumes it. To regenerate a single figure, pass its
short name (or any unique prefix):

    python paper/methods/make_figures.py teaser
    python paper/methods/make_figures.py density rcloud
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "paper" / "methods" / "data"
# Single home for the published paper figures (see module docstring).
PLOTS_DIR = REPO_ROOT / "paper" / "plots"


# ``args(bundle, outdir)`` builds the trailing arguments after
# ``python -m <module>``. The bundle path always appears verbatim so we can
# existence-check it before spawning the subprocess. ``outdir`` funnels every
# figure into ``PLOTS_DIR``; most scripts take it as a directory via ``-o``,
# but paper_rcloud_smoothing's ``-o`` is a full file path instead.
FIGURES = [
    dict(
        name="density",
        description="density profile ingredients + phase timeline",
        module="paper.methods.figures.paper_densityProfile",
        bundle=DATA_DIR / "densityProfile.npz",
        args=lambda bundle, outdir: ["--from-npz", str(bundle), "-o", str(outdir)],
    ),
    dict(
        name="teaser",
        description="teaser: R_b/v_b, feedback decomposition, Q_i budget",
        module="paper.methods.figures.paper_teaser",
        bundle=DATA_DIR / "diagnostics.npz",
        args=lambda bundle, outdir: ["--from-npz", str(bundle), "-o", str(outdir)],
    ),
    dict(
        name="radiusComparison",
        description="R(t) comparison: TRINITY vs WARPFIELD vs scaling laws",
        module="paper.methods.figures.paper_radiusComparison",
        bundle=DATA_DIR / "radiusComparison.npz",
        args=lambda bundle, outdir: ["--from-npz", str(bundle), "-o", str(outdir)],
    ),
    dict(
        name="rcloud_smoothing",
        description="rCloud smoothing + before/after LSODA trajectories",
        module="paper.methods.figures.paper_rcloud_smoothing",
        bundle=DATA_DIR / "app_LSODA.npz",
        # paper_rcloud_smoothing takes the bundle as a positional argument,
        # and its ``-o`` is a full output file path rather than a directory.
        args=lambda bundle, outdir: [str(bundle), "-o", str(outdir / "rcloud_smoothing.pdf")],
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


def _run_one(fig):
    cmd = [sys.executable, "-m", fig["module"],
           *fig["args"](fig["bundle"], PLOTS_DIR)]
    print(f"\n[{fig['name']}] {fig['description']}")
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    figures = _select(argv)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    missing, failed = [], []
    for fig in figures:
        if not fig["bundle"].exists():
            print(f"[{fig['name']}] skipped — bundle not found: "
                  f"{fig['bundle'].relative_to(REPO_ROOT)}")
            missing.append(fig)
            continue
        rc = _run_one(fig)
        if rc != 0:
            failed.append((fig, rc))

    print("\n" + "=" * 60)
    print(f"Ran {len(figures) - len(missing)} of {len(figures)} figure(s); "
          f"output in {PLOTS_DIR.relative_to(REPO_ROOT)}/")
    if missing:
        print("Skipped (bundle not yet published):")
        for fig in missing:
            print(f"  - {fig['name']:<18}  expects {fig['bundle'].relative_to(REPO_ROOT)}")
    if failed:
        print("Failed:")
        for fig, rc in failed:
            print(f"  - {fig['name']:<18}  exit code {rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
