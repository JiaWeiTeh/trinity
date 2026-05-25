# -*- coding: utf-8 -*-
"""
Shared I/O helpers for ``src._calc`` analysis scripts.

Provides the common ``extract_rejected`` helper, the
``regenerate_summary_pdf`` hook, and the PHII-variant selection
utilities used by individual scripts.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# PHII filtering is shared with the paper_* layer; importing keeps the
# semantics (which folder suffixes count as yes/no) in lockstep.
from src._plots.grid_template import (
    filter_sim_files_by_phii,
    phii_file_prefix,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# PHII variant selection (mirrors ``src._plots.cli`` for _calc scripts)
# ----------------------------------------------------------------------

def add_phii_argument(parser: argparse.ArgumentParser) -> None:
    """Register the standard ``--show-noPHII`` flag on *parser*.

    The default (flag absent) runs the analysis only on ``_yesPHII``
    folders plus any folder without a PHII suffix (legacy / untagged).
    When the flag is set, the caller is expected to additionally re-run
    the analysis on ``_noPHII`` folders and tag those outputs via
    :func:`phii_file_prefix`.
    """
    parser.add_argument(
        "--show-noPHII",
        action="store_true",
        default=False,
        dest="show_noPHII",
        help="Also analyse simulation folders ending in '_noPHII'. By "
             "default these are ignored; only '_yesPHII' and untagged "
             "(legacy) folders are processed.",
    )


def iter_phii_modes(args) -> List[str]:
    """Return the PHII modes to run for the parsed CLI ``args``.

    Always yields ``"yes"``; additionally yields ``"no"`` when
    ``args.show_noPHII`` is true.  Sub-scripts loop over the result so
    a single run produces both variants when requested.
    """
    modes: List[str] = ["yes"]
    if getattr(args, "show_noPHII", False):
        modes.append("no")
    return modes


__all__ = [
    "extract_rejected",
    "regenerate_summary_pdf",
    "add_phii_argument",
    "iter_phii_modes",
    "filter_sim_files_by_phii",
    "phii_file_prefix",
]


def extract_rejected(
    fit: Dict,
    extra_keys: Tuple[str, ...] = (),
) -> List[Dict]:
    """Extract identifying info for sigma-clipped (rejected) points.

    Parameters
    ----------
    fit : dict returned by a fitting function (must contain ``mask``).
    extra_keys : additional per-point keys to include beyond the default
        ``("nCore", "mCloud", "sfe")``.
    """
    mask = fit.get("mask")
    if mask is None:
        return []
    keys = ("nCore", "mCloud", "sfe") + tuple(extra_keys)
    rejected: List[Dict] = []
    for i, m in enumerate(mask):
        if not m:
            info: Dict = {}
            for k in keys:
                arr = fit.get(k)
                if arr is not None and i < len(arr):
                    info[k] = float(arr[i])
            flds = fit.get("folders")
            if flds is not None and i < len(flds):
                info["folder"] = flds[i]
            if info:
                rejected.append(info)
    return rejected


def regenerate_summary_pdf(output_dir: Path, fmt: str = "pdf") -> None:
    """Re-generate the cross-script scaling-relations summary PDF.

    Loads ``run_all.generate_summary_pdf`` via importlib so that an
    individual script can update the PDF without importing run_all at
    module level.
    """
    try:
        import importlib.util

        import matplotlib.pyplot as plt

        _mod_path = str(Path(__file__).resolve().parent.parent / "run_all.py")
        spec = importlib.util.spec_from_file_location("_run_all", _mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with plt.rc_context({"text.usetex": False}):
            mod.generate_summary_pdf(output_dir, fmt=fmt)
    except Exception as exc:
        logger.warning("Could not regenerate summary PDF: %s", exc)
