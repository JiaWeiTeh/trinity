# -*- coding: utf-8 -*-
"""
Shared I/O helpers for ``src._calc`` analysis scripts.

Provides the common ``extract_rejected`` helper and the
``regenerate_summary_pdf`` hook used by individual scripts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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
