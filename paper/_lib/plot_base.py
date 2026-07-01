# -*- coding: utf-8 -*-
"""
Shared boilerplate and utilities for ``paper.*.figures`` paper scripts.

Provides:
- ``FIG_DIR``: canonical output directory (``<project_root>/fig/``)
- ``smooth_1d`` / ``smooth_2d``: moving-average smoothers
- Automatic loading of ``trinity.mplstyle``

Importing this module is sufficient to set up paths and plot style.
"""

import sys
import os
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Project root and sys.path
# ------------------------------------------------------------------
# paper/_lib/plot_base.py -> parents[2] is the repo root. (Was
# parent*4, correct only when this lived at paper/figures/_lib/; the
# per-paper reorg moved _lib up a level, which made it overshoot to the
# directory *above* the repo and create a stray fig/ there.)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# Output directory
# ------------------------------------------------------------------
FIG_DIR = PROJECT_ROOT / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Apply trinity plot style
# ------------------------------------------------------------------
_STYLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trinity.mplstyle")
if os.path.exists(_STYLE_PATH):
    plt.style.use(_STYLE_PATH)
    # trinity.mplstyle sets text.usetex=True for the publication look, but that
    # shells out to `latex`. On a node with no TeX toolchain (Helix compute nodes;
    # tools/cluster/matplotlibrc sets usetex=False but the style above re-enables
    # it) fall back to mathtext so rendering still works — mathtext.fontset=cm keeps
    # the Computer-Modern look. No effect where latex exists.
    if shutil.which("latex") is None:
        plt.rcParams["text.usetex"] = False


# ------------------------------------------------------------------
# Smoothing utilities
# ------------------------------------------------------------------
def smooth_1d(y, window, mode="edge"):
    """Moving-average 1-D smoother with edge padding."""
    if window is None or window <= 1:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode=mode)
    return np.convolve(ypad, kernel, mode="valid")


def smooth_2d(arr, window, mode="edge"):
    """Apply ``smooth_1d`` independently to each row of a 2-D array."""
    if window is None or window <= 1:
        return arr
    return np.vstack([smooth_1d(row, window, mode=mode) for row in arr])
