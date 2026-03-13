# -*- coding: utf-8 -*-
"""
Shared boilerplate and utilities for ``src._plots`` paper scripts.

Provides:
- ``FIG_DIR``: canonical output directory (``<project_root>/fig/``)
- ``smooth_1d`` / ``smooth_2d``: moving-average smoothers
- Automatic loading of ``trinity.mplstyle``

Importing this module is sufficient to set up paths and plot style.
"""

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Project root and sys.path
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
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
