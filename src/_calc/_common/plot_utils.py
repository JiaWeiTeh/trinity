# -*- coding: utf-8 -*-
"""
Shared plotting constants and helpers for ``src._calc`` analysis scripts.

Centralises figure-output paths, the trinity style sheet, marker/color
palettes, and outcome styling so every script draws from the same source.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Output directory: ./fig/ at project root
# ------------------------------------------------------------------
FIG_DIR = Path(__file__).parent.parent.parent.parent / "fig"

# ------------------------------------------------------------------
# Apply the trinity plot style if available
# ------------------------------------------------------------------
_STYLE_PATH = Path(__file__).parent.parent.parent / "_plots" / "trinity.mplstyle"
if _STYLE_PATH.exists():
    plt.style.use(str(_STYLE_PATH))

# ------------------------------------------------------------------
# Marker cycle (used by most scripts for nCore differentiation)
# ------------------------------------------------------------------
MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")

# ------------------------------------------------------------------
# Colour-blind-safe palette (Wong 2011)
# ------------------------------------------------------------------
C_BLUE = "#0072B2"
C_VERMILLION = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_BLACK = "#000000"

# ------------------------------------------------------------------
# Outcome categories and their visual styles
# ------------------------------------------------------------------
EXPAND = "expand"
COLLAPSE = "collapse"
STALLED = "stalled"

OUTCOME_COLORS = {EXPAND: "C0", COLLAPSE: "C3", STALLED: "0.55"}
OUTCOME_LABELS = {EXPAND: "Expand", COLLAPSE: "Collapse", STALLED: "Stalled"}
OUTCOME_MARKERS = {EXPAND: "o", COLLAPSE: "x", STALLED: "+"}
