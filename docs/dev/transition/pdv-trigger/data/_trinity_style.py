"""Shared TRINITY plot style for the pdv-trigger storyline figures — one source of truth.

Loads the canonical `paper/_lib/trinity.mplstyle` (serif, in-direction ticks, minor ticks, frameless
legend, A&A/MNRAS look) so every figure in this workstream is consistent, then overrides the LaTeX
settings: this container has NO system LaTeX, so `text.usetex:True` (the .mplstyle default) would error.
We fall back to mathtext with the Computer-Modern font set, which keeps the serif/CM aesthetic without
a LaTeX install. Call `use_trinity_style()` once at import time in each figure script.

    from _trinity_style import use_trinity_style, COLORS
    use_trinity_style()
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
# data/ -> pdv-trigger -> transition -> dev -> docs -> repo root, then paper/_lib/trinity.mplstyle
_REPO = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 5)))
_MPLSTYLE = os.path.join(_REPO, "paper", "_lib", "trinity.mplstyle")

# A small, fixed semantic palette so colours mean the same thing across every figure in the storyline.
COLORS = {
    "fk1": "#1f77b4",      # f_kappa = 1 (baseline)
    "fk2": "#9467bd",      # f_kappa = 2
    "fk4": "#d62728",      # f_kappa = 4
    "pdv": "#ff7f0e",      # the PdV term / dominant sink
    "fire": "#2ca02c",     # a trigger threshold / firing
    "cool": "#d62728",     # the cooling_balance (radiative) threshold
    "compact": "#1f77b4",
    "mid": "#9467bd",
    "diffuse": "#ff7f0e",
    "dense": "#8c564b",
}


def use_trinity_style():
    """Apply the trinity.mplstyle aesthetic, LaTeX-free (mathtext CM fallback)."""
    if os.path.exists(_MPLSTYLE):
        plt.style.use(_MPLSTYLE)
    # Override the LaTeX-dependent keys (no system latex in this container) while keeping the
    # serif / Computer-Modern look via mathtext.
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        # the .mplstyle sizes (labelsize/titlesize 15) target single-column paper panels (figsize
        # 3.5x2.8); storyline figures are multi-panel with longer labels, so scale the TYPE down to
        # storyline-appropriate sizes (matching the workstream's make_storyline_figs.py convention)
        # while KEEPING the trinity aesthetic (serif, in-direction ticks, minor ticks, frameless legend).
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": False,
    })
