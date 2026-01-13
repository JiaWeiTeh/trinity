"""
TRINITY plotting style utilities.

Provides centralized matplotlib styling for scientific publications,
compatible with A&A and MNRAS journal standards.

Usage:
    from plot_style import apply_style, SINGLE_COLUMN, DOUBLE_COLUMN
    apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 2.5))

    # Or use convenience functions:
    from plot_style import figure_single, figure_double
    fig, ax = figure_single()  # Creates 3.5" wide figure
    fig, ax = figure_double()  # Creates 7.1" wide figure

Author: Claude Code
Date: 2026-01-13
"""

import os
import matplotlib.pyplot as plt

# =============================================================================
# A&A/MNRAS COLUMN WIDTHS
# =============================================================================
SINGLE_COLUMN = 3.5   # inches (88mm) - single column width
DOUBLE_COLUMN = 7.1   # inches (180mm) - full page width

# =============================================================================
# STYLE FILE PATH
# =============================================================================
_STYLE_PATH = os.path.join(os.path.dirname(__file__), '..', '_plots', 'trinity.mplstyle')

# =============================================================================
# COLORBLIND-FRIENDLY COLORS
# =============================================================================
# Based on Paul Tol's color scheme (https://personal.sron.nl/~pault/)
COLORS = {
    'blue': '#0072B2',
    'orange': '#D55E00',
    'green': '#009E73',
    'pink': '#CC79A7',
    'yellow': '#F0E442',
    'cyan': '#56B4E9',
    'red': '#E69F00',
}

# Color cycle for line plots
COLOR_CYCLE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']


def apply_style(use_tex=True):
    """
    Apply TRINITY publication style to matplotlib.

    Parameters
    ----------
    use_tex : bool, optional
        Whether to use LaTeX for text rendering. Default True.
        Set to False if LaTeX is not installed.

    Example
    -------
    >>> from plot_style import apply_style
    >>> apply_style()
    >>> # Now all plots will use the TRINITY style
    """
    plt.style.use(_STYLE_PATH)
    if not use_tex:
        plt.rcParams['text.usetex'] = False


def figure_single(height_ratio=0.8, **kwargs):
    """
    Create a single-column figure (3.5" wide).

    Parameters
    ----------
    height_ratio : float, optional
        Height as fraction of width. Default 0.8 gives 3.5" x 2.8".
    **kwargs
        Additional arguments passed to plt.subplots().

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Example
    -------
    >>> from plot_style import apply_style, figure_single
    >>> apply_style()
    >>> fig, ax = figure_single()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    """
    return plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN * height_ratio), **kwargs)


def figure_double(height_ratio=0.5, **kwargs):
    """
    Create a double-column figure (7.1" wide).

    Parameters
    ----------
    height_ratio : float, optional
        Height as fraction of width. Default 0.5 gives 7.1" x 3.55".
    **kwargs
        Additional arguments passed to plt.subplots().

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Example
    -------
    >>> from plot_style import apply_style, figure_double
    >>> apply_style()
    >>> fig, axes = figure_double(ncols=2)  # Two side-by-side panels
    """
    return plt.subplots(figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * height_ratio), **kwargs)


def get_color(name):
    """
    Get a colorblind-friendly color by name.

    Parameters
    ----------
    name : str
        Color name: 'blue', 'orange', 'green', 'pink', 'yellow', 'cyan', 'red'

    Returns
    -------
    str
        Hex color code

    Example
    -------
    >>> from plot_style import get_color
    >>> ax.plot(x, y, color=get_color('blue'))
    """
    return COLORS.get(name.lower(), '#000000')
