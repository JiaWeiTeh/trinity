#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:36:36 2023

@author: Jia Wei Teh

This script contains useful functions that help compute stuffs
"""

from functools import reduce
from typing import Tuple, Union, Sequence

import numpy as np
import src._functions.unit_conversions as cvt

def _simplify(
    x_arr: Union[np.ndarray, Sequence[float]],
    y_arr: Union[np.ndarray, Sequence[float]],
    nmin: int = 100,
    grad_inc: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic downsampling of a curve y(x) to approximately ``nmin`` points,
    preserving the most physically and visually important features.

    This is useful when a simulation or measurement produces thousands of
    data points but only a compact, faithful representation is needed for
    output, plotting, or storage.

    Algorithm overview
    ------------------
    Three independent strategies select "important" indices, which are then
    merged together with the two endpoints:

    1. **Gradient-change detection** (curvature proxy)
       Computes the fractional change in the first derivative between
       consecutive points:  ``pct[i] = (dy'[i+1] - dy'[i]) / dy'[i]``.
       Points where ``|pct| > grad_inc`` are kept.  These mark sharp bends
       in the curve -- e.g., shocks, discontinuities, or phase transitions.

    2. **Sign-change detection** (local extrema)
       Keeps every point where the first derivative changes sign, i.e.,
       every local minimum and maximum of ``y(x)``.

    3. **Cumulative-distance sampling** (uniform arc-length in y)
       The total variation of ``y`` is divided into ``nmin`` equal "distance
       bins".  One point is selected at each bin boundary.  This gives dense
       sampling where ``y`` changes rapidly and sparse sampling where ``y``
       is nearly flat -- adapting automatically to the curve shape.

    For a perfectly flat curve (zero range), the algorithm falls back to
    uniformly spaced indices.

    Parameters
    ----------
    x_arr : array-like
        Independent variable (e.g., position, time, wavelength).
        Must be the same length as ``y_arr``.
    y_arr : array-like
        Dependent variable (e.g., temperature, density, flux).
        Must be the same length as ``x_arr``.
    nmin : int, optional
        Target *minimum* number of output samples.  The actual number of
        returned points may be larger if extra feature points are found by
        the gradient and sign-change detectors.  Clamped to >= 100.
        Default is 100.
    grad_inc : float, optional
        Fractional gradient-change threshold.  A point is flagged as
        "important" when the local gradient changes by more than this
        fraction relative to the previous gradient.  Lower values keep
        more points (more sensitive to curvature); higher values keep fewer.
        Default is 1.0 (i.e., 100 % change).

    Returns
    -------
    x_out : np.ndarray
        Downsampled independent variable.
    y_out : np.ndarray
        Downsampled dependent variable (same length as ``x_out``).

    Raises
    ------
    ValueError
        If ``x_arr`` and ``y_arr`` have different lengths.

    Examples
    --------
    **1. Basic usage with numpy arrays:**

    >>> import numpy as np
    >>> from src._functions.operations import _simplify
    >>> x = np.linspace(0, 10, 5000)
    >>> y = np.sin(x) + 0.5 * np.sin(5 * x)
    >>> x_s, y_s = _simplify(x, y, nmin=100)
    >>> print(f"Reduced {x.size} points -> {x_s.size} points")

    **2. Works with plain Python lists (no numpy needed at call site):**

    >>> x = [0.0, 0.1, 0.2, 0.3, ..., 10.0]
    >>> y = [1.2, 1.5, 1.3, 1.8, ..., 0.9]
    >>> x_s, y_s = _simplify(x, y)

    **3. Tuning sensitivity with ``grad_inc``:**

    Lower ``grad_inc`` = keep more points (more sensitive to bends).
    Higher ``grad_inc`` = keep fewer points (only the sharpest features).

    >>> # Sensitive: keep points where gradient changes by > 50%
    >>> x_s, y_s = _simplify(x, y, nmin=100, grad_inc=0.5)
    >>> # Aggressive: only keep points where gradient changes by > 200%
    >>> x_s, y_s = _simplify(x, y, nmin=100, grad_inc=2.0)

    **4. Reading from a file and simplifying:**

    >>> data = np.loadtxt("profile.csv", delimiter=",")
    >>> x_s, y_s = _simplify(data[:, 0], data[:, 1], nmin=200)
    >>> np.savetxt("profile_simplified.csv", np.column_stack([x_s, y_s]),
    ...            delimiter=",", header="x,y")

    **5. Using in a plotting script (e.g. with matplotlib):**

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 4 * np.pi, 10000)
    >>> y = np.exp(-0.1 * x) * np.sin(x)
    >>> x_s, y_s = _simplify(x, y, nmin=150)
    >>> plt.plot(x, y, 'k-', alpha=0.3, label=f"original ({x.size} pts)")
    >>> plt.plot(x_s, y_s, 'ro-', ms=2, label=f"simplified ({x_s.size} pts)")
    >>> plt.legend()
    >>> plt.show()

    **6. Checking simplification quality:**

    >>> x_s, y_s = _simplify(x, y, nmin=100)
    >>> metrics = _simplify_error(x, y, x_s, y_s)
    >>> print(f"RMSE = {metrics['rms_err']:.2e}")
    >>> print(f"R²   = {metrics['r_squared']:.6f}")
    >>> print(f"Compression = {metrics['compression']:.1f}x")

    **7. Visualising before/after with error panel:**

    >>> _simplify_plot(x, y, x_s, y_s, title="My data", save_path="comparison.png")

    **8. CLI usage (no scripting needed):**

    .. code-block:: bash

        python src/_functions/operations.py profile.csv -o reduced.csv --nmin 200
        python src/_functions/operations.py profile.csv --plot  # visualise
    """
    # --- Input validation ---
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)

    # Nothing to simplify for empty arrays.
    if x.size == 0 or y.size == 0:
        return x, y
    if x.size != y.size:
        raise ValueError(
            f"_simplify(): x and y must have the same length. "
            f"Got {x.size} and {y.size}"
        )
    # If the array is already short enough, return as-is.
    if nmin >= x.size:
        return x, y
    # Enforce a floor of 100 samples so the output is always useful.
    nmin = max(int(nmin), 100)

    # =====================================================================
    # Strategy 1: Gradient-based feature detection
    # =====================================================================
    # Compute the numerical first derivative dy/dx (using central differences).
    grad = np.gradient(y)

    # Build a safe denominator for the fractional change calculation.
    # Where |grad[i]| is essentially zero (flat region), replace with a
    # tiny positive constant `eps` to avoid division by zero.
    eps = 1e-30
    denom = np.where(
        np.abs(grad[:-1]) < eps, eps, grad[:-1]
    )

    # Fractional change in the gradient between consecutive points:
    #   pct[i] = (grad[i+1] - grad[i]) / grad[i]
    # This is a discrete second-derivative normalised by the local slope.
    # Large |pct| means the curve is bending sharply at that point.
    pct = np.diff(grad) / denom

    # Keep indices where the fractional gradient change exceeds the threshold.
    important_grad = np.where(np.abs(pct) > grad_inc)[0]

    # Keep indices where the derivative changes sign (local extrema).
    # np.sign(grad) is -1, 0, or +1; a nonzero diff marks a sign flip.
    important_sign = np.where(np.diff(np.sign(grad)) != 0)[0]

    # =====================================================================
    # Strategy 2: Cumulative-distance sampling in y
    # =====================================================================
    # Total range of y values.
    yrng = float(np.nanmax(y) - np.nanmin(y))

    if not np.isfinite(yrng) or yrng == 0:
        # Special case: perfectly flat curve (or all NaN).
        # Fall back to uniformly spaced indices.
        idx = np.unique(np.linspace(0, x.size - 1, nmin).astype(int))
        return x[idx], y[idx]

    # Maximum allowed cumulative y-distance between kept points.
    # Dividing the total range by nmin gives roughly nmin bins.
    maxdist = yrng / nmin

    # Cumulative absolute change in y along the array.
    y_cum = np.cumsum(np.abs(np.diff(y)))

    # Assign each point to a "distance bin".  When the bin number changes
    # between consecutive points, that boundary is a selected sample.
    bins = (y_cum / maxdist).astype(int)
    idx_dist = np.where(bins[:-1] != bins[1:])[0]

    # =====================================================================
    # Strategy 3: Merge all candidates + endpoints
    # =====================================================================
    # Union all selected indices from the three strategies, plus the first
    # and last points (endpoints are always kept).
    merged = reduce(
        np.union1d,
        [
            np.array([0], dtype=int),              # first point
            important_grad.astype(int),            # sharp bends
            important_sign.astype(int),            # local extrema
            idx_dist.astype(int),                  # arc-length samples
            np.array([x.size - 1], dtype=int),     # last point
        ],
    )

    # Safety: clip to valid index range and deduplicate.
    merged = np.unique(np.clip(merged, 0, x.size - 1))

    return x[merged], y[merged]


def _simplify_error(
    x_orig: Union[np.ndarray, Sequence[float]],
    y_orig: Union[np.ndarray, Sequence[float]],
    x_simp: Union[np.ndarray, Sequence[float]],
    y_simp: Union[np.ndarray, Sequence[float]],
) -> dict:
    """
    Compute error metrics comparing a simplified curve to the original.

    The simplified curve is linearly interpolated back onto the original
    x-grid, and the pointwise residuals are used to compute several
    standard error measures.

    Parameters
    ----------
    x_orig, y_orig : array-like
        Original (full-resolution) curve.
    x_simp, y_simp : array-like
        Simplified (downsampled) curve, as returned by ``_simplify()``.

    Returns
    -------
    metrics : dict
        Dictionary with the following keys:

        - ``"max_abs_err"`` : float
            Maximum absolute error (L-infinity norm).  The worst-case
            pointwise deviation between simplified and original.
        - ``"mean_abs_err"`` : float
            Mean absolute error (MAE).  Average pointwise deviation.
        - ``"rms_err"`` : float
            Root-mean-square error (RMSE).  Penalises large deviations
            more than MAE.
        - ``"max_rel_err"`` : float
            Maximum relative error, ``max(|residual| / |y_orig|)``,
            skipping points where ``|y_orig| < 1e-30``.  Useful when
            the signal spans many orders of magnitude.
        - ``"r_squared"`` : float
            Coefficient of determination (R^2).  1.0 = perfect
            reconstruction; values close to 1.0 indicate the simplified
            curve captures nearly all variance of the original.
        - ``"compression"`` : float
            Compression ratio, ``len(x_orig) / len(x_simp)``.  Higher
            means more aggressive downsampling.
        - ``"n_orig"`` : int
            Number of points in the original curve.
        - ``"n_simp"`` : int
            Number of points in the simplified curve.

    Examples
    --------
    >>> x = np.linspace(0, 10, 5000)
    >>> y = np.sin(x)
    >>> x_s, y_s = _simplify(x, y)
    >>> metrics = _simplify_error(x, y, x_s, y_s)
    >>> print(f"RMSE = {metrics['rms_err']:.2e}, R² = {metrics['r_squared']:.6f}")
    """
    x_o = np.asarray(x_orig, dtype=float)
    y_o = np.asarray(y_orig, dtype=float)
    x_s = np.asarray(x_simp, dtype=float)
    y_s = np.asarray(y_simp, dtype=float)

    # Interpolate the simplified curve back onto the original x-grid.
    y_interp = np.interp(x_o, x_s, y_s)

    # Pointwise residuals.
    residual = y_o - y_interp

    # --- Error metrics ---
    max_abs = float(np.max(np.abs(residual)))
    mean_abs = float(np.mean(np.abs(residual)))
    rms = float(np.sqrt(np.mean(residual ** 2)))

    # Relative error (skip near-zero original values to avoid division blow-up).
    eps = 1e-30
    mask = np.abs(y_o) > eps
    if np.any(mask):
        max_rel = float(np.max(np.abs(residual[mask]) / np.abs(y_o[mask])))
    else:
        max_rel = 0.0

    # R² (coefficient of determination).
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_o - np.mean(y_o)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "rms_err": rms,
        "max_rel_err": max_rel,
        "r_squared": r_squared,
        "compression": float(x_o.size) / float(x_s.size) if x_s.size > 0 else float("inf"),
        "n_orig": int(x_o.size),
        "n_simp": int(x_s.size),
    }


def _simplify_plot(
    x_orig: Union[np.ndarray, Sequence[float]],
    y_orig: Union[np.ndarray, Sequence[float]],
    x_simp: Union[np.ndarray, Sequence[float]],
    y_simp: Union[np.ndarray, Sequence[float]],
    title: str = "Curve simplification",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Visualise original vs. simplified curves with an error panel.

    Produces a two-panel figure:

    - **Top panel**: original curve (grey line) overlaid with simplified
      points (red dots + line).
    - **Bottom panel**: pointwise residual (original minus linearly
      interpolated simplified curve), with max/mean/RMS error annotated.

    Parameters
    ----------
    x_orig, y_orig : array-like
        Original (full-resolution) curve.
    x_simp, y_simp : array-like
        Simplified (downsampled) curve.
    title : str, optional
        Figure title.  Default: ``"Curve simplification"``.
    save_path : str or None, optional
        If given, save the figure to this file path (e.g. ``"plot.png"``).
        The format is inferred from the extension.  Default: ``None`` (no save).
    show : bool, optional
        Whether to call ``plt.show()``.  Set to ``False`` in non-interactive
        environments or when saving only.  Default: ``True``.

    Examples
    --------
    >>> x = np.linspace(0, 10, 5000)
    >>> y = np.sin(x) + 0.5 * np.sin(5 * x)
    >>> x_s, y_s = _simplify(x, y, nmin=100)
    >>> _simplify_plot(x, y, x_s, y_s, save_path="comparison.png")
    """
    import matplotlib.pyplot as plt

    x_o = np.asarray(x_orig, dtype=float)
    y_o = np.asarray(y_orig, dtype=float)
    x_s = np.asarray(x_simp, dtype=float)
    y_s = np.asarray(y_simp, dtype=float)

    # Interpolate simplified curve onto original grid for residual.
    y_interp = np.interp(x_o, x_s, y_s)
    residual = y_o - y_interp

    # Compute metrics for annotation.
    metrics = _simplify_error(x_o, y_o, x_s, y_s)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # --- Top panel: curves ---
    ax1.plot(x_o, y_o, "-", color="0.6", lw=1.0,
             label=f"original ({metrics['n_orig']} pts)")
    ax1.plot(x_s, y_s, "o-", color="tab:red", ms=2.5, lw=0.8,
             label=f"simplified ({metrics['n_simp']} pts)")
    ax1.set_ylabel("y")
    ax1.set_title(title)
    ax1.legend(loc="best", fontsize=9)

    # --- Bottom panel: residual ---
    ax2.fill_between(x_o, residual, 0, color="tab:blue", alpha=0.3)
    ax2.plot(x_o, residual, "-", color="tab:blue", lw=0.5)
    ax2.axhline(0, color="k", lw=0.5, ls="--")
    ax2.set_ylabel("residual")
    ax2.set_xlabel("x")

    # Annotate with key metrics.
    info = (
        f"RMSE = {metrics['rms_err']:.2e}    "
        f"MAE = {metrics['mean_abs_err']:.2e}    "
        f"max|err| = {metrics['max_abs_err']:.2e}    "
        f"R\u00b2 = {metrics['r_squared']:.6f}    "
        f"compression = {metrics['compression']:.1f}x"
    )
    ax2.text(
        0.5, -0.35, info,
        transform=ax2.transAxes, ha="center", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="0.8"),
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to '{save_path}'.")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _simplify_animate(
    x_orig: Union[np.ndarray, Sequence[float]],
    y_orig: Union[np.ndarray, Sequence[float]],
    x_simp: Union[np.ndarray, Sequence[float]],
    y_simp: Union[np.ndarray, Sequence[float]],
    save_path: str = "simplify.gif",
    fps: int = 30,
    duration: float = 3.0,
    title: str = "Curve simplification",
) -> None:
    """
    Create an animated GIF showing the original curve morphing into the
    simplified representative points.

    The animation has three phases:

    1. **Fade** (0 – 40 %): the original dense curve fades from solid to
       transparent while the simplified points fade in.
    2. **Collapse** (40 – 70 %): intermediate original points slide towards
       their nearest simplified neighbour, visually "collapsing" the dense
       curve onto the kept points.
    3. **Hold** (70 – 100 %): the final simplified curve is shown with
       error metrics annotated, so the viewer can inspect the result.

    Parameters
    ----------
    x_orig, y_orig : array-like
        Original (full-resolution) curve.
    x_simp, y_simp : array-like
        Simplified (downsampled) curve.
    save_path : str, optional
        Output file path.  Default: ``"simplify.gif"``.
        Also supports ``.mp4`` if ``ffmpeg`` is installed.
    fps : int, optional
        Frames per second.  Default: 30.
    duration : float, optional
        Total animation duration in seconds.  Default: 3.0.
    title : str, optional
        Figure title.  Default: ``"Curve simplification"``.

    Examples
    --------
    >>> x = np.linspace(0, 10, 5000)
    >>> y = np.sin(x) + 0.5 * np.sin(5 * x)
    >>> x_s, y_s = _simplify(x, y, nmin=100)
    >>> _simplify_animate(x, y, x_s, y_s, "demo.gif", duration=4.0)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    x_o = np.asarray(x_orig, dtype=float)
    y_o = np.asarray(y_orig, dtype=float)
    x_s = np.asarray(x_simp, dtype=float)
    y_s = np.asarray(y_simp, dtype=float)

    n_frames = int(fps * duration)
    metrics = _simplify_error(x_o, y_o, x_s, y_s)

    # For the collapse phase, precompute where each original point
    # should move to (its nearest simplified point).
    nearest_idx = np.searchsorted(x_s, x_o).clip(0, x_s.size - 1)
    # Refine: check if the previous simplified point is actually closer.
    prev = (nearest_idx - 1).clip(0, x_s.size - 1)
    use_prev = np.abs(x_o - x_s[prev]) < np.abs(x_o - x_s[nearest_idx])
    nearest_idx[use_prev] = prev[use_prev]
    x_target = x_s[nearest_idx]
    y_target = y_s[nearest_idx]

    # Precompute the final residual for the error subplot.
    y_interp = np.interp(x_o, x_s, y_s)
    residual = y_o - y_interp
    res_max = max(np.max(np.abs(residual)), 1e-30)

    # --- Set up figure with two subplots ---
    fig, (ax, ax_err) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    margin = 0.05 * (np.nanmax(y_o) - np.nanmin(y_o) + 1e-30)
    ax.set_xlim(x_o[0], x_o[-1])
    ax.set_ylim(np.nanmin(y_o) - margin, np.nanmax(y_o) + margin)
    ax.set_ylabel("y")
    ax.set_title(title)

    ax_err.set_xlim(x_o[0], x_o[-1])
    ax_err.set_ylim(-res_max * 1.3, res_max * 1.3)
    ax_err.set_ylabel("residual")
    ax_err.set_xlabel("x")
    ax_err.axhline(0, color="k", lw=0.5, ls="--")

    # Main plot elements.
    line_orig, = ax.plot([], [], "-", color="0.4", lw=1.2)
    scatter_moving = ax.scatter([], [], s=4, color="tab:blue", zorder=3)
    line_simp, = ax.plot([], [], "o-", color="tab:red", ms=4, lw=1.0, zorder=4)
    info_text = ax.text(
        0.5, 0.02, "", transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="0.8"),
    )

    # Error subplot elements.
    err_fill = ax_err.fill_between(x_o, 0, 0, color="tab:blue", alpha=0.0)
    err_line, = ax_err.plot([], [], "-", color="tab:blue", lw=0.7)
    err_text = ax_err.text(
        0.98, 0.95, "", transform=ax_err.transAxes, ha="right", va="top",
        fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.8),
    )

    def _update(frame):
        nonlocal err_fill
        t = frame / max(n_frames - 1, 1)  # normalised time 0..1

        if t < 0.4:
            # Phase 1: Fade — original fades out, simplified points fade in.
            p = t / 0.4  # 0..1 within this phase
            line_orig.set_data(x_o, y_o)
            line_orig.set_alpha(1.0 - 0.7 * p)
            # Show all original points as scatter, fading to blue.
            scatter_moving.set_offsets(np.column_stack([x_o, y_o]))
            scatter_moving.set_alpha(p * 0.6)
            line_simp.set_data([], [])
            info_text.set_text(
                f"{metrics['n_orig']} points"
            )
            # Error panel: still blank during fade.
            err_line.set_data([], [])
            err_fill.remove()
            err_fill = ax_err.fill_between(x_o, 0, 0, color="tab:blue", alpha=0.0)
            err_text.set_text("")

        elif t < 0.7:
            # Phase 2: Collapse — points slide to nearest simplified location.
            p = (t - 0.4) / 0.3  # 0..1 within this phase
            # Ease-in-out (smooth step).
            p = p * p * (3 - 2 * p)
            x_curr = x_o + p * (x_target - x_o)
            y_curr = y_o + p * (y_target - y_o)
            line_orig.set_data(x_o, y_o)
            line_orig.set_alpha(0.3 * (1 - p))
            scatter_moving.set_offsets(np.column_stack([x_curr, y_curr]))
            scatter_moving.set_alpha(0.6)
            line_simp.set_data([], [])
            info_text.set_text(
                f"{metrics['n_orig']} \u2192 {metrics['n_simp']} points"
            )
            # Error panel: residual fades in as collapse progresses.
            current_res = residual * p
            err_line.set_data(x_o, current_res)
            err_line.set_alpha(p)
            err_fill.remove()
            err_fill = ax_err.fill_between(
                x_o, current_res, 0, color="tab:blue", alpha=0.25 * p,
            )
            err_text.set_text("")

        else:
            # Phase 3: Hold — show final simplified curve with metrics.
            line_orig.set_data(x_o, y_o)
            line_orig.set_alpha(0.2)
            scatter_moving.set_offsets(np.empty((0, 2)))
            line_simp.set_data(x_s, y_s)
            line_simp.set_alpha(1.0)
            info_text.set_text(
                f"{metrics['n_simp']} pts  |  "
                f"RMSE={metrics['rms_err']:.2e}  |  "
                f"R\u00b2={metrics['r_squared']:.4f}  |  "
                f"{metrics['compression']:.1f}x compression"
            )
            # Error panel: full residual with annotation.
            err_line.set_data(x_o, residual)
            err_line.set_alpha(1.0)
            err_fill.remove()
            err_fill = ax_err.fill_between(
                x_o, residual, 0, color="tab:blue", alpha=0.3,
            )
            err_text.set_text(
                f"max|err|={metrics['max_abs_err']:.2e}  "
                f"RMSE={metrics['rms_err']:.2e}"
            )

        return line_orig, scatter_moving, line_simp, info_text, err_line, err_fill, err_text

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=True)

    # Save — use pillow for GIF, ffmpeg for mp4.
    if save_path.lower().endswith(".mp4"):
        writer = "ffmpeg"
    else:
        writer = "pillow"
    anim.save(save_path, writer=writer, fps=fps, dpi=120)
    plt.close(fig)
    print(f"Animation saved to '{save_path}' ({n_frames} frames, {duration:.1f}s).")


def find_nearest(array, value):
    """
    finds index idx in array for which array[idx] is closest to value
    """
    # make sure that we deal with an numpy array
    array = np.array(array)
    # index
    idx = (np.abs(array-value)).argmin()
    # return
    return idx

def find_nearest_lower(array, value):
    """
    This fucntion finds idx in array for which array[idx] satisfies:
        1) smaller or equal to value; and
        2) closest to value.
    Elements in array need be monotonically increasing or decreasing!
    """
    # check whether array is monotonic 
    # debug
    if any(array < 0):
        print(array)
        
    if not monotonic(array):
        print(f"array has to be monotonic! Instead got {array}.")
        # np.save(warpfield_params.out_dir + 'T_array_monotonic_check.npy', array)
        raise MonotonicError()
    
    # is it increasing?
    mon_incr = kindof_increasing(array)
    
    # get index
    idx = find_nearest(array, value)
    #---
    #---
    if array[idx] - value > 0: # then this element is the closest, but it is larger than value
        if mon_incr: 
            idx += -1 # take the element before, it will be smaller than value (if array is monotonically increasing)
        else: 
            idx += 1
    # Notes: boundary conditions, just in case. Although when these happen, it means that
    # the returned idx is actually higher than the value instead of the desired 
    # lower. Not quite sure what to do with that for now, but this part of 
    # the code shouldnt need to run anyway.
    if idx >= len(array): 
        idx = len(array) - 1
    if idx < 0: 
        idx = 0
    # return
    return idx

#  kind of, because includes equal values like [1,2,3,3,4]
def kindof_increasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def kindof_decreasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return kindof_increasing(L) or kindof_decreasing(L)


def find_nearest_higher(array, value):
    """
    This fucntion finds idx in array for which array[idx] satisfies:
        1) higher or equal to value; and
        2) closest to value.
    Elements in array need be monotonically increasing or decreasing!
    """
    # check whether array is monotonic 
    # debug
    if any(array < 0):
        print(array)
        
    if not monotonic(array):
        print(f"array has to be monotonic! Instead got {array}.")
        # np.save(warpfield_params.out_dir + 'T_array_monotonic_check.npy', array)
        raise MonotonicError()

    # is it increasing?
    mon_incr = kindof_increasing(array)
    
    
    # get index
    idx = find_nearest(array, value)
    #---
    #---
    if array[idx] - value < 0: # then this element is the closest, but it is larger than value
        if mon_incr: 
            idx += 1 # take the element before, it will be smaller than value (if array is monotonically increasing)
        else: 
            idx += -1
    # Notes: boundary conditions, just in case. Although when these happen, it means that
    # the returned idx is actually higher than the value instead of the desired 
    # lower. Not quite sure what to do with that for now, but this part of 
    # the code shouldnt need to run anyway.
    if idx >= len(array): 
        idx = len(array) - 1
    if idx < 0: 
        idx = 0
    # return
    return idx

class MonotonicError(Exception):
    pass

def get_soundspeed(T, params):
    """
    This function computes the isothermal soundspeed, c_s, given temperature
    T and mean molecular weight mu.

    Parameters
    ----------
    T : float (Units: K)
        Temperature of the gas.

    Returns
    -------
    The isothermal soundspeed c_s (Units: Myr/pc)

    """    
    if T > 1e4:
        mu = params['mu_ion'] * cvt.Msun2g
    else:
        mu = params['mu_atom'] * cvt.Msun2g
    
    return  np.sqrt(params['gamma_adia'] * (params['k_B'] * cvt.k_B_au2cgs) * T / mu) * cvt.v_cms2au


# =============================================================================
# CLI entry point for _simplify
# =============================================================================
def _simplify_cli():
    """
    Command-line interface for the _simplify curve downsampling function.

    Reads x and y data from a two-column text file (whitespace- or
    comma-separated), runs the simplification algorithm, and writes the
    reduced data to an output file.

    Usage
    -----
    python operations.py input.csv -o output.csv --nmin 150 --grad-inc 0.5

    Positional arguments
    --------------------
    infile : str
        Path to input data file.  Must contain two columns (x, y).
        Lines starting with '#' are treated as comments and skipped.
        Both whitespace- and comma-delimited formats are accepted.

    Optional arguments
    ------------------
    -o, --output : str
        Path to the output file.  Default: ``simplified_output.csv``.
    --nmin : int
        Minimum number of output samples (default: 100).
    --grad-inc : float
        Fractional gradient-change threshold (default: 1.0).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="_simplify",
        description=(
            "Downsample a two-column (x, y) data file while preserving "
            "sharp features, local extrema, and overall curve shape."
        ),
        epilog=(
            "Example:\n"
            "  python operations.py data.csv -o reduced.csv --nmin 200\n\n"
            "The algorithm combines three strategies:\n"
            "  1. Gradient-change detection  -- keeps sharp bends\n"
            "  2. Sign-change detection      -- keeps local extrema\n"
            "  3. Cumulative-distance sampling -- uniform arc-length in y\n"
            "All selected indices are merged with the endpoints to produce\n"
            "a compact, faithful representation of the original curve."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "infile",
        help="Path to input data file (two columns: x y).",
    )
    parser.add_argument(
        "-o", "--output",
        default="simplified_output.csv",
        help="Path to output file (default: simplified_output.csv).",
    )
    parser.add_argument(
        "--nmin",
        type=int,
        default=100,
        help="Minimum number of output samples (default: 100).",
    )
    parser.add_argument(
        "--grad-inc",
        type=float,
        default=1.0,
        help="Fractional gradient-change threshold (default: 1.0).",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Print error metrics (RMSE, MAE, R², etc.) after simplification.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a before/after comparison plot with residuals.",
    )
    parser.add_argument(
        "--plot-save",
        default=None,
        metavar="PATH",
        help="Save the comparison plot to a file (e.g. comparison.png).",
    )
    parser.add_argument(
        "--animate",
        default=None,
        metavar="PATH",
        help=(
            "Save an animated GIF/MP4 showing the original curve morphing "
            "into the simplified points (e.g. simplify.gif)."
        ),
    )
    parser.add_argument(
        "--animate-duration",
        type=float,
        default=3.0,
        help="Animation duration in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--animate-fps",
        type=int,
        default=30,
        help="Animation frames per second (default: 30).",
    )

    args = parser.parse_args()

    # --- Read input ---
    # Try comma-delimited first, fall back to whitespace.
    try:
        data = np.loadtxt(args.infile, delimiter=",", comments="#")
    except ValueError:
        data = np.loadtxt(args.infile, comments="#")

    if data.ndim != 2 or data.shape[1] < 2:
        raise SystemExit(
            f"Error: expected at least 2 columns in '{args.infile}', "
            f"got shape {data.shape}."
        )

    x, y = data[:, 0], data[:, 1]

    # --- Simplify ---
    x_out, y_out = _simplify(x, y, nmin=args.nmin, grad_inc=args.grad_inc)

    # --- Write output ---
    np.savetxt(
        args.output,
        np.column_stack([x_out, y_out]),
        delimiter=",",
        header="x,y",
        comments="# ",
    )
    print(
        f"Simplified {x.size} points -> {x_out.size} points.  "
        f"Written to '{args.output}'."
    )

    # --- Optional: print error metrics ---
    if args.metrics or args.plot or args.plot_save:
        metrics = _simplify_error(x, y, x_out, y_out)

    if args.metrics:
        print()
        print("  Error metrics")
        print("  " + "-" * 40)
        print(f"  Max absolute error : {metrics['max_abs_err']:.4e}")
        print(f"  Mean absolute error: {metrics['mean_abs_err']:.4e}")
        print(f"  RMS error          : {metrics['rms_err']:.4e}")
        print(f"  Max relative error : {metrics['max_rel_err']:.4e}")
        print(f"  R-squared          : {metrics['r_squared']:.6f}")
        print(f"  Compression ratio  : {metrics['compression']:.1f}x")

    # --- Optional: plot ---
    if args.plot or args.plot_save:
        _simplify_plot(
            x, y, x_out, y_out,
            title=f"Simplification of '{args.infile}'",
            save_path=args.plot_save,
            show=args.plot,
        )

    # --- Optional: animation ---
    if args.animate:
        _simplify_animate(
            x, y, x_out, y_out,
            save_path=args.animate,
            fps=args.animate_fps,
            duration=args.animate_duration,
            title=f"Simplification of '{args.infile}'",
        )


if __name__ == "__main__":
    import sys

    # If command-line arguments are given, run the file-based CLI.
    # Otherwise, run a quick interactive demo so users can see how it works.
    if len(sys.argv) > 1:
        _simplify_cli()
    else:
        # ---- Quick demo ----
        print("=" * 60)
        print("  _simplify() — interactive demo")
        print("=" * 60)
        print()

        # Generate a test signal: decaying sine with a sharp spike.
        x = np.linspace(0, 4 * np.pi, 5000)
        y = np.exp(-0.1 * x) * np.sin(x)
        # Add a sharp spike at the midpoint to test feature detection.
        y[2500] += 3.0

        x_s, y_s = _simplify(x, y, nmin=100)

        # Show error metrics.
        metrics = _simplify_error(x, y, x_s, y_s)

        print(f"  Original  : {metrics['n_orig']} points")
        print(f"  Simplified: {metrics['n_simp']} points")
        print(f"  Compression: {metrics['compression']:.1f}x")
        print()
        print("  Error metrics:")
        print(f"    RMSE             = {metrics['rms_err']:.4e}")
        print(f"    MAE              = {metrics['mean_abs_err']:.4e}")
        print(f"    Max |error|      = {metrics['max_abs_err']:.4e}")
        print(f"    Max relative err = {metrics['max_rel_err']:.4e}")
        print(f"    R-squared        = {metrics['r_squared']:.6f}")
        print()
        print("  Endpoints preserved:")
        print(f"    first = ({x_s[0]:.4f}, {y_s[0]:.4f})")
        print(f"    last  = ({x_s[-1]:.4f}, {y_s[-1]:.4f})")
        print()
        print("  To use in your own script:")
        print()
        print("    from src._functions.operations import _simplify")
        print("    from src._functions.operations import _simplify_error")
        print("    from src._functions.operations import _simplify_plot")
        print("    from src._functions.operations import _simplify_animate")
        print()
        print("    x_s, y_s = _simplify(x, y, nmin=100)")
        print("    metrics  = _simplify_error(x, y, x_s, y_s)")
        print("    _simplify_plot(x, y, x_s, y_s)")
        print("    _simplify_animate(x, y, x_s, y_s, 'simplify.gif')")
        print()
        print("  Or from the command line:")
        print()
        print("    python src/_functions/operations.py data.csv -o out.csv --metrics")
        print("    python src/_functions/operations.py data.csv --plot")
        print("    python src/_functions/operations.py data.csv --animate simplify.gif")
        print()
        print("  Run with --help for all options.")
        print("=" * 60)

