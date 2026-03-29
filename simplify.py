#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone curve-simplification module.

Heuristic downsampling of 1-D curves while preserving physically and
visually important features (sharp bends, local extrema, arc-length
uniformity).  No dependencies beyond numpy / matplotlib / stdlib.

Functions
---------
_simplify          Core downsampling algorithm.
_simplify_error    Error metrics (RMSE, MAE, R², compression, …).
_simplify_plot     Static before/after comparison plot.
_simplify_animate  Animated GIF/MP4 of the simplification process.
_random_test_curve Generate a random curve that exercises all strategies.
_simplify_cli      Command-line interface (reads two-column text files).
"""

from functools import reduce
from typing import Tuple, Union, Sequence

import numpy as np


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
    >>> from simplify import _simplify
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

        python simplify.py profile.csv -o reduced.csv --nmin 200
        python simplify.py profile.csv --metrics          # print error table
        python simplify.py profile.csv --plot             # interactive plot
        python simplify.py profile.csv --animate out.gif  # animated GIF
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
    save_path: Union[str, None] = None,
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

    # Compute metrics and residual (interpolation done once, inside _simplify_error).
    metrics = _simplify_error(x_o, y_o, x_s, y_s)
    residual = y_o - np.interp(x_o, x_s, y_s)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
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

    # Annotate with key metrics inside the residual panel.
    info = (
        f"RMSE = {metrics['rms_err']:.2e}    "
        f"MAE = {metrics['mean_abs_err']:.2e}    "
        f"max|err| = {metrics['max_abs_err']:.2e}    "
        f"R\u00b2 = {metrics['r_squared']:.6f}    "
        f"compression = {metrics['compression']:.1f}x"
    )
    ax2.text(
        0.5, 0.02, info,
        transform=ax2.transAxes, ha="center", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="0.8", alpha=0.9),
    )

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
    save_path: str = "simplify.gif",
    fps: int = 30,
    duration: float = 6.0,
    title: str = "Curve simplification",
    n_steps: int = 30,
) -> None:
    """
    Create an animated GIF showing progressive curve simplification.

    The animation sweeps through decreasing numbers of retained points,
    from the full original down past the optimal simplification and into
    aggressive over-simplification.  Two panels are shown:

    - **Top panel**: the underlying curve as a thin line, with the current
      simplified points overlaid as small dots.
    - **Bottom panel**: RMSE vs. number of retained points, building up
      as the animation progresses (log-log scale).

    Parameters
    ----------
    x_orig, y_orig : array-like
        Original (full-resolution) curve.
    save_path : str, optional
        Output file path.  Default: ``"simplify.gif"``.
        Also supports ``.mp4`` if ``ffmpeg`` is installed.
    fps : int, optional
        Frames per second.  Default: 30.
    duration : float, optional
        Total animation duration in seconds.  Default: 6.0.
    title : str, optional
        Figure title.  Default: ``"Curve simplification"``.
    n_steps : int, optional
        Number of distinct simplification levels to sweep through.
        Default: 30.

    Examples
    --------
    >>> x = np.linspace(0, 10, 5000)
    >>> y = np.sin(x) + 0.5 * np.sin(5 * x)
    >>> _simplify_animate(x, y, "demo.gif", duration=6.0)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # --- LaTeX-style serif fonts ---
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    })

    x_o = np.asarray(x_orig, dtype=float)
    y_o = np.asarray(y_orig, dtype=float)
    n_orig = x_o.size

    # --- Precompute simplification at many nmin values ---
    # Sweep from ~n_orig down to a very aggressive level (5 points).
    # Use log-spaced nmin values so we see smooth progression at both ends.
    nmin_values = np.unique(
        np.logspace(np.log10(max(n_orig, 101)), np.log10(5), n_steps)
        .astype(int)
    )
    # Always include endpoints.
    nmin_values = np.unique(np.concatenate([[n_orig], nmin_values, [5]]))
    nmin_values = np.sort(nmin_values)[::-1]  # descending

    # Run _simplify for each nmin and store results + error.
    steps = []
    for nmin in nmin_values:
        x_s, y_s = _simplify(x_o, y_o, nmin=int(nmin))
        m = _simplify_error(x_o, y_o, x_s, y_s)
        steps.append({
            "npts": m["n_simp"],
            "x": x_s,
            "y": y_s,
            "rms": m["rms_err"],
            "r2": m["r_squared"],
        })

    # Deduplicate steps with same npts (can happen when _simplify returns
    # more points than nmin).
    seen = set()
    unique_steps = []
    for s in steps:
        if s["npts"] not in seen:
            seen.add(s["npts"])
            unique_steps.append(s)
    steps = unique_steps

    # Collect arrays for the error subplot.
    all_npts = np.array([s["npts"] for s in steps])
    all_rms = np.array([s["rms"] for s in steps])

    # --- Animation timing ---
    # Each step gets equal time, plus a short hold on the last frame.
    n_anim_steps = len(steps)
    hold_frac = 0.15  # fraction of duration to hold the final frame
    sweep_duration = duration * (1 - hold_frac)
    hold_duration = duration * hold_frac
    n_frames = int(fps * duration)
    sweep_frames = int(fps * sweep_duration)

    # --- Set up figure ---
    fig, (ax, ax_err) = plt.subplots(
        2, 1, figsize=(10, 6.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.30},
    )
    margin = 0.05 * (np.nanmax(y_o) - np.nanmin(y_o) + 1e-30)
    ax.set_xlim(x_o[0], x_o[-1])
    ax.set_ylim(np.nanmin(y_o) - margin, np.nanmax(y_o) + margin)
    ax.set_ylabel(r"$y$", fontsize=12)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Thin underlying curve (always visible).
    ax.plot(x_o, y_o, "-", color="0.75", lw=0.8, zorder=1)

    # Simplified dots + connecting line (updated each frame).
    line_simp, = ax.plot([], [], "-", color="tab:red", lw=0.6, zorder=2)
    scatter_simp = ax.scatter([], [], s=6, color="tab:red", zorder=3,
                              edgecolors="none")
    info_text = ax.text(
        0.98, 0.96, "", transform=ax.transAxes, ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    # --- Error subplot: RMSE vs n_points ---
    rms_nonzero = all_rms[all_rms > 0]
    if rms_nonzero.size > 0:
        rms_lo = rms_nonzero.min() * 0.3
        rms_hi = all_rms.max() * 3.0
    else:
        rms_lo, rms_hi = 1e-10, 1.0
    ax_err.set_xlim(max(all_npts.min() * 0.7, 3), all_npts.max() * 1.5)
    ax_err.set_ylim(rms_lo, rms_hi)
    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.set_xlabel(r"Number of points $n$", fontsize=12)
    ax_err.set_ylabel(r"RMSE", fontsize=12)
    ax_err.invert_xaxis()  # more points on left, fewer on right

    err_line, = ax_err.plot([], [], "o-", color="tab:blue", ms=3, lw=1.0)
    err_marker = ax_err.scatter([], [], s=40, color="tab:red", zorder=5,
                                edgecolors="black", linewidths=0.5)

    def _update(frame):
        if frame < sweep_frames:
            # Map frame to step index.
            step_idx = int(frame / max(sweep_frames - 1, 1) * (n_anim_steps - 1))
            step_idx = min(step_idx, n_anim_steps - 1)
        else:
            # Hold phase — stay on last step.
            step_idx = n_anim_steps - 1

        s = steps[step_idx]

        # --- Top panel: update simplified points ---
        line_simp.set_data(s["x"], s["y"])
        scatter_simp.set_offsets(np.column_stack([s["x"], s["y"]]))
        info_text.set_text(
            f"$n = {s['npts']}$    "
            f"$R^2 = {s['r2']:.6f}$"
        )

        # --- Bottom panel: build up error curve ---
        # Show all steps up to current.
        vis_npts = all_npts[:step_idx + 1]
        vis_rms = all_rms[:step_idx + 1]
        err_line.set_data(vis_npts, vis_rms)
        # Highlight current point.
        err_marker.set_offsets([[s["npts"], s["rms"]]])

        return line_simp, scatter_simp, info_text, err_line, err_marker

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)

    # Save — use pillow for GIF, ffmpeg for mp4.
    if save_path.lower().endswith(".mp4"):
        writer = "ffmpeg"
    else:
        writer = "pillow"
    anim.save(save_path, writer=writer, fps=fps, dpi=120)
    plt.close(fig)

    # Restore default rcParams.
    plt.rcParams.update(plt.rcParamsDefault)

    print(f"Animation saved to '{save_path}' ({n_frames} frames, {duration:.1f}s).")


# =============================================================================
# Random test-curve generator
# =============================================================================
def _random_test_curve(
    npts: int = 10_000,
    seed: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random 1-D curve that exercises every simplification strategy.

    The curve is built from several additive components, each designed to
    trigger a different branch of ``_simplify``:

    - **Smooth base** – sum of 3–6 sinusoids with random frequencies,
      amplitudes and phases.  Produces gentle curvature and many local
      extrema (tests sign-change detection).
    - **Flat plateaus** – 1–3 constant-value segments spliced into the
      curve.  These are "redundant" regions where cumulative-distance
      sampling should thin aggressively.
    - **Sharp spikes** – 2–5 narrow Gaussian pulses of random height.
      Their steep flanks trigger gradient-change detection.
    - **Step discontinuities** – 1–3 Heaviside-like jumps (smoothed over
      a handful of points) that create abrupt level shifts.
    - **Gaussian noise** – low-amplitude noise added everywhere, so the
      algorithm must distinguish real features from jitter.

    Parameters
    ----------
    npts : int, optional
        Number of points in the output curve.  Default: 10 000.
    seed : int or None, optional
        Random seed for reproducibility.  ``None`` gives a different curve
        each time.

    Returns
    -------
    x : np.ndarray, shape (npts,)
        Independent variable on [0, 1].
    y : np.ndarray, shape (npts,)
        Dependent variable (the composite random curve).

    Examples
    --------
    >>> x, y = _random_test_curve(npts=8000, seed=42)
    >>> x_s, y_s = _simplify(x, y, nmin=120)
    >>> _simplify_plot(x, y, x_s, y_s, title="random test curve")
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, npts)
    y = np.zeros_like(x)

    # ------------------------------------------------------------------
    # 1. Smooth sinusoidal base (sign-change + gentle curvature)
    # ------------------------------------------------------------------
    n_sines = rng.integers(2, 5)                       # 2–4 terms
    for _ in range(n_sines):
        freq  = rng.uniform(1.0, 12.0)                # cycles across [0,1]
        amp   = rng.uniform(0.3, 2.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        y += amp * np.sin(2.0 * np.pi * freq * x + phase)

    # ------------------------------------------------------------------
    # 2. Flat plateaus (redundant regions)
    #    Use a smooth blend so the entry/exit don't create artificial
    #    discontinuities that swamp the gradient detector.
    # ------------------------------------------------------------------
    n_plateaus = rng.integers(1, 4)                    # 1–3 plateaus
    for _ in range(n_plateaus):
        width = rng.uniform(0.05, 0.15)               # 5–15 % of domain
        centre = rng.uniform(width, 1.0 - width)
        level = rng.uniform(np.min(y), np.max(y))
        # Smooth top-hat: product of two logistics (rise then fall).
        blend_k = npts / 15                            # transition ~15 pts
        rise = 1.0 / (1.0 + np.exp(np.clip(-blend_k * (x - (centre - width / 2)), -500, 500)))
        fall = 1.0 / (1.0 + np.exp(np.clip( blend_k * (x - (centre + width / 2)), -500, 500)))
        window = rise * fall
        y = y * (1.0 - window) + level * window

    # ------------------------------------------------------------------
    # 3. Sharp spikes (gradient-change detection)
    # ------------------------------------------------------------------
    n_spikes = rng.integers(2, 6)                      # 2–5 spikes
    for _ in range(n_spikes):
        loc   = rng.uniform(0.05, 0.95)
        sigma = rng.uniform(0.002, 0.008)              # narrow but resolved
        amp   = rng.uniform(2.0, 6.0) * rng.choice([-1, 1])
        y += amp * np.exp(-0.5 * ((x - loc) / sigma) ** 2)

    # ------------------------------------------------------------------
    # 4. Step discontinuities (sharp level shifts)
    #    Steepness is set so the transition spans ~20 grid points —
    #    sharp enough to trigger gradient detection but not so steep
    #    that hundreds of transition points are all flagged.
    # ------------------------------------------------------------------
    n_steps = rng.integers(1, 4)                       # 1–3 steps
    for _ in range(n_steps):
        loc    = rng.uniform(0.10, 0.90)
        height = rng.uniform(1.0, 4.0) * rng.choice([-1, 1])
        transition_pts = 20                            # grid points across step
        steepness = npts / max(transition_pts, 1)
        arg = np.clip(-steepness * (x - loc), -500.0, 500.0)
        y += height / (1.0 + np.exp(arg))

    # ------------------------------------------------------------------
    # 5. Tiny Gaussian noise — just enough to be realistic, not enough
    #    to trigger gradient-change detection on smooth stretches.
    # ------------------------------------------------------------------
    noise_amp = 0.001 * (np.max(y) - np.min(y) + 1e-30)
    y += rng.normal(0.0, noise_amp, size=npts)

    return x, y


# =============================================================================
# CLI entry point
# =============================================================================
def _simplify_cli():
    """
    Command-line interface for the _simplify curve downsampling function.

    Reads x and y data from a two-column text file (whitespace- or
    comma-separated), runs the simplification algorithm, and writes the
    reduced data to an output file.

    Usage
    -----
    python simplify.py input.csv -o output.csv --nmin 150 --grad-inc 0.5
    python simplify.py input.csv --metrics --plot
    python simplify.py input.csv --animate simplify.gif --animate-duration 5

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
    --metrics : flag
        Print error metrics (RMSE, MAE, R², etc.) after simplification.
    --plot : flag
        Show an interactive before/after comparison plot with residuals.
    --plot-save : str
        Save the comparison plot to a file (e.g. ``comparison.png``).
    --animate : str
        Save an animated GIF/MP4 showing the original curve morphing
        into the simplified points (e.g. ``simplify.gif``).
    --animate-duration : float
        Animation duration in seconds (default: 3.0).
    --animate-fps : int
        Animation frames per second (default: 30).
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
            "  python simplify.py data.csv -o reduced.csv --nmin 200\n\n"
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
        nargs="?",
        default=None,
        help="Path to input data file (two columns: x y).  "
             "Not required when --random is used.",
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
    parser.add_argument(
        "--random",
        action="store_true",
        help=(
            "Generate a random test curve instead of reading a file.  "
            "The curve contains sinusoidal bases, flat plateaus, sharp "
            "spikes, step discontinuities, and noise — designed to "
            "exercise every simplification strategy."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random (default: None = non-reproducible).",
    )
    parser.add_argument(
        "--random-npts",
        type=int,
        default=10_000,
        help="Number of points in the random curve (default: 10000).",
    )

    args = parser.parse_args()

    # --- Obtain input data ---
    if args.random:
        x, y = _random_test_curve(npts=args.random_npts, seed=args.seed)
        seed_str = f", seed={args.seed}" if args.seed is not None else ""
        source_label = f"random curve ({args.random_npts} pts{seed_str})"
        print(f"Generated {source_label}.")
    elif args.infile is not None:
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
        source_label = f"'{args.infile}'"
    else:
        parser.error("either provide an input file or use --random.")

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
            title=f"Simplification of {source_label}",
            save_path=args.plot_save,
            show=args.plot,
        )

    # --- Optional: animation ---
    if args.animate:
        _simplify_animate(
            x, y,
            save_path=args.animate,
            fps=args.animate_fps,
            duration=args.animate_duration,
            title=f"Simplification of {source_label}",
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
        print("    from simplify import _simplify")
        print("    from simplify import _simplify_error")
        print("    from simplify import _simplify_plot")
        print("    from simplify import _simplify_animate")
        print()
        print("    x_s, y_s = _simplify(x, y, nmin=100)")
        print("    metrics  = _simplify_error(x, y, x_s, y_s)")
        print("    _simplify_plot(x, y, x_s, y_s)")
        print("    _simplify_animate(x, y, 'simplify.gif')")
        print()
        print("  Or from the command line:")
        print()
        print("    python simplify.py data.csv -o out.csv --metrics")
        print("    python simplify.py data.csv --plot")
        print("    python simplify.py data.csv --animate simplify.gif")
        print()
        print("  Run with --help for all options.")
        print("=" * 60)