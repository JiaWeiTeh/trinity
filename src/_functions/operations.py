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
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 5000)
    >>> y = np.sin(x) + 0.5 * np.sin(5 * x)
    >>> x_s, y_s = _simplify(x, y, nmin=100)
    >>> print(f"Reduced {x.size} points to {x_s.size} points")
    Reduced 5000 points to ... points
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


if __name__ == "__main__":
    _simplify_cli()

