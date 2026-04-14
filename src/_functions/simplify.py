#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curve-simplification module.

Heuristic downsampling of 1-D curves while preserving physically and
visually important features (sharp bends, local extrema, arc-length
uniformity).  No dependencies beyond numpy / stdlib.

Functions
---------
_simplify          Core downsampling algorithm.
_simplify_error    Error metrics (RMSE, MAE, R², compression, …).
_peak_prominences  1-D topological persistence (O(n log n)).
"""

from typing import Tuple, Union, Sequence

import numpy as np


def _prev_next_strict(y: np.ndarray, greater: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-pass monotonic-stack computation of previous/next strictly-greater
    (or strictly-less) indices for every position in ``y``.

    Each index is pushed and popped at most once, so the total cost is
    amortised O(n) despite the nested ``while`` loop.  Returns two int64
    arrays ``prev_s`` and ``next_s`` such that for every ``i``:

    * ``prev_s[i]`` is the largest ``j < i`` with ``y[j] > y[i]`` (greater
      case) or ``y[j] < y[i]`` (less case); ``-1`` if no such ``j`` exists.
    * ``next_s[i]`` is the smallest ``j > i`` with the same condition;
      ``n`` if no such ``j`` exists.
    """
    n = y.size
    prev_s = np.empty(n, dtype=np.int64)
    next_s = np.full(n, n, dtype=np.int64)
    stk: list = []
    if greater:
        for i in range(n):
            yi = y[i]
            # Pop anything not strictly greater than y[i]; those are
            # elements for which i is the next strictly-greater position.
            while stk and y[stk[-1]] <= yi:
                next_s[stk.pop()] = i
            prev_s[i] = stk[-1] if stk else -1
            stk.append(i)
    else:
        for i in range(n):
            yi = y[i]
            while stk and y[stk[-1]] >= yi:
                next_s[stk.pop()] = i
            prev_s[i] = stk[-1] if stk else -1
            stk.append(i)
    return prev_s, next_s


def _sparse_table(y: np.ndarray, reducer) -> np.ndarray:
    """
    Build a sparse table for O(1) range-min or range-max queries on ``y``.

    ``reducer`` is ``np.minimum`` or ``np.maximum``.  Preprocessing is
    O(n log n) — fully vectorised numpy — and storage is (log2(n)+1, n).
    A later query over ``[lo, hi]`` uses two overlapping blocks of
    length 2**k, cf. Bender–Farach-Colton.
    """
    n = y.size
    k_max = max(1, int(np.floor(np.log2(max(n, 1)))) + 1)
    st = np.empty((k_max, n), dtype=y.dtype)
    st[0] = y
    for k in range(1, k_max):
        step = 1 << (k - 1)
        span = 1 << k
        limit = n - span + 1
        if limit <= 0:
            st[k] = st[k - 1]
        else:
            st[k, :limit] = reducer(
                st[k - 1, :limit], st[k - 1, step:step + limit]
            )
            st[k, limit:] = st[k - 1, limit:]
    return st


def _rmq(st: np.ndarray, lo: np.ndarray, hi: np.ndarray, reducer) -> np.ndarray:
    """
    Vectorised range-min/range-max query over inclusive intervals
    ``[lo[i], hi[i]]`` using a precomputed sparse table ``st``.
    """
    length = hi - lo + 1
    k = np.floor(np.log2(length)).astype(np.int64)
    return reducer(st[k, lo], st[k, hi - (1 << k) + 1])


def _peak_prominences(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Compute topological persistence (peak prominence) for local extrema.

    For each index ``p`` in ``idx`` (a local maximum or minimum of ``y``),
    returns how much the curve must descend from a max (or ascend from a
    min) before reaching a point more extreme than ``y[p]``, or the
    boundary.  This is equivalent to the persistence of the extremum in
    the sublevel-set filtration of ``y``.

    A very tall, narrow spike has large prominence.  A small wiggle has
    small prominence.  The measure is *not* affected by a feature's
    width — only its amplitude relative to surrounding terrain.

    Complexity
    ----------
    O(n log n) total, fully deterministic: two monotonic-stack passes
    of length ``n`` produce the prev/next strictly-greater and
    strictly-less indices, two sparse tables give O(1) range-min /
    range-max queries, and all per-candidate work is vectorised.

    Parameters
    ----------
    y : np.ndarray
        The 1-D signal.
    idx : np.ndarray
        Indices of local extrema (peaks and troughs).

    Returns
    -------
    prominences : np.ndarray
        Non-negative prominence value for each index in ``idx``.
    """
    n = y.size
    proms = np.zeros(idx.size, dtype=float)
    if idx.size == 0 or n == 0:
        return proms

    p = np.asarray(idx, dtype=np.int64)
    y_p = y[p]

    # Classify each candidate as (weak) max / min from immediate neighbours.
    lo_nb = np.maximum(p - 1, 0)
    hi_nb = np.minimum(p + 1, n - 1)
    is_max = (y_p >= y[lo_nb]) & (y_p >= y[hi_nb])
    is_min = (y_p <= y[lo_nb]) & (y_p <= y[hi_nb]) & ~is_max

    # MAX candidates: walk outward until a point with y > y[p]; track
    # the minimum of y on each side using a min-RMQ.
    if np.any(is_max):
        PG, NG = _prev_next_strict(y, greater=True)
        st_min = _sparse_table(y, np.minimum)
        pm = p[is_max]
        pg = PG[pm]
        ng = NG[pm]
        # Inclusive walk ranges: left = [pg+1, pm-1], right = [pm+1, ng-1].
        # Boundaries: pg = -1 gives lo=0; ng = n gives hi=n-1.
        left_lo = pg + 1
        left_hi = pm - 1
        right_lo = pm + 1
        right_hi = ng - 1
        left_valid = left_lo <= left_hi
        right_valid = right_lo <= right_hi
        left_min = np.full(pm.size, np.inf)
        right_min = np.full(pm.size, np.inf)
        if np.any(left_valid):
            left_min[left_valid] = _rmq(
                st_min, left_lo[left_valid], left_hi[left_valid], np.minimum
            )
        if np.any(right_valid):
            right_min[right_valid] = _rmq(
                st_min, right_lo[right_valid], right_hi[right_valid], np.minimum
            )
        # Walks always include at least one neighbour for a true extremum,
        # so at least one side is valid.  Use the valid sides; if a side
        # is empty (shouldn't happen for real extrema) treat its shoulder
        # as +inf so the other side dominates.
        shoulder = np.maximum(left_min, right_min)
        proms[is_max] = y[pm] - shoulder

    # MIN candidates: mirror image using max-RMQ.
    if np.any(is_min):
        PL, NL = _prev_next_strict(y, greater=False)
        st_max = _sparse_table(y, np.maximum)
        pn = p[is_min]
        pl = PL[pn]
        nl = NL[pn]
        left_lo = pl + 1
        left_hi = pn - 1
        right_lo = pn + 1
        right_hi = nl - 1
        left_valid = left_lo <= left_hi
        right_valid = right_lo <= right_hi
        left_max = np.full(pn.size, -np.inf)
        right_max = np.full(pn.size, -np.inf)
        if np.any(left_valid):
            left_max[left_valid] = _rmq(
                st_max, left_lo[left_valid], left_hi[left_valid], np.maximum
            )
        if np.any(right_valid):
            right_max[right_valid] = _rmq(
                st_max, right_lo[right_valid], right_hi[right_valid], np.maximum
            )
        shoulder = np.minimum(left_max, right_max)
        proms[is_min] = shoulder - y[pn]

    # Clamp tiny negative values from floating-point rounding (prominence
    # is non-negative by definition).
    np.clip(proms, 0.0, None, out=proms)
    return proms


def _simplify(
    x_arr: Union[np.ndarray, Sequence[float]],
    y_arr: Union[np.ndarray, Sequence[float]],
    nmin: int = 100,
    grad_inc: float = 1.0,
    r2_target: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic downsampling of a curve y(x) to approximately ``nmin`` points,
    preserving the most physically and visually important features.

    This is useful when a simulation or measurement produces thousands of
    data points but only a compact, faithful representation is needed for
    output or storage.

    Algorithm overview
    ------------------
    Three independent strategies select "important" indices, which are
    merged together with the two endpoints into a pool of feature points.
    A prominence-based filter then marks a subset as mandatory, and a
    final R²-based thinning step chooses the smallest subset that meets
    the requested quality target.

    1. **Menger curvature detection** (exact discrete curvature)
       Computes the Menger curvature κ for each triplet of consecutive
       points — the reciprocal of the circumradius of the triangle they
       form.  Points where ``κ > grad_inc`` are kept.  These mark sharp
       bends in the curve — shocks, discontinuities, or phase transitions.

    2. **Sign-change detection** (local extrema)
       Keeps every point where the first derivative changes sign, i.e.,
       every local minimum and maximum of ``y(x)``.

    3. **Cumulative-distance sampling** (uniform arc-length in y)
       The total variation of ``y`` (i.e., ``sum(|diff(y)|)``) is divided
       into ``nmin`` equal "distance bins".  One point is selected at each
       bin boundary.  This gives dense sampling where ``y`` changes rapidly
       and sparse sampling where ``y`` is nearly flat — adapting
       automatically to the curve shape.

    4. **Topological-persistence filter** (mandatory set)
       ``_peak_prominences`` computes the prominence of every local
       extremum (the minimum descent required to reach a strictly
       higher point).  Extrema whose prominence exceeds 5 % of the
       y-range are flagged as *mandatory* — they are always retained
       regardless of the R² budget, so a deep dip or tall spike never
       flickers in and out across neighbouring point counts.

    5. **R²-based thinning** (optional, ``r2_target``)
       The remaining feature points are traversed in hierarchical-
       bisection order (endpoints → midpoint → quartiles → …) so that
       the subset at budget N is always a superset of the subset at
       N-1.  A binary search plus a stability check picks the smallest
       ``k`` for which the trial  ``mandatory ∪ bisection[:k]``
       achieves ``R² ≥ r2_target``.

    For a perfectly flat curve (zero total variation), the algorithm falls
    back to uniformly spaced indices.

    Parameters
    ----------
    x_arr : array-like
        Independent variable (e.g., position, time, wavelength).
        Must be the same length as ``y_arr``.
    y_arr : array-like
        Dependent variable (e.g., temperature, density, flux).
        Must be the same length as ``x_arr``.
    nmin : int, optional
        Target *minimum* number of output samples.  Acts as the number of
        bins for the cumulative-distance sampler; the final count may
        differ after curvature/sign-change/prominence merging and the
        R² thinning step.  Clamped to >= 100.  Default is 100.
    grad_inc : float, optional
        Menger curvature threshold.  A point is flagged as "important"
        when the Menger curvature of its triplet exceeds this value.
        Units are 1/length in the (x, y) plane, so the appropriate
        value depends on the scale of the data.  Lower values keep more
        points (more sensitive to bends); higher values keep fewer.
        Default is 1.0.
    r2_target : float, optional
        Target R² (coefficient of determination).  After the feature
        detection selects important points, the result is thinned to the
        minimum number of points that still achieves this R² value.
        Set to ``None`` to disable R²-based thinning and keep all
        detected feature points.  Default is 0.99.

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
    # Strategy 1: Menger curvature feature detection
    # =====================================================================
    # Compute Menger curvature for each interior triplet of consecutive
    # points.  The Menger curvature κ_i is the reciprocal of the
    # circumradius of (P_{i-1}, P_i, P_{i+1}).  High curvature marks
    # sharp bends — shocks, discontinuities, phase transitions.
    dx1 = np.diff(x[:-1])                          # x[i] - x[i-1]
    dy1 = np.diff(y[:-1])                          # y[i] - y[i-1]
    dx2 = np.diff(x[1:])                           # x[i+1] - x[i]
    dy2 = np.diff(y[1:])                           # y[i+1] - y[i]

    # 2× signed area of the triangle formed by the triplet.
    cross = dx1 * (dy1 + dy2) - dy1 * (dx1 + dx2)

    # Side lengths of the triangle.
    a = np.sqrt(dx1**2 + dy1**2)
    b = np.sqrt(dx2**2 + dy2**2)
    c_len = np.sqrt((dx1 + dx2)**2 + (dy1 + dy2)**2)

    denom = a * b * c_len
    denom[denom < 1e-30] = 1e-30                   # guard degenerate triplets

    kappa = 2.0 * np.abs(cross) / denom            # Menger curvature, len n-2

    # Keep interior indices where curvature exceeds the threshold.
    # kappa[i] corresponds to original index i+1.
    important_curv = np.where(kappa > grad_inc)[0] + 1

    # Keep indices where the derivative changes sign (local extrema).
    grad = np.gradient(y)
    # np.sign(grad) is -1, 0, or +1; a nonzero diff marks a sign flip.
    important_sign = np.where(np.diff(np.sign(grad)) != 0)[0]

    # ---------------------------------------------------------------
    # Topological persistence: identify the extrema that are large
    # enough to be genuine features of the curve (as opposed to
    # noise-level perturbations), and mark them as MANDATORY — they
    # are always included in every trial subset, so once a big dip
    # or spike is in the output at budget N it is also in the output
    # at N+1, N+2, …  This prevents the "dip flickers in at some
    # random n" artefact.
    #
    # _peak_prominences is O(n log n), so running it directly on
    # every sign-change index is cheap even for noisy curves with
    # thousands of extrema.
    # ---------------------------------------------------------------
    y_range = float(np.nanmax(y) - np.nanmin(y))
    prom_thresh_frac = 0.05               # 5 % of total y-range
    prom_thresh = prom_thresh_frac * y_range
    # Prominence is only consumed by the R² thinning step; skip it when
    # the caller has disabled thinning (``r2_target=None``) to save the
    # O(n log n) sparse-table build on large inputs.
    if (r2_target is not None and r2_target < 1.0
            and important_sign.size > 0 and y_range > 0):
        proms = _peak_prominences(y, important_sign)
        prominent_idx = important_sign[proms >= prom_thresh]
    else:
        prominent_idx = np.array([], dtype=int)

    # =====================================================================
    # Strategy 2: Cumulative-distance sampling in y
    # =====================================================================
    # Cumulative absolute change in y (total variation up to each point).
    y_cum = np.cumsum(np.abs(np.diff(y)))
    total_variation = float(y_cum[-1]) if y_cum.size > 0 else 0.0

    if not np.isfinite(total_variation) or total_variation == 0:
        # Special case: perfectly flat curve (or all NaN).
        # Fall back to uniformly spaced indices.
        idx = np.unique(np.linspace(0, x.size - 1, nmin).astype(int))
        return x[idx], y[idx]

    # Maximum allowed cumulative y-distance between kept points.
    # Dividing the total variation by nmin gives roughly nmin bins.
    maxdist = total_variation / nmin

    # Assign each point to a "distance bin".  When the bin number changes
    # between consecutive points, that boundary is a selected sample.
    bins = (y_cum / maxdist).astype(int)
    idx_dist = np.where(bins[:-1] != bins[1:])[0]

    # =====================================================================
    # Merge all candidates + endpoints via boolean mask
    # =====================================================================
    mask = np.zeros(x.size, dtype=bool)
    mask[0] = True                                  # first point
    mask[-1] = True                                 # last point
    mask[important_curv] = True                     # Menger curvature
    mask[important_sign] = True                     # local extrema
    mask[idx_dist] = True                           # cumulative-distance
    merged = np.where(mask)[0]

    # =================================================================
    # R²-based build-up: start from 5 points and increase until R² target.
    # Uses hierarchical bisection of the merged index array so that the
    # subset at budget N is always a superset of the subset at N-1.
    # This prevents turning points from randomly appearing/disappearing
    # at intermediate point counts.
    # =================================================================
    if r2_target is not None and r2_target < 1.0 and len(merged) > 5:
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot > 0:
            # Build a hierarchical bisection ordering of merged indices.
            # Level 0: endpoints (first and last of merged).
            # Level 1: midpoint of merged.
            # Level 2: quartile points.
            # Level k: 2^(k-1) new points at each level.
            # Taking the first N from this ordering always includes the
            # first N-1, and spreads points evenly across the x-range.
            n_m = len(merged)
            order = np.empty(n_m, dtype=int)
            order[0] = 0
            order[1] = n_m - 1
            count = 2

            # BFS-style bisection: queue of (lo, hi) intervals to split.
            queue = [(0, n_m - 1)]
            while queue:
                next_queue = []
                for lo_q, hi_q in queue:
                    if hi_q - lo_q <= 1:
                        continue
                    mid_q = (lo_q + hi_q) // 2
                    order[count] = mid_q
                    count += 1
                    next_queue.append((lo_q, mid_q))
                    next_queue.append((mid_q, hi_q))
                queue = next_queue

            # order[:count] maps position-in-merged to bisection priority.
            # Convert to actual data indices via merged[order].
            bisection_pool = merged[order[:count]]

            # Split into MANDATORY (prominent extrema, always kept) and
            # OPTIONAL (bisection pool minus anything already mandatory).
            # Every trial subset is  mandatory ∪ optional[:k]  — i.e. the
            # binary search only varies how many optional points to add;
            # mandatory points are never removable.  This is a strict
            # strengthening of "prepend prominent to the pool": there is
            # no value of k for which a mandatory point is absent.
            if prominent_idx.size > 0:
                mandatory = prominent_idx
                rest_mask = ~np.isin(bisection_pool, mandatory)
                optional = bisection_pool[rest_mask]
            else:
                mandatory = np.array([], dtype=int)
                optional = bisection_pool

            # Helper: compute R² for mandatory ∪ first k of optional.
            def _r2_at(k):
                if k <= 0:
                    trial = np.sort(mandatory) if mandatory.size else optional[:5]
                else:
                    trial = np.sort(np.concatenate([mandatory, optional[:k]]))
                y_interp = np.interp(x, x[trial], y[trial])
                return 1.0 - np.sum((y - y_interp) ** 2) / ss_tot

            # Minimum optional count so the total trial size is >= 5.
            k_min = max(0, 5 - int(mandatory.size))
            k_max = len(optional)

            # Binary search: minimum k for which R² >= target.
            lo, hi = k_min, k_max
            while lo < hi:
                mid = (lo + hi) // 2
                if _r2_at(mid) >= r2_target:
                    hi = mid
                else:
                    lo = mid + 1

            # Stability check: scan forward from lo until R² >= target
            # for 3 consecutive k, guarding against local dips caused
            # by noisy points.
            stable_run = 0
            k = lo
            while k <= k_max:
                if _r2_at(k) >= r2_target:
                    stable_run += 1
                    if stable_run >= 3:
                        break
                else:
                    stable_run = 0
                k += 1

            merged = np.sort(np.concatenate([mandatory, optional[:k]]))

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
