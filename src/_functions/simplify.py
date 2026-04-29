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

import warnings
from typing import Optional, Tuple, Union, Sequence

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
    warn_below_r2: Optional[float] = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic downsampling of a curve y(x) to ``nmin`` points,
    preserving the most physically and visually important features.

    This is useful when a simulation or measurement produces thousands of
    data points but only a compact, faithful representation is needed for
    output or storage.

    Algorithm overview
    ------------------
    Three independent strategies select "important" indices, which are
    merged together with the two endpoints into a pool of feature points.
    A prominence-based filter promotes a subset to mandatory, and a fixed
    point budget then trims the pool down to ``nmin`` points using
    hierarchical-bisection priority.

    1. **Menger curvature detection** (on rescaled coordinates)
       Computes the Menger curvature κ for each triplet of consecutive
       points using ``(x, y)`` rescaled to ``[0, 1]`` so the threshold
       is unit-free.  Points where ``κ > grad_inc`` are kept.  These mark
       sharp bends — shocks, discontinuities, or phase transitions.

    2. **Sign-change detection** (local extrema)
       Keeps every point where the first derivative changes sign, i.e.,
       every local minimum and maximum of ``y(x)``.

    3. **Cumulative-distance sampling** (uniform arc-length in y)
       The total variation of ``y`` (``sum(|diff(y)|)``) is divided into
       ``nmin`` equal bins.  One point is selected at each bin boundary,
       giving dense sampling where ``y`` changes rapidly and sparse
       sampling where it is nearly flat.

    4. **Topological-persistence filter** (mandatory set)
       ``_peak_prominences`` computes the prominence of every local
       extremum.  Extrema whose prominence exceeds 5 % of the y-range
       are *mandatory* — they are always kept, so deep dips or tall
       spikes don't flicker in and out as ``nmin`` varies.

    5. **Budget-based selection with mandatory override**
       Output size is normally ``nmin``.  Endpoints and every high-
       prominence extremum are always retained — if that set already
       exceeds ``nmin``, output size is the mandatory-set size (we'd
       rather overshoot the budget than drop a real high-prominence
       feature).  Remaining slots are filled in hierarchical-bisection
       order (endpoints → midpoint → quartiles → …) so the subset at
       any budget N is a superset of the subset at N − 1.

    6. **Reconstruction-quality warning** (post-hoc, optional)
       After selection, the linear interpolation R² of the simplified
       curve against the original grid is computed.  If it falls below
       ``warn_below_r2``, a ``UserWarning`` is emitted advising the user
       to raise ``nmin``.  Pass ``None`` to disable the warning.

    For a perfectly flat curve (zero total variation), the algorithm
    falls back to ``nmin`` uniformly spaced indices.

    Input/output contract
    ---------------------
    Input may be ascending, descending, or non-monotonic in x.  Output
    values are returned in the caller's original positional order, so
    ascending stays ascending and descending stays descending.  For
    non-monotonic input, output is a thinned subsequence in the input's
    original order.

    Parameters
    ----------
    x_arr : array-like
        Independent variable (e.g., position, time, wavelength).
        Must be the same length as ``y_arr``.
    y_arr : array-like
        Dependent variable (e.g., temperature, density, flux).
        Must be the same length as ``x_arr``.
    nmin : int, optional
        Target output size.  Clamped to >= 100.  Output is normally
        ``nmin`` points; it may be larger when the curve has more than
        ``nmin`` high-prominence extrema (those are never dropped).
        Default is 100.
    grad_inc : float, optional
        Menger curvature threshold on rescaled ``[0, 1]`` axes.  Lower
        values keep more sharp-bend points; higher values keep fewer.
        Default is 1.0.
    warn_below_r2 : float or None, optional
        Reconstruction R² threshold for emitting a UserWarning.  After
        selection, the simplified curve is linearly interpolated back
        onto the original x-grid; if R² falls below this value, a
        warning is raised.  Pass ``None`` to disable.  Default is 0.9.

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

    # Sort by x and work on an ascending copy. `np.interp` (used in the
    # post-hoc R² warning check) requires its reference x to be ascending;
    # the rest of the algorithm is sequence-based (curvature on triplets,
    # sign changes, cumulative |Δy|, peak persistence) and is unaffected
    # by the temporary reordering. Output values are restored to the
    # caller's original positional order on every return path: ascending
    # stays ascending, descending stays descending, and a non-monotonic
    # input comes back as a thinned subsequence in its original order.
    x_orig = x
    y_orig = y
    sort_order = np.argsort(x, kind="stable")
    needs_reorder = (
        sort_order.size > 1
        and not bool(np.array_equal(sort_order, np.arange(sort_order.size)))
    )
    if needs_reorder:
        x = x[sort_order]
        y = y[sort_order]

    def _restore(working_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map indices in the (ascending) working array back to caller's
        positional order and return the corresponding original values."""
        idx = np.asarray(working_idx, dtype=np.int64).ravel()
        if needs_reorder:
            idx = sort_order[idx]
        idx = np.sort(idx)
        return x_orig[idx], y_orig[idx]

    # If the array is already short enough, return as-is.
    if nmin >= x.size:
        return x_orig, y_orig
    # Enforce a floor of 100 samples so the output is always useful.
    nmin = max(int(nmin), 100)

    # =====================================================================
    # Strategy 1: Menger curvature feature detection
    # =====================================================================
    # Compute Menger curvature for each interior triplet of consecutive
    # points.  The Menger curvature κ_i is the reciprocal of the
    # circumradius of (P_{i-1}, P_i, P_{i+1}).  High curvature marks
    # sharp bends — shocks, discontinuities, phase transitions.
    #
    # Curvature has units of 1/length in the (x, y) plane, so a fixed
    # threshold like ``grad_inc=1.0`` only behaves consistently when the
    # axes share a common scale.  Rescale (x, y) to the unit square for
    # the curvature computation only — the cumulative-distance step,
    # sign-change detection, and the post-hoc R² check still operate on
    # raw arrays.
    range_x = float(x[-1] - x[0])
    range_y = float(np.nanmax(y) - np.nanmin(y))
    inv_x = 1.0 / range_x if range_x > 1e-30 else 1.0
    inv_y = 1.0 / range_y if range_y > 1e-30 else 1.0

    dx1 = np.diff(x[:-1]) * inv_x                  # x[i] - x[i-1]
    dy1 = np.diff(y[:-1]) * inv_y                  # y[i] - y[i-1]
    dx2 = np.diff(x[1:])  * inv_x                  # x[i+1] - x[i]
    dy2 = np.diff(y[1:])  * inv_y                  # y[i+1] - y[i]

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
    # `np.diff(np.sign(grad))` flags between indices i and i+1, so the raw
    # sign-change index is half a step before/after the actual extremum.
    # Refine to the true extremum so prominence classification (which
    # compares y[i] to its immediate neighbours) sees the real peak/trough.
    grad = np.gradient(y)
    sd = np.diff(np.sign(grad))
    sc = np.where(sd != 0)[0]
    if sc.size > 0:
        is_max_transition = sd[sc] < 0     # + → -  (local maximum)
        y_at_sc = y[sc]
        y_at_next = y[sc + 1]
        pick_i = np.where(
            is_max_transition,
            y_at_sc >= y_at_next,           # max: pick whichever y is larger
            y_at_sc <= y_at_next,           # min: pick whichever y is smaller
        )
        important_sign = np.unique(np.where(pick_i, sc, sc + 1))
    else:
        important_sign = np.array([], dtype=int)

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
    # Prominence ranks extrema by amplitude. Stored sorted descending so
    # the budget-trim step can cheaply take the most prominent first.
    if important_sign.size > 0 and y_range > 0:
        proms = _peak_prominences(y, important_sign)
        keep_mask = proms >= prom_thresh
        kept_idx = important_sign[keep_mask]
        kept_proms = proms[keep_mask]
        prom_order = np.argsort(kept_proms, kind="stable")[::-1]
        prominent_idx = kept_idx[prom_order]
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
        return _restore(idx)

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
    # Budget-based selection with mandatory-feature override.
    #
    # Priority order over candidate indices:
    #   1. endpoints (always — first and last input points)
    #   2. high-prominence extrema, sorted by prominence DESC
    #   3. remaining merged-pool points, in hierarchical-bisection order
    #
    # Output size is ``max(nmin, |endpoints ∪ prominent_idx|)`` — i.e.
    # ``nmin`` is the *normal* target, but every high-prominence feature
    # is always kept even if that pushes the count over ``nmin``. This
    # matches the user-visible promise that prominent extrema (peak/
    # trough with prominence ≥ 5 % of the y-range) never disappear.
    # =================================================================
    if len(merged) > nmin:
        # Build hierarchical bisection ordering over merged-pool positions.
        n_m = len(merged)
        order = np.empty(n_m, dtype=int)
        order[0] = 0
        order[1] = n_m - 1
        count = 2
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
        bisection_pool = merged[order[:count]]

        # Build the priority list with stable first-seen-order deduplication.
        endpoints_idx = np.array([0, x.size - 1], dtype=int)
        all_priorities = np.concatenate([endpoints_idx, prominent_idx, bisection_pool])
        _, unique_pos = np.unique(all_priorities, return_index=True)
        priority_indices = all_priorities[np.sort(unique_pos)]

        # Mandatory floor: endpoints + every prominent extremum. If this
        # exceeds nmin, the mandatory floor wins (we'd rather over-shoot
        # the budget than drop a real high-prominence feature).
        mandatory_set = np.unique(np.concatenate([endpoints_idx, prominent_idx]))
        budget = max(int(nmin), int(mandatory_set.size))
        budget = min(budget, priority_indices.size)
        merged = np.sort(priority_indices[:budget])

    # =================================================================
    # Post-hoc reconstruction-quality warning.  Compute R² of the
    # linearly interpolated simplified curve against the input grid;
    # warn if it falls below the threshold so callers know nmin is too
    # tight for this particular curve.
    # =================================================================
    if warn_below_r2 is not None and merged.size >= 2:
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot > 0:
            y_interp = np.interp(x, x[merged], y[merged])
            r2 = 1.0 - float(np.sum((y - y_interp) ** 2)) / ss_tot
            if r2 < warn_below_r2:
                warnings.warn(
                    f"_simplify: reconstruction R²={r2:.3f} below "
                    f"threshold {warn_below_r2:.2f} "
                    f"(N_in={x.size}, N_out={merged.size}). "
                    f"Consider increasing nmin.",
                    UserWarning,
                    stacklevel=2,
                )

    return _restore(merged)


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

    # np.interp requires ascending reference x; sort the simplified curve
    # internally so this works for descending or non-monotonic inputs too.
    if x_s.size > 1 and not bool(np.all(np.diff(x_s) >= 0)):
        ord_s = np.argsort(x_s, kind="stable")
        x_s = x_s[ord_s]
        y_s = y_s[ord_s]

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
