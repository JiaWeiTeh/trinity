"""Tests for the tolerant monotonicity guard used by find_nearest_higher.

Background: a backward bubble-temperature integration can leave tiny,
provably-numerical non-monotonicities in T_array (a sub-percent dip at the
T_init=3e4 outer edge, or an isolated single-point dense-output spike).  These
used to crash ``find_nearest_higher`` with a ``MonotonicError``.  The guard now
tolerates numerical noise -- an isolated single-point spike (any depth) or a
shallow, localized multi-point wiggle -- while still rejecting non-finite
profiles and *deep* or *sustained interior* non-monotonicity (a real inversion
or a dead integrator).  ``monotonic`` itself is left strict and unchanged.
"""
import numpy as np
import pytest

from trinity._functions import operations as ops


# --- monotonic() stays strict (byte-identical to before) --------------------

def test_monotonic_strict_true_on_clean_arrays():
    assert ops.monotonic(np.array([1.0, 2.0, 3.0, 4.0]))
    assert ops.monotonic(np.array([4.0, 3.0, 2.0, 1.0]))


def test_monotonic_strict_false_on_any_wiggle():
    # even a single tiny dip makes the strict checker False
    assert not ops.monotonic(np.array([1.0, 2.0, 1.999, 3.0]))


# --- _is_monotonic_or_tolerable: tolerate shallow + localized ---------------

def test_tolerable_clean_increasing_and_decreasing():
    assert ops._is_monotonic_or_tolerable(np.array([1.0, 2.0, 3.0, 4.0]))
    assert ops._is_monotonic_or_tolerable(np.array([4.0, 3.0, 2.0, 1.0]))


def test_tolerable_isolated_single_point_spike():
    # one point spikes ~0.5% above trend, recovers next step -> isolated
    L = np.linspace(1.0, 2.0, 200)
    L[100] += 0.005 * L[100]
    assert not ops.monotonic(L)
    assert ops._is_monotonic_or_tolerable(L)


def test_tolerable_boundary_dip():
    # a *sustained* sub-percent descending run confined to the first 1% of the
    # array (startup transient) -- tolerated only because it is at the boundary
    # (its 5-step length exceeds MAX_SPIKE_LEN, so the isolated clause does not
    # apply).
    L = np.linspace(3.0e4, 1.0e6, 5000)
    for k in range(3, 8):
        L[k] = L[2] * (1 - 1e-3 * (k - 2))   # 0.1%..0.5% below L[2], descending
    assert not ops.monotonic(L)
    assert ops._is_monotonic_or_tolerable(L)


# --- _is_monotonic_or_tolerable: reject deep or sustained -------------------

def test_reject_sustained_interior_inversion_even_if_shallow():
    # a shallow (<1%) but *sustained* dip in the interior is NOT tolerated:
    # it could be a real local temperature maximum, not numerical noise.
    L = np.linspace(1.0, 2.0, 1000).astype(float)
    # 50-point gently descending run in the middle (sustained wrong direction)
    L[500:550] = np.linspace(L[500], L[500] - 0.003 * L[500], 50)
    assert not ops._is_monotonic_or_tolerable(L)


def test_tolerate_isolated_deep_spike():
    # an isolated single-point spike is tolerated regardless of depth: one point
    # cannot be a physical inversion (closes the >1% single-point crash gap).
    L = np.linspace(1.0, 2.0, 200)
    L[100] += 0.5 * L[100]
    assert ops._is_monotonic_or_tolerable(L)


def test_reject_multipoint_deep_inversion():
    # a *multi-point* deep run (>=2 wrong steps) is still rejected -- the
    # single-point tolerance must not leak to genuine multi-point inversions.
    L = np.linspace(1.0, 2.0, 1000)
    L[500] = L[499] * 0.97
    L[501] = L[499] * 0.94   # two consecutive descending steps (~6% deep)
    assert not ops._is_monotonic_or_tolerable(L)


def test_reject_nonfinite():
    # a non-finite profile (e.g. a dead integrator) is a genuine failure
    L = np.linspace(1.0, 2.0, 200)
    L[150] = np.nan
    assert not ops._is_monotonic_or_tolerable(L)


def test_reject_dead_integrator_tail():
    # LSODA gives up: zero tail at the hot (inner) end -> huge drawdown
    L = np.linspace(3.0e4, 1.0e6, 500)
    L[400:] = 0.0
    assert not ops._is_monotonic_or_tolerable(L)


# --- find_nearest_higher: tolerated noise no longer crashes ------------------

def test_find_nearest_higher_does_not_crash_on_tolerable_spike():
    L = np.linspace(1.0e4, 1.0e6, 500)
    L[250] += 0.004 * L[250]   # isolated sub-percent spike
    # _CIEswitch-like lookup must return a sane in-range index, not raise
    idx = ops.find_nearest_higher(L, 10 ** 5.5)
    assert 0 <= idx < len(L)
    assert L[idx] >= 10 ** 5.5


def test_find_nearest_higher_still_raises_on_real_inversion():
    L = np.linspace(1.0, 2.0, 1000).astype(float)
    L[400:600] = np.linspace(L[400], L[400] - 0.2 * L[400], 200)  # deep sustained
    with pytest.raises(ops.MonotonicError):
        ops.find_nearest_higher(L, 1.5)


# --- direction fix: correct on both increasing and decreasing ----------------

def test_find_nearest_higher_direction_increasing():
    L = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    idx = ops.find_nearest_higher(L, 2.5)
    assert L[idx] >= 2.5 and L[idx] == 3.0


def test_find_nearest_higher_direction_decreasing():
    L = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    idx = ops.find_nearest_higher(L, 2.5)
    assert L[idx] >= 2.5 and L[idx] == 3.0
