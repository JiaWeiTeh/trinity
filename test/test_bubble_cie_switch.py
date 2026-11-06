"""The radius inserted where the bubble's T crosses 10**5.5 K stays strictly between its two grid points.

The grid descends and, near the switch, is spaced far below the former absolute xtol of 1e-8 pc (median 4e-11 pc on
the ladder), so the former search over the whole interpolation slice could return a root outside the two neighbours
and the grid stepped backwards (output-profiles/CIE_BRACKET.md). Values: the outer edge of a 3 pc bubble.
"""
import numpy as np
import scipy.optimize
from scipy.interpolate import interp1d

from trinity.bubble_structure.bubble_luminosity import _cie_switch_radius

TSW = 10**5.5
DR = 4e-11                                  # pc, the ladder's median grid spacing at the switch
R = 3.0 - DR * np.arange(60)                # descending, as the solver's grid
I = 25                                      # first point with T >= the switch


def _slice():
    return R[: I + 20]                      # r_interp in _bubble_luminosity


def test_root_between_its_two_grid_points():
    r_c = R[I - 1] - 0.37 * DR              # the crossing, between R[I-1] and R[I]
    T = TSW * np.exp((r_c - R) / 1e-3)      # T rises inwards through the switch, by e over 1e-3 pc
    assert T[I - 1] < TSW <= T[I]
    ri = _slice()
    f = interp1d(ri, T[: I + 20] - TSW, kind="cubic")
    old = scipy.optimize.brentq(f, np.min(ri), np.max(ri), xtol=1e-8)
    assert not (R[I] < old < R[I - 1])      # the former search: outside, the grid stepped backwards
    new = _cie_switch_radius(f, R, I, ri)
    assert R[I] < new < R[I - 1]
    assert abs(new - r_c) < 1e-2 * DR


def test_root_on_a_grid_point_is_kept_off_it():
    f = lambda r: R[I] - r                  # noqa: E731  zero exactly on R[I]
    new = _cie_switch_radius(f, R, I, _slice())
    assert R[I] < new < R[I - 1]


def test_no_sign_change_on_the_bracket_falls_back():
    f = lambda r: r - R[40]                 # noqa: E731  same sign on R[I], R[I-1]; a root elsewhere in the slice
    ri = _slice()
    assert _cie_switch_radius(f, R, I, ri) == scipy.optimize.brentq(f, np.min(ri), np.max(ri), xtol=1e-8)
