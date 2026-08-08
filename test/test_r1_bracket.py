"""
Tests for ``get_bubbleParams.solve_R1`` (shared R1 root-finding helper).

Pins the Phase-1 safety fix: the brentq bracket is the full [0, R2] (the
former [1e-3*R2, R2] missed small roots and raised), ``Lmech_total <= 0``
short-circuits to R1 = 0 without root finding, and failures raise instead
of fabricating a fallback value.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize

import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
from trinity.phase1b_energy_implicit.get_betadelta import compute_R1_Pb

# K = Lmech / (v_mech * Eb); the root satisfies r1 = sqrt(K * (R2^3 - r1^3)),
# so for r1 << R2 the analytic limit is r1 ≈ sqrt(K * R2^3).

R2 = 10.0


def test_small_root_old_bracket_raises_new_converges():
    # K = 1e-9 -> root ~ 1e-3, below the old bracket's lower end 1e-3*R2 = 1e-2
    Lmech, v_mech, Eb = 1.0, 1e4, 1e5
    args = ([Lmech, Eb, v_mech, R2],)

    with pytest.raises(ValueError):
        scipy.optimize.brentq(get_bubbleParams.get_r1, 1e-3 * R2, R2, args=args)

    R1 = get_bubbleParams.solve_R1(R2, Eb, Lmech, v_mech, 5 / 3)
    analytic = np.sqrt(Lmech / v_mech / Eb * R2**3)
    assert abs(R1 - analytic) / analytic < 0.01


def test_lmech_nonpositive_returns_zero_without_brentq(monkeypatch):
    def _no_brentq(*a, **k):
        raise AssertionError("brentq must not be called for Lmech_total <= 0")

    monkeypatch.setattr(scipy.optimize, "brentq", _no_brentq)
    assert get_bubbleParams.solve_R1(R2, 1e5, 0.0, 2e3, 5 / 3) == 0.0
    assert get_bubbleParams.solve_R1(R2, 1e5, -1.0, 2e3, 5 / 3) == 0.0


def test_midrange_root_matches_old_bracket():
    # K = 1/999 -> root exactly 1.0, well inside the old bracket
    Lmech, v_mech, Eb = 1.0, 1.0, 999.0
    args = ([Lmech, Eb, v_mech, R2],)

    old = scipy.optimize.brentq(get_bubbleParams.get_r1, 1e-3 * R2, R2, args=args)
    new = get_bubbleParams.solve_R1(R2, Eb, Lmech, v_mech, 5 / 3)
    assert old == pytest.approx(1.0, rel=1e-9)
    assert new == pytest.approx(old, rel=1e-9)


def test_failure_raises_instead_of_fabricating():
    # NaN energy poisons the equation; the old code fabricated R1 = 0.01*R2
    with pytest.raises(ValueError):
        get_bubbleParams.solve_R1(R2, np.nan, 1.0, 1.0, 5 / 3)


def test_compute_R1_Pb_returns_true_small_root():
    # Same small-root case as above through the Phase-1b wrapper: the old
    # fallback would have fabricated R1 = 0.01*R2 = 0.1
    Lmech, v_mech, Eb = 1.0, 1e4, 1e5
    R1, Pb = compute_R1_Pb(R2, Eb, Lmech, v_mech, gamma_adia=5.0 / 3.0)
    analytic = np.sqrt(Lmech / v_mech / Eb * R2**3)
    assert abs(R1 - analytic) / analytic < 0.01
    assert np.isfinite(Pb) and Pb > 0


# =============================================================================
# Regression: gamma_adia must reach the R1 solve.
#
# get_r1 balances wind ram pressure at R1 against bubble pressure:
#     R1**2 = 2*Lmech*(R2**3 - R1**3) / (3*(gamma - 1) * v_mech * Eb)
# At gamma = 5/3, 3*(gamma-1) is exactly 2 and cancels the leading 2, so the
# index vanished from the expression -- analytically, which is why grepping the
# Weaver chain for "5/3" found nothing. bubble_E2P meanwhile honoured whatever
# the user set, so the two halves disagreed by 67 % at gamma = 1.4 (audit
# SIGN-01).
# =============================================================================


def test_solve_R1_at_default_gamma_is_bit_identical_to_the_gamma_free_form():
    """The fix must not move the default-gamma root by even one bit.

    gamma_factor = 3*(gamma-1)/2 is exactly 1.0 at gamma=5/3, and IEEE-754
    multiplication by 1.0 is exact, so get_r1 receives byte-identical arguments.
    """
    for R2, Eb, Lmech, v_mech in [
        (5.0, 1e5, 1e4, 2e3),
        (0.31, 6.6e5, 3.7e3, 3.7e3),
        (17.6, 2.0e7, 9.1e4, 1.2e3),
    ]:
        reference = scipy.optimize.brentq(
            get_bubbleParams.get_r1, 0.0, R2,
            args=([Lmech, Eb, v_mech, R2]),
        )
        assert get_bubbleParams.solve_R1(R2, Eb, Lmech, v_mech, 5 / 3) == reference


def test_solve_R1_actually_responds_to_gamma():
    """A non-default gamma must move R1, and by the predicted amount.

    R1 scales as 1/sqrt(3*(gamma-1)/2), so gamma=1.4 gives 1/sqrt(0.6) = 1.29099.
    """
    R2, Eb, Lmech, v_mech = 5.0, 1e5, 1e4, 2e3

    R1_default = get_bubbleParams.solve_R1(R2, Eb, Lmech, v_mech, 5 / 3)
    R1_soft = get_bubbleParams.solve_R1(R2, Eb, Lmech, v_mech, 1.4)

    assert R1_soft > R1_default, "a softer equation of state must push R1 outward"
    # R1**3 enters the bracket too, so the ratio is not exactly 1/sqrt(0.6);
    # bracket it generously but tightly enough to catch a dropped factor.
    ratio = R1_soft / R1_default
    assert 1.20 < ratio < 1.35, ratio


def test_solve_R1_requires_gamma_explicitly():
    """No default: silently inheriting 5/3 is the defect being removed."""
    with pytest.raises(TypeError):
        get_bubbleParams.solve_R1(5.0, 1e5, 1e4, 2e3)
