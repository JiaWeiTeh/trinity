"""Pin the ``dt_switchon = 1e-3`` Myr R1 ramp in ``get_effective_bubble_pressure``
(magic-number audit finding #2 — docs/dev/magic-numbers/SWEEP2_PLAN.md §5,
SWITCHON_BRIEF.md).

The ramp is LOAD-BEARING, not inert: with it ablated, ``f1edge_hidens``
(nCore=1e6) drains Eb 180 -> 29 au within four segments and the bubble-structure
solve grinds ~20 min/segment instead of completing in minutes (measured twice,
independently: docs/dev/phase1a-init/data/e8b_hidens_noramp_STALLED.csv and
docs/dev/magic-numbers/data/switchon_stall_probe.csv). Its trajectory cost on
healthy configs is bounded at |dR2| <= 0.006-0.017% beyond the early window.
These tests exist so that deleting the ramp as "inert" — the audit's original,
now-struck recommendation — fails loudly instead of stalling the stiff edge in
production. They pin the *current, measured* behaviour; a gated scale-relative
successor would re-pin them.

State values are the segment-1 entry state of the f1edge_hidens run (the regime
the ramp protects), with R1/R2 at the measured peak leverage (R1/R2)^3 = 0.673.
"""
import numpy as np
import pytest

from trinity.bubble_structure.get_bubbleParams import (
    bubble_E2P,
    get_effective_bubble_pressure,
)

# f1edge_hidens segment-1 entry (docs/dev/magic-numbers/data/
# switchon_repro_hidens_active.csv row 1), TRINITY AU:
EB = 180.33898437948136     # Msun*pc^2/Myr^2
R2 = 7.314603970843167e-4   # pc
R1 = 0.673 ** (1.0 / 3.0) * R2  # peak measured ramp leverage
GAMMA = 5.0 / 3.0
TSF = 0.0
DT_SWITCHON = 1e-3          # Myr — the constant under pin


def _p(t):
    return get_effective_bubble_pressure(
        current_phase='energy', Eb=EB, R2=R2, R1=R1, gamma=GAMMA, t=t, tSF=TSF)


def test_ramp_is_linear_in_time_and_suppresses_R1():
    """Inside the window the ramp hands bubble_E2P exactly (t-tSF)/1e-3 * R1.
    Suppressing R1 inflates the shell volume, so the ramped pressure is strictly
    BELOW the unramped one — the direction that keeps early Pb integrable at
    nCore=1e6 (delete the ramp and Eb collapses; see module docstring)."""
    p_full = bubble_E2P(EB, R2, R1, GAMMA)
    for frac in (0.1, 0.5, 0.9):
        t = TSF + frac * DT_SWITCHON
        expected = bubble_E2P(EB, R2, frac * R1, GAMMA)
        assert _p(t) == pytest.approx(expected, rel=1e-12)
        assert _p(t) < p_full


def test_ramp_window_closes_continuously():
    """At t = tSF + 1e-3 the ramp factor is exactly 1, so the in-window branch
    meets the unramped branch with no Pb jump."""
    p_full = bubble_E2P(EB, R2, R1, GAMMA)
    at_close = _p(TSF + DT_SWITCHON)
    just_after = _p(TSF + DT_SWITCHON * (1 + 1e-12))
    assert at_close == pytest.approx(p_full, rel=1e-12)
    assert just_after == pytest.approx(p_full, rel=1e-12)


def test_t_none_ablates_the_ramp():
    """The ablation contract the E8b/R3 harnesses rely on: forwarding t=None
    (or tSF=None) skips the ramp branch and changes nothing else."""
    p_full = bubble_E2P(EB, R2, R1, GAMMA)
    for t, tsf in ((None, TSF), (TSF + 1e-5, None), (None, None)):
        p = get_effective_bubble_pressure(
            current_phase='energy', Eb=EB, R2=R2, R1=R1, gamma=GAMMA, t=t, tSF=tsf)
        assert p == pytest.approx(p_full, rel=1e-15)


def test_ramp_active_early_in_window():
    """The load-bearing pin: early in the window the ramp must actually fire.
    If someone deletes the branch, the pressure at t-tSF = 1e-4 Myr reverts to
    the unramped value and this assertion fails — by measurement that deletion
    stalls f1edge_hidens outright."""
    t = TSF + 1e-4
    ratio = _p(t) / bubble_E2P(EB, R2, R1, GAMMA)
    # ramped volume: R2^3 - (0.1*R1)^3 vs unramped R2^3 - R1^3
    expected_ratio = (R2**3 - R1**3) / (R2**3 - (0.1 * R1) ** 3)
    assert ratio == pytest.approx(expected_ratio, rel=1e-9)
    assert ratio < 0.5  # at (R1/R2)^3 = 0.673 the ramp is a >2x pressure effect
    assert np.isfinite(_p(t))
