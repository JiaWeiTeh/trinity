"""B1 of docs/dev/turnaround-stall/PLAN.md (ruled 2026-11-20), re-planned on main ef0ec37a (harness/apply_b1_v2.py).

Since B1 a turnaround in phase 1b keeps Eb and 1b goes on as a one-zone bubble: its loss is Lloss/Lgain of the last
beta/delta solve times the live gain, and T0 is held. Where that fraction cannot be had, or the bubble is spent, the
2026-10-01 rule stays (Eb to the floor). Unit level only: the fraction the turnaround freezes (onezone_theta, new), and
how the one-zone mode holds T0 on top of S1 (delta 0). The runner's wiring is checked end to end by PLAN.md's G1 gates.
"""
import math

import pytest

import trinity._functions.unit_conversions as cvt
from trinity.phase1b_energy_implicit.get_betadelta import delta2dTdt_pure
from trinity.phase1b_energy_implicit.run_energy_implicit_phase import DT_DELTA_FRAC, cap_dt_by_delta, onezone_theta

# PLAN.md §4's gate run `ta` (v6 1e4_sfe050_n1e3, P/k 1e6) turns around in 1b at 0.6685 Myr with theta 0.112
T_TURN = 0.6685                      # Myr
THETA = 0.112
LGAIN = 5e37 * cvt.L_cgs2au          # Lmech, order of magnitude for its M_* = 5e3 Msun [Msun pc^2 Myr^-3]
EB = 5e50 * cvt.E_cgs2au             # ~ (5/11) Lmech t at the turnaround, a live bubble [Msun pc^2 Myr^-2]


def test_the_last_solves_fraction_is_frozen():
    assert onezone_theta(THETA * LGAIN, LGAIN, EB) == pytest.approx(THETA, rel=1e-12)
    assert onezone_theta(0.0, LGAIN, EB) == 0.0


def test_no_fraction_keeps_the_old_rule():
    """No luminosity stored (None, or NaN as at the 1b hand-off), no gain, or a fraction outside [0, 1)."""
    for Lloss, Lgain in ((None, LGAIN), (THETA * LGAIN, None), (math.nan, LGAIN), (THETA * LGAIN, math.nan),
                         (THETA * LGAIN, 0.0), (THETA * LGAIN, -LGAIN), (LGAIN, LGAIN), (2.0 * LGAIN, LGAIN),
                         (-0.1 * LGAIN, LGAIN)):
        assert onezone_theta(Lloss, Lgain, EB) is None


def test_a_spent_bubble_keeps_the_old_rule():
    """Eb crossed zero (or went non-finite) inside the segment before v2 did: no one-zone bubble to continue."""
    for Eb in (0.0, -EB, math.nan, math.inf):
        assert onezone_theta(THETA * LGAIN, LGAIN, Eb) is None


def test_delta_zero_holds_T0_and_lifts_the_delta_cap():
    """S1 integrates T0 live, dT0/dt = delta T0 / t, and caps a segment at DT_DELTA_FRAC t / |delta|. The one-zone mode
    passes delta 0: T0 does not move and the cap leaves dt alone (the last solve's delta would still cap it)."""
    T0, dt, delta_last = 3.0e6, 0.05, -0.5          # K; Myr (DT_SEGMENT_MAX); a late-1b delta
    assert delta2dTdt_pure(T_TURN, T0, 0.0) == 0.0
    assert cap_dt_by_delta(dt, 0.0, T_TURN) == dt
    capped = cap_dt_by_delta(dt, delta_last, T_TURN)
    assert capped == pytest.approx(DT_DELTA_FRAC * T_TURN / abs(delta_last)) and capped < dt
