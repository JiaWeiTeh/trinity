"""
Tests for the catastrophic-cooling collapse guard (large-cloud `Eb=nan` fix).

Pins the robustness fix for `docs/dev/failed-large-clouds`: under catastrophic
cooling a massive/dense cloud's bubble energy `Eb` collapses, the wind shock
`R1 -> R2`, and the shell volume `R2**3 - R1**3` underflows to 0 in float64.
The old `bubble_E2P` divided by that zero (-> inf / ZeroDivisionError -> Eb=nan,
crashing the run). The guard floors the volume so the divide stays finite, and
the energy phases detect the collapse (Eb<=0) and hand off:

- a FINITE Eb<=0 is an energy->momentum transition (Pb has floored to ~P_ram) and
  ROUTES to the momentum phase via `classify_energy_collapse` (phase 1b);
- a NON-FINITE Eb is unrecoverable and ends on `SimulationEndCode.ENERGY_COLLAPSED`.

The guard is *bit-identical* on every physical bubble (shell volume > 0).
See docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md.
"""

from __future__ import annotations

import numpy as np
import pytest

import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
import trinity._functions.unit_conversions as cvt
from trinity._output.simulation_end import SimulationEndCode

GAMMA = 5.0 / 3.0


def _old_bubble_E2P(Eb, r2, r1, gamma):
    """The pre-fix formula (verbatim), for the bit-identity check."""
    r1 = r1 * cvt.pc2cm
    r2 = r2 * cvt.pc2cm
    Eb = Eb * cvt.E_au2cgs
    r2 += 1e-10
    Pb = (gamma - 1) * Eb / (r2**3 - r1**3) / (4 * np.pi / 3)
    return Pb * cvt.Pb_cgs2au


@pytest.mark.parametrize("Eb,R2,R1", [
    (1.0e9, 7.0475, 3.1788),   # healthy energy-driven bubble
    (1.0e8, 7.0475, 7.0231),   # cooling, R1 closing on R2 (rel vol ~1e-2)
    (1.0e2, 7.0475, 7.0470),   # deep in collapse but volume still > 0 (R1 < R2)
])
def test_bubble_E2P_bit_identical_when_volume_positive(Eb, R2, R1):
    # The guard only changes behaviour when the shell volume underflows to <= 0;
    # for every physical bubble (R1 < R2) it must be byte-for-byte the old value.
    assert get_bubbleParams.bubble_E2P(Eb, R2, R1, GAMMA) == _old_bubble_E2P(Eb, R2, R1, GAMMA)


def test_bubble_E2P_finite_when_shell_collapses():
    # R1 == R2 -> shell volume underflows to exactly 0. The old code divided by
    # zero (inf / ZeroDivisionError); the guard must return a finite value so the
    # phase loop can detect the collapse (Eb<=0) and stop cleanly.
    R2 = 7.047540
    Pb = get_bubbleParams.bubble_E2P(Eb=1e-4, r2=R2, r1=R2, gamma=GAMMA)
    assert np.isfinite(Pb)


def test_solve_R1_returns_zero_for_nonphysical_R2():
    # During the collapse the RK45 integrator can evaluate the RHS at a
    # non-physical trial state (R2 <= 0 or NaN). The old code let brentq hit
    # get_r1(0) = sqrt(<0) -> NaN and raised "function value at x=0 is NaN",
    # crashing the run in phase 1a (the real Helix point). solve_R1 now returns
    # 0.0 (no positive radius -> no wind shock), keeping the RHS finite so the
    # integrator can reject the bad step; the Eb<=0 check then stops cleanly.
    assert get_bubbleParams.solve_R1(-154.0, -4.4e31, 5e12, 3739.0) == 0.0
    assert get_bubbleParams.solve_R1(0.0, 1e5, 1.0, 1.0) == 0.0
    assert get_bubbleParams.solve_R1(np.nan, 1e5, 1.0, 1.0) == 0.0
    # Physical R2 + NaN Eb is pinned in
    # test_r1_bracket.py::test_failure_raises_instead_of_fabricating.


def test_energy_collapsed_endcode_is_inspection_required():
    # New end code used by the energy phases when Eb collapses; lives in the
    # 50-59 "inspection required (completed, warrants a human look)" band.
    code = SimulationEndCode.ENERGY_COLLAPSED
    assert code.code == 51
    assert 50 <= code.code <= 59
    assert code.value[1] == "energy_collapsed"


# --- Energy-collapse ROUTING (docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md) ---
# A spent bubble (Pb <= phaseSwitch_PbPram * P_ram(R2), or a FINITE Eb <= 0) is an
# energy->momentum transition and must route to the momentum phase, NOT dead-stop.
# Only a NON-FINITE Eb is unrecoverable and keeps ENERGY_COLLAPSED. A strong wind:
# 3e8 code units (1.8e38 erg/s) at 2e3 pc/Myr, R2 = 20 pc; a young weak wind: 1e36
# erg/s, its Weaver bubble at 1e-3 Myr (Eb = (5/11) L t, R2 = 0.104 pc).
_STRONG = dict(R2=20.0, Lmech_total=3e8, v_mech_total=2e3, gamma=5 / 3, PbPram=1.01)
_L_WEAK = 1e36 / (cvt.E_au2cgs / cvt.Myr2s)   # erg/s -> Msun pc^2 / Myr^3


def _classify(Eb, **kw):
    from trinity.phase1b_energy_implicit.run_energy_implicit_phase import classify_energy_collapse
    return classify_energy_collapse(Eb, **{**_STRONG, **kw})


def test_classify_energy_collapse_routes_finite_collapse_to_momentum():
    assert _classify(0.0) == 'momentum'
    assert _classify(-9.143e8) == 'momentum'   # the fail_repro collapse value
    assert _classify(-1.0, Lmech_total=0.0, v_mech_total=0.0) == 'momentum'   # no wind: Eb <= 0 alone


def test_classify_energy_collapse_routes_a_spent_bubble_to_momentum():
    assert _classify(1e3) == 'momentum'   # Pb = 1.0002 P_ram(R2) for the strong wind


def test_classify_energy_collapse_stops_only_on_nonfinite():
    assert _classify(np.nan) == 'stop'
    assert _classify(np.inf) == 'stop'
    assert _classify(-np.inf) == 'stop'


def test_classify_energy_collapse_none_while_energy_driven():
    assert _classify(1.0e8) is None   # Pb = 33.5 P_ram(R2)
    # a young weak-wind bubble BELOW the old 1e3 floor still drives: Pb = 9.07 P_ram(R2)
    Eb_weak = 5 / 11 * _L_WEAK * 1e-3
    assert Eb_weak < 1e3
    assert _classify(Eb_weak, R2=0.104, Lmech_total=_L_WEAK) is None
