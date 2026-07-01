"""
Tests for the catastrophic-cooling collapse guard (large-cloud `Eb=nan` fix).

Pins the robustness fix for `docs/dev/failed-large-clouds`: under catastrophic
cooling a massive/dense cloud's bubble energy `Eb` collapses, the wind shock
`R1 -> R2`, and the shell volume `R2**3 - R1**3` underflows to 0 in float64.
The old `bubble_E2P` divided by that zero (-> inf / ZeroDivisionError -> Eb=nan,
crashing the run). The guard floors the volume so the divide stays finite, and
the energy phases stop cleanly with `SimulationEndCode.ENERGY_COLLAPSED`.

The guard is *bit-identical* on every physical bubble (shell volume > 0).
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
    # A physical R2 > 0 with NaN Eb must STILL raise (real failure, not fabricated)
    with pytest.raises(ValueError):
        get_bubbleParams.solve_R1(10.0, np.nan, 1.0, 1.0)


def test_energy_collapsed_endcode_is_inspection_required():
    # New end code used by the energy phases when Eb collapses; lives in the
    # 50-59 "inspection required (completed, warrants a human look)" band.
    code = SimulationEndCode.ENERGY_COLLAPSED
    assert code.code == 51
    assert 50 <= code.code <= 59
    assert code.value[1] == "energy_collapsed"


# --- Energy-collapse ROUTING (docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md) ---
# When an energy-driven bubble collapses, a FINITE Eb<=0 is an energy->momentum
# transition (Pb has floored to ~P_ram) and must route to the momentum phase, NOT
# dead-stop. Only a NON-FINITE Eb is unrecoverable and keeps ENERGY_COLLAPSED.

def test_classify_energy_collapse_routes_finite_collapse_to_momentum():
    from trinity.phase1b_energy_implicit.run_energy_implicit_phase import classify_energy_collapse
    assert classify_energy_collapse(0.0) == 'momentum'
    assert classify_energy_collapse(-9.143e8) == 'momentum'   # the fail_repro collapse value


def test_classify_energy_collapse_stops_only_on_nonfinite():
    from trinity.phase1b_energy_implicit.run_energy_implicit_phase import classify_energy_collapse
    assert classify_energy_collapse(np.nan) == 'stop'
    assert classify_energy_collapse(np.inf) == 'stop'
    assert classify_energy_collapse(-np.inf) == 'stop'


def test_classify_energy_collapse_none_while_energy_driven():
    from trinity.phase1b_energy_implicit.run_energy_implicit_phase import classify_energy_collapse
    # A healthy, finite, positive Eb must NOT route or stop -> byte-identical (G0).
    assert classify_energy_collapse(1.0e8) is None
    assert classify_energy_collapse(1e-6) is None
