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


def test_energy_collapsed_endcode_is_inspection_required():
    # New end code used by the energy phases when Eb collapses; lives in the
    # 50-59 "inspection required (completed, warrants a human look)" band.
    code = SimulationEndCode.ENERGY_COLLAPSED
    assert code.code == 51
    assert 50 <= code.code <= 59
    assert code.value[1] == "energy_collapsed"
