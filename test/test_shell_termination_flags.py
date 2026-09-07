#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The ionised-region integration terminates at ONE radius; its state flags must
describe THAT radius.

The loop integrates outward in slices and stops at the first index where either the
shell's mass is accounted for OR the ionising photons are exhausted:

    idx = nonzero(massCondition | phiCondition)[0]

Until 2026-09-04 the two flags were read as `any(massCondition)` / `any(phiCondition)`
over the WHOLE slice -- including radii past `idx`, which are not part of the shell.
Two consequences, one cosmetic and one not:

  * mass fires at idx, phi crosses later  -> `is_phiDepleted` True on rows whose
    escape fraction at the front was 0.67, contradicting the flag's own docstring;
  * phi fires at idx, mass crosses later  -> `is_allMassSwept` True, hence
    `has_neutral = is_phiDepleted and not is_allMassSwept` False, and the neutral
    region beyond the front is never integrated. That changes rShell, the shell
    thickness, nMax, tau_kappa_IR, the gravity integral -- and the trajectory.

These tests pin the contract rather than the implementation: whatever the loop does,
the flags it returns must be mutually consistent, consistent with the escape
fraction, and consistent with the profile it actually built.

docs/dev/phii-identity/PLAN.md W32, W33.
"""
from pathlib import Path

import pytest

from trinity._input.read_param import read_param
from trinity.shell_structure.shell_structure import (
    shell_structure_pure, _PHI_DEPLETED_EPS,
)

REPO = Path(__file__).resolve().parents[1]

# Each regime is a full state, because the branch taken is decided by the whole of
# (Qi, Li, Ln, Pb, shell_mass, R2) and not by any one of them. The first four are
# synthetic and all terminate on MASS. `depleted_front` is a real state lifted from
# a simple_cluster run (t = 1.15e-5 Myr, energy phase) and is the only one that
# terminates on PHI and grows a neutral region -- without it every assertion about
# has_neutral below is vacuous, which is exactly what this file looked like when it
# was first written. See test_a_depleting_regime_is_actually_covered.
REGIMES = [
    pytest.param(dict(Qi=1.0e63, Pb=1.0e-3, mShell=1.0e3), id="photon_poor_massive"),
    pytest.param(dict(Qi=1.0e65, Pb=1.0e-3, mShell=1.0e3), id="photon_rich_massive"),
    pytest.param(dict(Qi=1.0e65, Pb=1.0e-2, mShell=1.0e2), id="photon_rich_light"),
    pytest.param(dict(Qi=1.0e64, Pb=1.0e-4, mShell=1.0e4), id="dense_heavy"),
    pytest.param(dict(Qi=5.10769707029876e64, Pb=568523787.5685359,
                      mShell=0.017255627982153413, R2=0.010596828851730349,
                      Li=96433084786.95534, Ln=75447216220.5714,
                      bubble_mass=0.003878283636525603), id="depleted_front"),
]


def _shell(Qi, Pb, mShell, R2=1.0, Li=2.0e3, Ln=1.0e3, bubble_mass=0.0):
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    p["Qi"].value, p["Li"].value, p["Ln"].value = Qi, Li, Ln
    p["Pb"].value = Pb
    p["R2"].value = R2
    p["shell_mass"].value = mShell
    p["rShell"].value = R2
    p["bubble_mass"].value = bubble_mass
    p["isDissolved"].value = False
    return shell_structure_pure(p)


def test_a_depleting_regime_is_actually_covered():
    """Coverage guard. Every has_neutral assertion in this file passes trivially if no
    regime ever depletes phi -- which was true of the first four, so the file proved
    nothing about the case it was written for. If this fails, the regimes have drifted
    and the rest of the file has gone vacuous again."""
    assert any(_shell(**prm.values[0]).has_neutral for prm in REGIMES), \
        "no regime terminates on phi; every has_neutral assertion here is vacuous"


@pytest.mark.parametrize("kw", REGIMES)
def test_flags_are_mutually_consistent(kw):
    s = _shell(**kw)
    if s.isDissolved:
        # The dissolved branch breaks assertion 3 BY CONSTRUCTION: it forces
        # is_phiDepleted = True and shell_fAbsorbedIon = 0.0, i.e. f_esc = 1.
        # There is no shell there, so the flag is a placeholder, not a measurement.
        pytest.skip("dissolved shell takes the degenerate path")

    # 1. the loop exits only when one of the two conditions fires AT idx, so at
    #    least one flag must be set. Under `any(...)` this was trivially true; it is
    #    a real assertion now.
    assert s.is_allMassSwept or s.is_phiDepleted

    # 2. has_neutral is exactly the stated conjunction, nothing else
    assert s.has_neutral == (s.is_phiDepleted and not s.is_allMassSwept)

    # 3. the flag and the escape fraction share one threshold, so they cannot
    #    disagree: a row the solver calls depleted must report f_esc <= eps
    f_esc = 1.0 - s.shell_fAbsorbedIon
    assert s.is_phiDepleted == (f_esc <= _PHI_DEPLETED_EPS), (
        f"is_phiDepleted={s.is_phiDepleted} but f_esc={f_esc!r}")


@pytest.mark.parametrize("kw", REGIMES)
def test_flags_match_the_profile_actually_built(kw):
    """has_neutral must correspond to a profile that really has a neutral part."""
    s = _shell(**kw)
    if s.isDissolved:
        pytest.skip("dissolved shell takes the degenerate path")
    n = len(s.shell_r_arr)
    if s.has_neutral:
        assert s.shell_ion_idx < n - 1, "neutral region claimed but profile is all ionised"
    else:
        assert s.shell_ion_idx == n - 1, "no neutral region claimed but profile has a tail"


@pytest.mark.parametrize("kw", REGIMES)
def test_R_IF_never_exceeds_the_shell(kw):
    """R_IF is the last ionised radius in BOTH regimes, so it is capped at rShell.
    This is a censoring property, recorded here so it cannot be forgotten: a value
    of R_IF/rShell == 1 means 'at or beyond the edge', not 'exactly at the edge'."""
    s = _shell(**kw)
    if s.isDissolved or s.R_IF == 0.0:
        pytest.skip("no front")
    assert s.R_IF <= s.rShell * (1.0 + 1e-12)


def test_flags_are_read_at_the_termination_index_not_any():
    """Source guard: the exact regression this file exists for. `any(massCondition)`
    or `any(phiCondition)` in the ionised loop reintroduces W32 silently, because
    every invariant above still passes in the common case."""
    src = (REPO / "trinity/shell_structure/shell_structure.py").read_text()
    head = src.split("Neutral region integration")[0]   # ionised loop only
    assert "is_allMassSwept = any(massCondition)" not in head
    assert "is_phiDepleted = any(phiCondition)" not in head
    assert "is_allMassSwept = bool(massCondition[idx])" in head
    assert "is_phiDepleted = bool(phiCondition[idx])" in head


def test_one_threshold_for_depletion():
    """phiCondition and f_esc must use the same epsilon; a literal here is how they
    drift apart."""
    src = (REPO / "trinity/shell_structure/shell_structure.py").read_text()
    assert "phiShell_arr <= _PHI_DEPLETED_EPS" in src
    assert "_phi_end <= _PHI_DEPLETED_EPS" in src


def test_neutral_loop_does_not_clobber_the_ionised_flag():
    """Source guard: the neutral loop must terminate on its OWN variable. When it
    reused `is_allMassSwept`, every row that grew a neutral region returned
    (phi=True, mass=True, neutral=True) -- so `has_neutral == is_phiDepleted and not
    is_allMassSwept` was false on exactly the rows this file cares about, and the
    stored column no longer said where the IONISED solve stopped."""
    src = (REPO / "trinity/shell_structure/shell_structure.py").read_text()
    tail = src.split("Neutral region integration")[1]
    assert "is_allMassSwept = any(massCondition)" not in tail
    assert "while not is_allMassSwept:" not in tail
    assert "neutral_massSwept = any(massCondition)" in tail
