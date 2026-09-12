#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The sky-partition totals close, and they are the ONLY place coverFraction enters
the shell solve.

Ruling 2026-09-12 (docs/dev/cover-fraction/PLAN.md, picture (A) after Harper-Clark &
Murray 2009): `coverFraction` = Cf is a vented *sky fraction*. A fraction Cf of
directions is closed by an intact shell; the rest is open and confines nothing. So
the shell ODE is solved along ONE COVERED RAY and carries no Cf -- the deleted
`f_cover` stub multiplied dtaudr only, which is the wrong operator (a picket-fence
screen absorbs Cf*(1-e^-tau), not 1-e^-(Cf*tau)) -- and the partition is applied
once, at budget assembly:

    total = Cf * (per-ray) + (1 - Cf) * (free escape through the holes)

Closure is exact given the per-ray LyC identity f_esc + f_gas + f_dust = 1 (W69),
so `test_per_ray_identity_is_the_premise` is not redundant: it pins the premise the
other assertions rest on. Closes paper/rosette/PLAN.md F-5 / P13, which asked for
these definitions to be stated once rather than described in prose.
"""
from pathlib import Path

import pytest

from trinity._input.read_param import read_param
from trinity.shell_structure.shell_structure import shell_structure_pure

REPO = Path(__file__).resolve().parents[1]

# The identity is numerical, not algebraic: f_gas and f_dust are trapezoid integrals
# over the ionised profile and f_esc is phi at the front, so they close to ~1e-7 on
# the archived rows (W69), not to machine epsilon. A forgotten Cf factor is O(0.1).
CLOSURE_TOL = 1e-5

REGIMES = [
    pytest.param(dict(Qi=1.0e63, Pb=1.0e-3, mShell=1.0e3), id="photon_poor_massive"),
    pytest.param(dict(Qi=1.0e65, Pb=1.0e-3, mShell=1.0e3), id="photon_rich_massive"),
    pytest.param(dict(Qi=1.0e65, Pb=1.0e-2, mShell=1.0e2), id="photon_rich_light"),
    # the one regime that depletes phi and grows a neutral rind (see
    # test_shell_termination_flags.py -- without it has_neutral is never exercised)
    pytest.param(dict(Qi=5.10769707029876e64, Pb=568523787.5685359,
                      mShell=0.017255627982153413, R2=0.010596828851730349,
                      Li=96433084786.95534, Ln=75447216220.5714,
                      bubble_mass=0.003878283636525603), id="depleted_front"),
]
CF_VALUES = [1.0, 0.99, 0.95, 0.77, 0.5]


def _shell(Qi, Pb, mShell, R2=1.0, Li=2.0e3, Ln=1.0e3, bubble_mass=0.0,
           coverFraction=1.0, isDissolved=False):
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    p["Qi"].value, p["Li"].value, p["Ln"].value = Qi, Li, Ln
    p["Pb"].value = Pb
    p["R2"].value = R2
    p["shell_mass"].value = mShell
    p["rShell"].value = R2
    p["bubble_mass"].value = bubble_mass
    p["isDissolved"].value = isDissolved
    p["coverFraction"].value = coverFraction
    return shell_structure_pure(p)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("cf", CF_VALUES)
def test_lyc_budget_closes(regime, cf):
    """Qi = Q_esc + Q_leak + Q_gas + Q_dust, as a fraction, for every Cf."""
    s = _shell(**regime, coverFraction=cf)
    total = (s.shell_fEscLyC_total
             + s.shell_fAbsorbedIonGas_total
             + s.shell_fAbsorbedIonDust_total)
    assert total == pytest.approx(1.0, abs=CLOSURE_TOL), (
        f"LyC budget does not close at Cf={cf}: esc={s.shell_fEscLyC_total} "
        f"gas={s.shell_fAbsorbedIonGas_total} dust={s.shell_fAbsorbedIonDust_total} "
        f"sum={total}")


@pytest.mark.parametrize("regime", REGIMES)
def test_per_ray_identity_is_the_premise(regime):
    """At Cf = 1 the closure IS the per-ray identity f_esc + f_gas + f_dust = 1.
    If this breaks, the totals are fine and the shell solver is not."""
    s = _shell(**regime, coverFraction=1.0)
    f_esc = 1.0 - s.shell_fAbsorbedIon
    assert (f_esc + s.shell_fAbsorbedIonGas_total
            + s.shell_fAbsorbedIonDust_total) == pytest.approx(1.0, abs=CLOSURE_TOL)


@pytest.mark.parametrize("regime", REGIMES)
def test_cf_one_reduces_to_the_per_ray_numbers(regime):
    """A sealed sky must leave every per-ray quantity untouched -- this is what makes
    Cf = 1 byte-identical to the pre-ruling code."""
    s = _shell(**regime, coverFraction=1.0)
    assert s.shell_fLeak == 0.0
    assert s.shell_fEscLyC_total == pytest.approx(1.0 - s.shell_fAbsorbedIon, rel=1e-12)
    assert s.shell_fAbsorbedIonGas_total == pytest.approx(s.shell_fAbsorbedIonGas, rel=1e-12)
    assert s.shell_fAbsorbedNeu_total == pytest.approx(s.shell_fAbsorbedNeu, rel=1e-12)


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("cf", CF_VALUES)
def test_totals_are_linear_in_cf_and_per_ray_values_are_not(regime, cf):
    """The partition is applied ONCE, at the end. Two things follow, and together they
    are the operational statement of picture (A): every absorbed total scales exactly
    as Cf, and no per-ray quantity moves with Cf at all."""
    ref = _shell(**regime, coverFraction=1.0)
    s = _shell(**regime, coverFraction=cf)

    assert s.shell_fLeak == pytest.approx(1.0 - cf, rel=1e-12)
    assert s.shell_fAbsorbedIonGas_total == pytest.approx(cf * ref.shell_fAbsorbedIonGas, rel=1e-9)
    assert s.shell_fAbsorbedIonDust_total == pytest.approx(cf * ref.shell_fAbsorbedIonDust_total, rel=1e-9)
    assert s.shell_fAbsorbedNeu_total == pytest.approx(cf * ref.shell_fAbsorbedNeu, rel=1e-9)

    # The shell solve itself must be Cf-blind: these feed the dynamics (P_ext gate,
    # P_HII, F_rad) and a Cf leaking into any of them would move the trajectory.
    for field in ("shell_fAbsorbedIon", "shell_fAbsorbedIonGas", "shell_fAbsorbedNeu",
                  "shell_fAbsorbedWeightedTotal", "shell_fIonisedDust", "rShell",
                  "shell_thickness", "R_IF", "n_IF", "shell_n0", "shell_nMax",
                  "shell_tauKappaRatio", "shell_mass_ion", "shell_mass_neutral"):
        a, b = getattr(ref, field), getattr(s, field)
        if a != a:          # NaN (shell_fIonisedDust on a dissolved shell)
            assert b != b, f"{field}: NaN at Cf=1 but {b} at Cf={cf}"
        else:
            assert a == b, f"{field} moved with Cf ({a} -> {b}): the shell solve is not Cf-blind"


def test_dissolved_shell_escapes_everything():
    """No absorber at all: every photon leaves, through holes or through nothing."""
    s = _shell(Qi=1.0e65, Pb=1.0e-3, mShell=1.0e3, coverFraction=0.8, isDissolved=True)
    assert s.isDissolved
    assert s.shell_fEscLyC_total == 1.0
    assert s.shell_fAbsorbedIonGas_total == 0.0
    assert s.shell_fAbsorbedIonDust_total == 0.0
    assert s.shell_fAbsorbedNeu_total == 0.0
    assert s.shell_fLeak == pytest.approx(0.2, rel=1e-12)
