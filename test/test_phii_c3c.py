#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3c: the photoionised pressure as a regime switch, not a relabelling of `Pb`.

`get_bubbleParams.get_phii_c3c` replaces the capped-Strömgren `P_HII` (which was an
exact algebraic relabelling of the confining pressure — see `test_phii_cap_identity.py`)
with a two-branch rule built on the cavity Strömgren density:

    P_C3a = (mu_convert/mu_ion_shell) * k_B * T_ion * sqrt(3 Qi_abs / (4 pi chi_e alpha_B R2^3))

    P_C3a <= P_conf :  confinement holds the ionised gas as a thin skin. It transmits the
                       confining pressure and contributes nothing of its own -> returns 0.0
    P_C3a >  P_conf :  confinement cannot hold it; it fills its own volume and drives at P_C3a

Returning exactly 0.0 on the confined branch is load-bearing: it is what makes every
existing `P_drive` expression come out right without editing any of them (energy/implicit
`max(Pb_eff, 0)`, transition `max(Pb, 0 + P_ram)`, momentum `0 + P_ram`). A future change
that returns a small non-zero value there silently alters all four phases.

These tests pin the contract that Batch 6 landed:

  (1) the confined branch returns exactly 0.0, and the driving branch returns P_C3a;
  (2) `Qi -> 0` drives `P_HII -> 0` (the decoupling property C3b failed structurally);
  (3) the value depends on Qi and R2 and NOT on `Pb`, except through the branch choice;
  (4) degenerate inputs return 0.0 rather than a nan/inf that would poison the ODE.

See `docs/dev/phii-identity/PLAN.md` §3c.
"""

from pathlib import Path

import numpy as np
import pytest

from trinity._input.read_param import read_param
from trinity.bubble_structure.get_bubbleParams import get_phii_c3c

REPO = Path(__file__).resolve().parents[1]

# Same real mid-run state used by test_phii_cap_identity.py (simple_cluster implicit
# phase, t = 0.052 Myr). Internal (astro) units throughout.
_REAL = dict(
    Pb=526031.6245459458,
    R2=1.290377449656595,
    shell_mass=31156.832760124395,
    bubble_mass=29.799120862170817,
    rShell=1.292814105540319,
    Qi=5.1227849481751455e64,
    Li=96730880812.02837,
    Ln=75459503069.95033,
)


class _Shell:
    """Minimal stand-in for ShellProperties — the helper reads one attribute."""

    def __init__(self, f_abs=1.0):
        self.shell_fAbsorbedIon = f_abs


def _params(**over):
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    for key, val in {**_REAL, **over}.items():
        p[key].value = val
    return p


def _p_c3a(params, f_abs=1.0):
    """P_C3a computed independently of the helper, from the same closed form."""
    n = np.sqrt(
        3.0
        * params["Qi"].value
        * f_abs
        / (
            4.0
            * np.pi
            * params["chi_e_shell"].value
            * params["caseB_alpha"].value
            * params["R2"].value ** 3
        )
    )
    return (
        (params["mu_convert"].value / params["mu_ion_shell"].value)
        * n
        * params["k_B"].value
        * params["TShell_ion"].value
    )


def test_confined_branch_returns_exactly_zero():
    """P_C3a <= Pb => the skin transmits and contributes nothing. Exactly 0.0, not small."""
    p = _params()
    p["Pb"].value = _p_c3a(p) * 10.0
    assert get_phii_c3c(p, _Shell()) == 0.0


def test_driving_branch_returns_p_c3a():
    """P_C3a > Pb => the ionised gas drives at its own pressure."""
    p = _params()
    expected = _p_c3a(p)
    p["Pb"].value = expected / 10.0
    assert get_phii_c3c(p, _Shell()) == pytest.approx(expected, rel=1e-12)


def test_switch_is_at_p_conf():
    """The branch flips as Pb crosses P_C3a, and only there."""
    p = _params()
    target = _p_c3a(p)
    p["Pb"].value = target * (1.0 + 1e-9)
    assert get_phii_c3c(p, _Shell()) == 0.0
    p["Pb"].value = target * (1.0 - 1e-9)
    assert get_phii_c3c(p, _Shell()) > 0.0


@pytest.mark.parametrize("Qi", [1.0e64, 1.0e60, 1.0e56])
def test_scales_with_sqrt_qi_and_is_independent_of_pb(Qi):
    """The driving value tracks sqrt(Qi) and carries no Pb dependence.

    This is the decoupling the workstream exists to obtain: under the old capped
    Strömgren density P_HII was Pb relabelled, so it could not respond to Qi at all.
    """
    p = _params(Qi=Qi)
    expected = _p_c3a(p)
    p["Pb"].value = expected / 3.0
    first = get_phii_c3c(p, _Shell())
    p["Pb"].value = expected / 300.0  # still driving; value must not move
    assert get_phii_c3c(p, _Shell()) == pytest.approx(first, rel=1e-12)
    assert first == pytest.approx(expected, rel=1e-12)


def test_qi_to_zero_gives_zero():
    """Switching the ionizing source off must switch the photoionised drive off.

    C3b (ambient density) failed exactly here — it had no Qi dependence at all.
    """
    p = _params(Qi=0.0)
    p["Pb"].value = 1.0e-30
    assert get_phii_c3c(p, _Shell()) == 0.0


def test_absorbed_fraction_scales_and_is_clamped():
    """f_abs enters as sqrt(f_abs); an out-of-range or non-float value falls back to 1.0."""
    p = _params()
    p["Pb"].value = 0.0
    assert get_phii_c3c(p, _Shell(0.25)) == pytest.approx(_p_c3a(p, 0.25), rel=1e-12)
    full = _p_c3a(p, 1.0)
    for bad in (-0.5, 1.5, None, "1.0"):
        assert get_phii_c3c(p, _Shell(bad)) == pytest.approx(full, rel=1e-12)


@pytest.mark.parametrize("bad", [dict(R2=0.0), dict(R2=-1.0), dict(Qi=-1.0)])
def test_degenerate_geometry_returns_zero_not_nan(bad):
    """Never hand the ODE a nan/inf: degenerate radius or rate returns 0.0."""
    p = _params(**bad)
    p["Pb"].value = 0.0
    out = get_phii_c3c(p, _Shell())
    assert out == 0.0 and np.isfinite(out)
