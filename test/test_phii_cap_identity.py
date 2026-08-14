#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The Strömgren cap and the P_HII identity it manufactures.

`shell_structure.py` caps the Strömgren density at the shell's inner density,
`n_IF_Str = min(n_IF_Str, shell_n0)`. Because `shell_n0` is itself *defined* by
pressure balance against `Pb`, and `P_HII` converts a density back to a pressure
with the same three factors, a bound cap makes `P_HII` an exact algebraic
relabelling of `Pb` — it carries no information about Qi, f_esc or the ionized
volume. Five workstreams measured this independently; see
`docs/dev/phii-identity/README.md`.

These tests pin the mechanism so a future cap change announces itself:

  (1) `n_IF_Str_raw` is the pre-cap value and the cap only ever reduces it;
  (2) where the cap binds, P_HII == Pb to <=2 ULP (the identity);
  (3) where the cap is slack, P_HII tracks the Strömgren balance instead.

⚠️ READ THIS BEFORE TRUSTING A GREEN RUN (2026-08-14). The cap WAS deliberately
replaced — `c43a50e` (C3c) — and this file did not announce it, because (2) and
(3) reconstruct the pressure with the local `_P_HII` helper below instead of
calling production. Nothing in `trinity/` computes P_HII this way any more; the
six call sites go through `get_bubbleParams.get_phii_c3c`, which returns exactly
0.0 while the ionised gas is confined. So:

  - (1) still guards live behaviour: `shell_structure.py` still caps `n_IF_Str`,
    and the cap still feeds the shell ODE's boundary condition, which is why it
    still matters even though it no longer sets P_HII.
  - (2) and (3) are now a HISTORICAL record of the defect the phii-identity
    workstream was created for. They are green against an algebraic relationship
    that is still true of the formula, and say nothing about what the code does.

The live P_HII contract is `test/test_phii_c3c.py` — that is the file to change
if the regime switch changes. Do not read this one as coverage of production.
"""
from pathlib import Path

import pytest

from trinity._input.read_param import read_param
from trinity.shell_structure.shell_structure import shell_structure_pure

REPO = Path(__file__).resolve().parents[1]
ULP = 2.0**-52


# A real mid-run state, lifted from a simple_cluster implicit-phase snapshot
# (t = 0.052 Myr) so the shell integrator sees a regime it actually runs in.
# Internal (astro) units throughout — Pb ~ 1e5 here is NOT cgs.
_REAL = dict(Pb=526031.6245459458, R2=1.290377449656595, shell_mass=31156.832760124395,
             bubble_mass=29.799120862170817, rShell=1.292814105540319,
             Qi=5.1227849481751455e64, Li=96730880812.02837, Ln=75459503069.95033)


def _params(Pb=None, Qi=None):
    """A runnable shell state from a real snapshot, with Pb and Qi optionally dialled."""
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    for key, val in _REAL.items():
        p[key].value = val
    if Pb is not None:
        p["Pb"].value = Pb
    if Qi is not None:
        p["Qi"].value = Qi
    return p


def _P_HII(params, n_IF_Str):
    """The conversion the four phase runners used verbatim BEFORE C3c (`c43a50e`).

    No production path computes P_HII this way now — this is a local
    reconstruction kept so the historical identity stays checkable.
    """
    return ((params["mu_convert"].value / params["mu_ion_shell"].value)
            * n_IF_Str * params["k_B"].value * params["TShell_ion"].value)


@pytest.mark.parametrize("Qi", [None, 1.0e58, 1.0e54])
def test_raw_is_precap_and_cap_only_reduces(Qi):
    """The shadow diagnostic is the value before the min(), so it never sits below
    the stored one, and the stored one is exactly min(raw, shell_n0). Holds whether
    or not the cap binds (Qi=None is the as-run value, which binds)."""
    s = shell_structure_pure(_params(Qi=Qi))
    assert s.n_IF_Str_raw >= s.n_IF_Str
    assert s.n_IF_Str == min(s.n_IF_Str_raw, s.shell_n0)


def test_identity_holds_where_the_cap_binds():
    """Cap bound (raw > shell_n0) => P_HII is Pb relabelled, to <=2 ULP.

    This is the defect the phii-identity workstream exists for: at the as-run
    ionizing rate the cap binds, so Qi cannot influence P_HII at all.
    """
    p = _params()  # as-run Qi = 5.1e64
    s = shell_structure_pure(p)
    assert s.n_IF_Str_raw > s.shell_n0, "cap did not bind at the as-run Qi"
    assert s.n_IF_Str == s.shell_n0

    Pb = p["Pb"].value
    assert abs(_P_HII(p, s.n_IF_Str) - Pb) / Pb <= 2 * ULP


def test_phii_tracks_stroemgren_where_the_cap_is_slack():
    """Cap slack (raw < shell_n0) => P_HII follows the ionizing field, not Pb.

    Guards the other side: a change that made the cap bind everywhere would
    silently delete the only regime where P_HII is real physics. Two decades of
    Qi apart must give two clearly different pressures.
    """
    p_hi, p_lo = _params(Qi=1.0e58), _params(Qi=1.0e54)
    s_hi, s_lo = shell_structure_pure(p_hi), shell_structure_pure(p_lo)
    assert s_hi.n_IF_Str_raw < s_hi.shell_n0, "cap bound; pick a smaller Qi"
    assert s_hi.n_IF_Str == s_hi.n_IF_Str_raw

    Pb = p_hi["Pb"].value
    P_hi, P_lo = _P_HII(p_hi, s_hi.n_IF_Str), _P_HII(p_lo, s_lo.n_IF_Str)
    assert P_lo < P_hi < Pb  # tracks Qi, and stays below the confining pressure
