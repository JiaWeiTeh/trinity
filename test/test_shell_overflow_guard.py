#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The ionised shell RHS caps nShell (`_NSHELL_MAX`) so the dn/dr ∝ +nShell**2
recombination pole in the DISCARDED post-front tail cannot overflow float64 (the
overflow is what drives LSODA's "t + h = t" warning flood). The cap sits ~55 orders
of magnitude above any physical shell density, so it must never change the used
region. These tests pin both halves of that contract:

  (1) inactive in the used regime -> the RHS is unchanged (bit-identical formula);
  (2) active only on the runaway tail -> derivatives stay finite where the unguarded
      nShell**2 would overflow to inf.
"""
from pathlib import Path

import numpy as np

from trinity._input.read_param import read_param
from trinity.shell_structure.get_shellODE import get_shellODE, _NSHELL_MAX

REPO = Path(__file__).resolve().parents[1]


def _params():
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    # runtime feedback values default to 0; set positive so the RHS is non-trivial
    p["Ln"].value, p["Li"].value, p["Qi"].value = 1.0e3, 2.0e3, 1.0e5
    return p


def test_guard_inactive_in_used_region():
    """A used-region density (~1e10 cm^-3 -> ~1e65 code units, far below the cap)
    is untouched by min(): the RHS equals the analytic +nShell**2 dphidr form."""
    p = _params()
    n, phi, tau, r = 1.0e65, 0.5, 0.3, 5.0
    assert n < _NSHELL_MAX  # the guard must not bite here

    dndr, dphidr, dtaudr = get_shellODE([n, phi, tau], r, 1.0, True, p)
    assert np.isfinite([dndr, dphidr, dtaudr]).all()

    chi_e = p["chi_e_shell"].value
    aB = p["caseB_alpha"].value
    Qi = p["Qi"].value
    sd = p["dust_sigma"].value
    expect_dphidr = -4 * np.pi * r**2 * chi_e * aB * n**2 / Qi - n * sd * phi
    assert np.isclose(dphidr, expect_dphidr, rtol=1e-12)


def test_guard_prevents_overflow():
    """On the runaway tail (nShell past the cap) the derivatives stay FINITE;
    without the cap nShell**2 alone would overflow float64 to inf."""
    p = _params()
    n_huge = 1.0e160  # nShell**2 = 1e320 > 1.8e308 -> inf, unguarded
    assert n_huge > _NSHELL_MAX

    out = get_shellODE([n_huge, 0.5, 0.3], 5.0, 1.0, True, p)
    assert np.isfinite(out).all()

    # sanity: confirm the unguarded square really does overflow (numpy -> inf)
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.float64(n_huge) ** 2)


def test_cap_far_above_physical_density():
    """Guard against future drift: the cap must stay an unphysical numerical rail,
    not creep down into the physical shell regime (front peaks ~1e65 code units)."""
    assert _NSHELL_MAX >= 1.0e100          # >> physical front (~1e65)
    assert _NSHELL_MAX < 1.34e154          # < sqrt(float64 max): nShell**2 stays finite
