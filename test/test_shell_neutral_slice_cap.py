#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A thick warm neutral layer must be solved on the neutral layer's own scale.

The neutral march used to cap every slice at a tenth of an IONISED length,
(R2 + R_St - R_IF)/10. With no radiation force the ionised layer is uniform and
R_IF**3 = R2**3 + R_St**3, so for a small bubble that cap falls to ~R2/10 and a
neutral layer a few percent of R_St thick took millions of points (StarBench's
late-phase benchmark ran 6 h unfinished). Since 2026-10-21 the adaptive neutral
march's cap never falls below 1 % of r (_SHELL_NEU_LAYER_SCALE, _SHELL_NEU_R_FRAC).

The state is StarBench-like (Bisbas et al. 2015 late phase): n = 1e3 cm^-3,
R_St = 1 pc, R2 = 1e-3 pc, a 1000 K neutral layer, no dust. Both layers are then
uniform, so the shell's outer radius is exact:
    rShell**3 = R_IF**3 + 3 (M_sh - M_ion) / (4 pi rho_neu),
with the neutral density set by pressure balance at the front. The legacy cap
gives 2.0e6 points at 3x and 8.0e6 at 10x the ionised mass; the new cap 3.0e4 and 8.5e4.

docs/dev/phii-identity PLAN.md Step 2b' (row 65b).
"""
import math
from pathlib import Path

import pytest

from trinity._input.read_param import read_param
import trinity._functions.unit_conversions as cvt
from trinity.shell_structure.shell_structure import shell_structure_pure

REPO = Path(__file__).resolve().parents[1]
N0_CGS = 1e3                    # cm^-3
H_NU_ERG = 13.6 * 1.602176634e-12


@pytest.mark.parametrize("m_over_m_ion, tol", [(3.0, 1e-6), (10.0, 1e-6)])
def test_thick_neutral_layer_is_cheap_and_exact(m_over_m_ion, tol):
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    g = lambda k: float(p[k].value)  # noqa: E731
    p["dust_sigma"].value = 0.0
    p["TShell_neu"].value = 1000.0
    n0 = N0_CGS * cvt.ndens_cgs2au
    # n0 is the density the shell solve derives from Pb (mu_ion_shell/mu_convert * Pb/(k_B T_ion))
    Pb = n0 * g("k_B") * g("TShell_ion") * g("mu_convert") / g("mu_ion_shell")
    Qi = 4 * math.pi / 3 * g("chi_e_shell") * g("caseB_alpha") * n0**2 * 1.0**3   # R_St = 1 pc
    R2 = 1e-3
    # the uniform ionised layer between R2 and R_IF: (R2^3 + R_St^3) - R2^3 = R_St^3
    m_ion = g("mu_convert") * n0 * 4 * math.pi / 3 * 1.0**3
    p["Qi"].value, p["Ln"].value = Qi, 0.0
    # tiny: no radiation force, but Li > 0 keeps the ionised march on
    p["Li"].value = 1e-20 * Qi * H_NU_ERG * cvt.E_cgs2au
    p["Pb"].value, p["R2"].value, p["rShell"].value = Pb, R2, R2
    p["shell_mass"].value = m_over_m_ion * m_ion
    p["bubble_mass"].value = 0.0
    p["isDissolved"].value = False

    sp = shell_structure_pure(p)

    assert len(sp.shell_r_arr) < 1_000_000
    n_neu = float(sp.n_IF) * g("mu_atom") / g("mu_ion_shell") * g("TShell_ion") / g("TShell_neu")
    m_neu = m_over_m_ion * m_ion - float(sp.shell_mass_ion)
    r_exact = (float(sp.R_IF) ** 3 + 3 * m_neu / (4 * math.pi * g("mu_convert") * n_neu)) ** (1 / 3)
    assert abs(float(sp.rShell) / r_exact - 1) < tol
