#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step (b) of the two-zone neutral layer: trinity/shell_structure/pdr.py against its sources.

docs/dev/phii-identity/weak-wind/neutral-shell/PDR_DEFAULT_PLAN.md, gate G-B1. The table is Kaufman et al. 1999's own numeric
grid (the PDR Toolbox's wk2006 tsweb.fits; pdr-table/PLAN.md there); the H/H2 column is Bialy & Sternberg 2016 eqs. 22 and 39.
"""
import math

import numpy as np
import pytest

from trinity.shell_structure import pdr

JUMP = (14 / 11) / (14 / 22) * 1e4          # (mu_atom/mu_ion_shell) T_ion at x_He 0.1, T_ion 1e4 K


def test_k99_text_value():
    # Kaufman et al. 1999: T_S ~ 400 K at n = 1e4 cm^-3, G0 = 10^3.5 (their sect. 4); their numeric grid gives 343 K
    assert pdr.t_surface_k99(10 ** 3.5, 1e4) == pytest.approx(400.0, rel=0.2)


def test_k99_grid_nodes_and_clamping():
    ln_n, ln_g, lt = pdr._k99()
    assert lt.shape == (25, 29) and ln_n[0] == 1.0 and ln_n[-1] == 7.0 and ln_g[0] == -0.5 and ln_g[-1] == 6.5
    for i in (0, 7, 24):
        for j in (0, 13, 28):
            assert math.log10(float(pdr.t_surface_k99(10 ** ln_g[j], 10 ** ln_n[i]))) == pytest.approx(lt[i, j], abs=1e-12)
    assert float(pdr.t_surface_k99(0.0, 1e4)) == float(pdr.t_surface_k99(10 ** -0.5, 1e4))    # no field: the lowest G0
    assert float(pdr.t_surface_k99(1e9, 1e9)) == float(pdr.t_surface_k99(10 ** 6.5, 1e7))     # beyond the grid: its corner


def test_t_pdr_is_the_warmest_root():
    for g0, n_ion in ((5000.0, 500.0), (10.0, 5.0), (1e5, 1e4), (0.0, 50.0), (300.0, 3e4)):
        t = pdr.t_pdr(g0, n_ion, JUMP)
        assert t == pytest.approx(float(pdr.t_surface_k99(g0, n_ion * JUMP / t)), rel=1e-8)
        lt = np.linspace(math.log10(t) + 0.01, 5.0, 400)                    # no warmer root
        assert np.all(lt - np.log10(pdr.t_surface_k99(g0, n_ion * JUMP / 10.0 ** lt)) > 0.0)


def test_bialy_sternberg_worked_example():
    # Bialy & Sternberg 2016: I_UV = 1, n = 1e3 cm^-3, Z' = 1 -> N_tran 2.3e18 cm^-2 (their alphaG 5.7e-2; eq. 22 gives 5.9e-2).
    # A beam of strength I_UV delivers F0/2 (Sternberg et al. 2014 sect. 2.1), so I_UV = 1 is an incident G0 of 1.7/2 = 0.85.
    assert pdr.n_tran_bs16(0.85, 1e3, 1.0) == pytest.approx(2.3e18, rel=0.06)


def test_bialy_sternberg_floor_and_trend():
    assert pdr.n_tran_bs16(0.0, 1e3, 1.0) == pdr.N_TRAN_MIN
    assert pdr.n_tran_bs16(1e4, 1e2, 1.0) > pdr.n_tran_bs16(1e2, 1e2, 1.0) > pdr.n_tran_bs16(1e2, 1e4, 1.0)


def test_g0_of_one_lsun_at_one_pc():
    # 1 L_sun of far-UV at 1 pc, no dust: 3.828e33 / (4 pi (3.0857e18)^2) / 1.6e-3 = 0.0200 Habing
    assert pdr.g0_habing(3.828e33, 3.0857e18, 0.0) == pytest.approx(0.0200, rel=2e-3)
    assert pdr.g0_habing(3.828e33, 3.0857e18, 1.0) == pytest.approx(0.0200 * math.exp(-1.0), rel=2e-3)
