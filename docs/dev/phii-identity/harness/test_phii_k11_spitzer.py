#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3a's photoionised pressure must reproduce classical D-type (Spitzer/HI) expansion.

§3 of `docs/dev/phii-identity/PLAN.md` placed a two-sided limiting-case obligation on the
C3 family: *wind-only -> Weaver-like, photo-only -> Spitzer-like*. Batch 5 stage 3
discharged the wind-only half. This pins the photo-only half (Batch 8).

Why it matters beyond bookkeeping: `get_phii_c3c`'s docstring records a KNOWN OPEN
BEHAVIOUR — the momentum phase comes out photoionisation-dominated in every configuration
measured — and asserts that this is "NOT an O(1) normalisation error". These tests are the
external anchor for that assertion. If C3a's magnitude were wrong, the classical limit is
where it would show, and `test_mis_normalisation_is_caught` demonstrates the check has the
power to see exactly that class of error.

The physics. Thin swept-up shell, uniform medium, no wind:

    d/dt (M R') = 4 pi R^2 P ,  M = (4/3) pi R^3 rho_0 ,  P = P_C3a(R) ~ R^-3/2

Self-similar with R = A t^(4/7); matching amplitudes gives
A = [(49/12) c_i^2 R_St^(3/2)]^(2/7), which is identically the large-t limit of
Hosokawa & Inutsuka (2006), R = R_St [1 + (7/4) sqrt(4/3) c_i t / R_St]^(4/7).
Spitzer (1978)'s ram-balance closure gives the same 4/7 index with amplitude lower by
(4/3)^(2/7) = 1.0855, so the two classical results bracket the answer.

NOTE the initial condition. HI's closed form has v = sqrt(4/3) c_i at t = 0 (differentiate
it there), NOT v = 0. Integrating from rest and comparing pointwise measures the startup
transient, not the amplitude — that mistake is what failed gate G8.4 as first registered
(-9.5% at R/R_St = 2, decaying to -0.04% by R/R_St = 50). Comparing like with like is the
amplitude test, and it is exact.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import trinity._functions.unit_conversions as cvt
from trinity._input.read_param import read_param
from trinity.bubble_structure.get_bubbleParams import get_phii_c3c

REPO = Path(__file__).resolve().parents[1]

INDEX_DTYPE = 4.0 / 7.0
HI_OVER_SPITZER = (4.0 / 3.0) ** (2.0 / 7.0)

# Dense-cloud ambient and a real cluster's ionising output (the simple_cluster value).
N0_CGS = 1e3
QI_AU = 5.1227849481751455e64


class _Shell:
    # G22.7': the ONLY change to this file. The committed stub supplies no R_IF, so any
    # scheme reading the shell solve's front returns 0.0 and cannot be tested here at all.
    # In this fixture's own idealisation the shell is thin and the front sits AT the shell,
    # so R_IF = R2 is the fixture's own geometry, not a concession to the scheme.
    def __init__(self, f_abs=1.0, params=None):
        self.shell_fAbsorbedIon = f_abs
        self._p = params

    @property
    def R_IF(self):
        return float(self._p["R2"].value)


@pytest.fixture(scope="module")
def setup():
    """Params with the wind off (Pb=0 => C3c always on its driving branch)."""
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    p["Pb"].value = 0.0
    p["nCore"].value = N0_CGS * cvt.ndens_cgs2au
    p["Qi"].value = QI_AU
    n0 = p["nCore"].value
    denom = 4.0 * np.pi * p["chi_e_shell"].value * p["caseB_alpha"].value * n0**2
    R_St = (3.0 * QI_AU / denom) ** (1.0 / 3.0)
    c_i = np.sqrt(p["k_B"].value * p["TShell_ion"].value / p["mu_ion_shell"].value)
    return p, _Shell(params=p), n0, n0 * p["mu_convert"].value, R_St, c_i


def _P(params, shell, R, demote=False):
    params["R2"].value = R
    P = get_phii_c3c(params, shell)
    if demote:  # drop the particles-per-H-nucleus factor: 2.2x low
        P /= params["mu_convert"].value / params["mu_ion_shell"].value
    return P


def _expand(params, shell, R_St, rho0, v0, r_stop, demote=False):
    def rhs(_t, y):
        R, v = y
        return [v, 3.0 * (_P(params, shell, R, demote=demote) / rho0 - v * v) / R]

    def stop(_t, y):
        return y[0] - r_stop * R_St

    stop.terminal, stop.direction = True, 1
    c_i = np.sqrt(params["k_B"].value * params["TShell_ion"].value / params["mu_ion_shell"].value)
    t_max = 400.0 * R_St / c_i
    sol = solve_ivp(
        rhs, (0.0, t_max), [R_St, v0], events=stop, rtol=1e-10, atol=1e-13, max_step=t_max / 4000.0
    )
    assert sol.success, sol.message
    ok = sol.t > 0
    return sol.t[ok], sol.y[0][ok]


def _hi(R_St, c_i, t):
    return R_St * (1.0 + (7.0 / 4.0) * np.sqrt(4.0 / 3.0) * c_i * t / R_St) ** INDEX_DTYPE


def test_stromgren_anchor(setup):
    """G8.1 — at R2 = R_St the cavity density the helper inverts IS the ambient density."""
    params, _, n0, _, R_St, _ = setup
    denom = 4.0 * np.pi * params["chi_e_shell"].value * params["caseB_alpha"].value * R_St**3
    assert np.sqrt(3.0 * QI_AU / denom) == pytest.approx(n0, rel=1e-12)


def test_pressure_normalisation_is_n_total_kT(setup):
    """G8.2 — P_C3a(R_St) = n_tot k T = rho_0 c_i^2, i.e. Spitzer's 2 n k T with He.

    The shipped `mu_convert/mu_ion_shell` prefactor must BE the particle count per hydrogen
    nucleus in the ionised gas: H+ + e + He+ + e = 2 + x_He (1 + Z_He_shell) = 2.2.
    """
    params, shell, n0, rho0, R_St, c_i = setup
    n_tot = 2.0 + params["x_He"].value * (1.0 + params["Z_He_shell"].value)
    assert params["mu_convert"].value / params["mu_ion_shell"].value == pytest.approx(
        n_tot, rel=1e-12
    )

    P = _P(params, shell, R_St)
    assert P == pytest.approx(
        n_tot * n0 * params["k_B"].value * params["TShell_ion"].value, rel=1e-12
    )
    assert P == pytest.approx(rho0 * c_i**2, rel=1e-12)


def test_reproduces_hosokawa_inutsuka(setup):
    """G8.4' — driven by the shipped helper, the momentum equation IS the HI solution.

    Compared from HI's own t=0 state, so a residual here is a pressure error and nothing
    else. It is exact to integrator tolerance over more than a decade in radius.
    """
    params, shell, _, rho0, R_St, c_i = setup
    t, R = _expand(params, shell, R_St, rho0, np.sqrt(4.0 / 3.0) * c_i, r_stop=20.0)
    win = (R / R_St >= 2.0) & (R / R_St <= 20.0)
    assert win.sum() > 50
    assert np.max(np.abs(R[win] / _hi(R_St, c_i, t[win]) - 1.0)) < 1e-4


def test_sits_above_spitzer_by_the_analytic_factor(setup):
    """The two classical closures bracket the answer, and we land on the momentum one.

    Spitzer's ram-balance closure is lower by exactly (4/3)^(2/7) = 1.0855. Landing on HI
    rather than Spitzer is what makes this a test of the momentum equation, not a fit.
    """
    params, shell, _, rho0, R_St, c_i = setup
    t, R = _expand(params, shell, R_St, rho0, np.sqrt(4.0 / 3.0) * c_i, r_stop=20.0)
    i = int(np.argmin(np.abs(R / R_St - 10.0)))
    spitzer = R_St * (1.0 + (7.0 / 4.0) * c_i * t[i] / R_St) ** INDEX_DTYPE
    assert R[i] / spitzer == pytest.approx(HI_OVER_SPITZER, rel=0.02)


def test_expansion_index_is_four_sevenths(setup):
    """G8.3 — the self-similar index, measured from rest once the attractor is reached."""
    params, shell, _, rho0, R_St, c_i = setup
    t, R = _expand(params, shell, R_St, rho0, 0.0, r_stop=30.0)
    i = int(np.argmin(np.abs(R / R_St - 25.0)))
    lo, hi = max(i - 8, 1), min(i + 8, len(R) - 1)
    slope = np.polyfit(np.log(t[lo : hi + 1]), np.log(R[lo : hi + 1]), 1)[0]
    assert slope == pytest.approx(INDEX_DTYPE, rel=0.01)


def test_mis_normalisation_is_caught(setup):
    """G8.5 — the control that makes the four passes above evidence rather than decoration.

    Dropping the particles-per-nucleus factor is 2.2x low in pressure, which the self-similar
    scaling turns into (1/2.2)^(2/7) - 1 = -20.17% in radius. If this test ever passes, the
    checks above have stopped being able to see an O(1) normalisation error.
    """
    params, shell, _, rho0, R_St, c_i = setup
    t, R = _expand(params, shell, R_St, rho0, np.sqrt(4.0 / 3.0) * c_i, r_stop=20.0, demote=True)
    i = int(np.argmin(np.abs(R / R_St - 10.0)))
    dev = R[i] / _hi(R_St, c_i, t[i]) - 1.0
    assert dev < -0.05, "a 2.2x pressure error must be visible"
    assert dev == pytest.approx((1.0 / 2.2) ** (2.0 / 7.0) - 1.0, abs=0.01)
