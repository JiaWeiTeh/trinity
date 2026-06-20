"""Runtime tests for the net_coolingcurve T-floor gate (audit finding #1).

``get_dudt`` floors a sub-table temperature up to the cooling file's minimum
tabulated T (``nonCIE_Tmin = min(cube.temp)``, log10 K) so it degrades to the
table edge instead of falling through to the ``raise`` below the table. This
replaced a hard-coded ``1e4`` floor that over-floored the entire valid
``[10**nonCIE_Tmin, 1e4)`` decade.

These pin the corrected behaviour:
  * below the table edge -> no raise, and clamps to the *table edge* (not 1e4);
  * inside the over-floored decade [edge, 1e4) -> evaluated at the real T
    (not floored up to 1e4);
  * at/above 1e4 (the only regime real runs reach) -> untouched.
"""
import os
import sys

import numpy as np
import pytest
import scipy.interpolate

sys.path.insert(0, os.getcwd())

import trinity._functions.unit_conversions as cvt
from trinity._input.read_param import read_param
import trinity.cooling.non_CIE.read_cloudy as non_CIE
import trinity.cooling.net_coolingcurve as ncc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def cooling_params():
    """params + non-CIE cube wired up exactly as the runner does (see main.py)."""
    params = read_param(os.path.join(REPO, "param", "simple_cluster.param"))
    params["t_now"].value = 0.1
    logT, logLambda = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    params["cStruc_cooling_CIE_logLambda"].value = logLambda
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logLambda, kind="linear"
    )
    cooling_nonCIE, heating_nonCIE, netcool = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cooling_nonCIE
    params["cStruc_heating_nonCIE"].value = heating_nonCIE
    params["cStruc_net_nonCIE_interpolation"].value = netcool
    _, nonCIE_Tmin = ncc._noncie_cutoffs(cooling_nonCIE)
    return params, nonCIE_Tmin


_NDENS_CGS = 1e2
_PHI_CGS = 1e10


def _dudt(params, T):
    # get_dudt mutates ndens/phi in place (/=), so pass fresh physical scalars.
    nd_au = _NDENS_CGS * cvt.ndens_cgs2au
    ph_au = _PHI_CGS * cvt.phi_cgs2au
    return ncc.get_dudt(0.1, nd_au, T, ph_au, params)


def _expected_noncie(params, T):
    """Reproduce get_dudt's non-CIE branch arithmetic EXACTLY (same ops, same
    order, incl. the in-place /= round-trip) so equality is bit-for-bit."""
    nd_eff = (_NDENS_CGS * cvt.ndens_cgs2au) / cvt.ndens_cgs2au
    ph_eff = (_PHI_CGS * cvt.phi_cgs2au) / cvt.phi_cgs2au
    netcool = params["cStruc_net_nonCIE_interpolation"].value
    dudt = netcool([np.log10(nd_eff), np.log10(T), np.log10(ph_eff)])[0]
    return -1 * dudt * cvt.dudt_cgs2au


def test_below_table_does_not_raise_and_clamps_to_edge(cooling_params):
    """T below the tabulated min must not raise, and must clamp to the table
    edge (10**nonCIE_Tmin) -- i.e. give the same dudt as evaluating exactly at
    the edge, NOT the old 1e4 floor."""
    params, nonCIE_Tmin = cooling_params
    T_edge = 10 ** nonCIE_Tmin
    deep = _dudt(params, 1000.0)          # well below the 3162 K edge
    at_edge = _dudt(params, T_edge)        # exactly at the edge
    assert np.isfinite(deep)
    assert deep == at_edge                 # clamped to the edge, bit-for-bit
    # and it is NOT the old over-floor-to-1e4 value
    assert deep != _dudt(params, 1e4)


def test_over_floored_decade_uses_real_temperature(cooling_params):
    """A temperature inside [edge, 1e4) is evaluated at its real value, not
    floored up to 1e4 as the old hard-coded gate did."""
    params, nonCIE_Tmin = cooling_params
    T_mid = 5000.0                         # in [3162, 10000)
    assert 10 ** nonCIE_Tmin <= T_mid < 1e4
    direct = _dudt(params, T_mid)
    assert direct != _dudt(params, 1e4)                  # the over-floor is gone
    assert direct != _dudt(params, 10 ** nonCIE_Tmin)    # nor pinned to the edge
    # it is evaluated at the real T: equals the non-CIE interpolation at T_mid
    assert direct == _expected_noncie(params, T_mid)


def test_real_run_regime_untouched(cooling_params):
    """At/above 1e4 (every value any real run reaches; measured min T = 30000)
    the floor is inert: dudt equals the direct non-CIE interpolation at T."""
    params, _ = cooling_params
    for T in (1e4, 3e4, 5e4):
        assert _dudt(params, T) == _expected_noncie(params, T)
