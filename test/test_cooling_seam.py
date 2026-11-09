"""The bubble-structure equation and the loss integrals read one cooling function (docs/dev/cooling-tables/SEAM_PLAN.md).

* get_dudt's non-CIE branch equals the loss integrals' reading of the cube (log10 cool and log10 heat interpolated
  separately, bubble_luminosity.py L2/L3);
* L1's rate between the cube's top (10^5.5 K) and the CIE grid's next node equals get_dudt's bridge there.
"""
import os

import numpy as np
import pytest
import scipy.interpolate

import trinity._functions.unit_conversions as cvt
import trinity.bubble_structure.bubble_luminosity as BL
import trinity.cooling.net_coolingcurve as ncc
import trinity.cooling.non_CIE.read_cloudy as non_CIE
from trinity._input.read_param import read_param

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def params():
    """params + cooling tables wired up as the runner does (main.py, read_cloudy.get_coolingStructure)."""
    p = read_param(os.path.join(REPO, "param", "simple_cluster.param"))
    p["t_now"].value = 0.1
    p["Qi"].value = 1e50 * 3.15576e13  # 1e50 ionising photons s^-1 (a ~1e4 Msun cluster), in Myr^-1
    logT, logL = np.loadtxt(p["path_cooling_CIE"].value, unpack=True)
    p["cStruc_cooling_CIE_logT"].value = logT
    p["cStruc_cooling_CIE_logLambda"].value = logL
    p["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logL, kind="linear"
    )
    cool, heat, net = non_CIE.get_coolingStructure(p)
    p["cStruc_cooling_nonCIE"].value = cool
    p["cStruc_heating_nonCIE"].value = heat
    p["cStruc_net_nonCIE_interpolation"].value = net
    return p


def _present(p, logT_lo, logT_hi, k=200, seed=1):
    """k random (log n, log T, log Phi) [cgs] inside the cube where both cubes interpolate to finite values."""
    cool, heat = p["cStruc_cooling_nonCIE"].value, p["cStruc_heating_nonCIE"].value
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < k:
        x = np.array(
            [
                rng.uniform(cool.ndens[0], cool.ndens[-1]),
                rng.uniform(logT_lo, logT_hi),
                rng.uniform(cool.phi[0], cool.phi[-1]),
            ]
        )
        if np.isfinite(cool.interp(x)[0]) and np.isfinite(heat.interp(x)[0]):
            out.append(x)
    return np.array(out)


def test_ode_reads_the_cube_as_the_loss_integrals_do(params):
    cool, heat = params["cStruc_cooling_nonCIE"].value, params["cStruc_heating_nonCIE"].value
    top, _ = ncc._noncie_cutoffs(cool)
    for x in _present(params, cool.temp[0], top):
        n_au, T, phi_au = 10 ** x[0] * cvt.ndens_cgs2au, 10 ** x[1], 10 ** x[2] * cvt.phi_cgs2au
        ode = ncc.get_dudt(0.1, n_au, T, phi_au, params)
        xe = [np.log10(n_au / cvt.ndens_cgs2au), np.log10(T), np.log10(phi_au / cvt.phi_cgs2au)]
        c, h = 10 ** cool.interp(xe)[0], 10 ** heat.interp(xe)[0]
        np.testing.assert_allclose(
            -ode, (c - h) * cvt.dudt_cgs2au, rtol=0, atol=1e-12 * (c + h) * cvt.dudt_cgs2au
        )


def test_L1_band_rate_is_the_ode_bridge(params):
    lo, _ = ncc._noncie_cutoffs(params["cStruc_cooling_nonCIE"].value)
    hi = ncc._cie_tcutoff(params["cStruc_cooling_CIE_logT"].value)
    assert 0 < hi - lo < 0.1  # the default curve 3: 5.5 -> 5.5481
    pts = _present(params, lo, lo)  # (n, Phi) where the cube's top row is present
    logT = np.linspace(lo, hi, len(pts) + 2)[1:-1]  # strictly inside the bridge
    n_au = 10 ** pts[:, 0] * cvt.ndens_cgs2au
    r = np.sqrt(
        params["Qi"].value / (4 * np.pi * 10 ** pts[:, 2] * cvt.phi_cgs2au)
    )  # Phi = Qi / (4 pi r^2)
    band = BL._seam_band_rate(params, n_au, 10 ** logT, r)
    phi_au = params["Qi"].value / (4 * np.pi * r ** 2)
    ode = np.array(
        [
            ncc.get_dudt(0.1, float(n), float(T), float(f), params)
            for n, T, f in zip(n_au, 10 ** logT, phi_au)
        ]
    )
    scale = np.abs(band) + np.abs(ode) + 1e-300
    assert np.max(np.abs(band + ode) / scale) < 1e-12


def test_bridge_endpoints(params):
    """At the cube's top the band rate is the cube's own rate; at the CIE node it is chi_e n^2 Lambda_CIE."""
    lo, _ = ncc._noncie_cutoffs(params["cStruc_cooling_nonCIE"].value)
    hi = ncc._cie_tcutoff(params["cStruc_cooling_CIE_logT"].value)
    x = _present(params, lo, lo, k=5)
    n_au = 10 ** x[:, 0] * cvt.ndens_cgs2au
    r = np.sqrt(params["Qi"].value / (4 * np.pi * 10 ** x[:, 2] * cvt.phi_cgs2au))
    phi_au = params["Qi"].value / (4 * np.pi * r ** 2)
    at_lo = BL._seam_band_rate(params, n_au, np.full(5, 10 ** lo), r)
    ode_lo = np.array(
        [
            -ncc.get_dudt(0.1, float(n), float(10 ** lo), float(f), params)
            for n, f in zip(n_au, phi_au)
        ]
    )
    np.testing.assert_allclose(at_lo, ode_lo, rtol=1e-12)
    at_hi = BL._seam_band_rate(params, n_au, np.full(5, 10 ** hi), r)
    lam = 10 ** params["cStruc_cooling_CIE_interpolation"].value(hi)
    np.testing.assert_allclose(
        at_hi,
        params["chi_e"].value * (n_au / cvt.ndens_cgs2au) ** 2 * lam * cvt.dudt_cgs2au,
        rtol=1e-12,
    )


def test_missing_cube_cell_raises(params):
    """A bridge point whose (n, Phi) falls in a missing (NaN) cell of the cube's top row is a solver error, not a NaN L1."""
    cool = params["cStruc_cooling_nonCIE"].value
    lo, _ = ncc._noncie_cutoffs(cool)
    rng = np.random.default_rng(2)
    while True:
        x = np.array(
            [rng.uniform(cool.ndens[0], cool.ndens[-1]), lo, rng.uniform(cool.phi[0], cool.phi[-1])]
        )
        if not np.isfinite(cool.interp(x)[0]):
            break
    n_au = np.array([10 ** x[0] * cvt.ndens_cgs2au])
    r = np.sqrt(params["Qi"].value / (4 * np.pi * 10 ** x[2] * cvt.phi_cgs2au)) * np.ones(1)
    with pytest.raises(BL.BubbleSolverError):
        BL._seam_band_rate(params, n_au, np.array([10 ** (lo + 0.01)]), r)
