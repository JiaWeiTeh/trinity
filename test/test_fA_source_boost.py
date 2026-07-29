"""Tests for the f_A interface source-term boost (cooling_boost_fA).

f_A (SOURCE_TERM_DESIGN.md) multiplies the net radiative source dudt inside the
bubble-structure ODE AND the resolved L2+L3 loss integrals, in the interface band
T < 10^5.5 K -- i.e. the bubble-side conduction front (L2+L3) draped on the R2
contact discontinuity, not the interface surface itself. Default 1.0 = byte-identical.
Two production edit sites in
trinity/bubble_structure/bubble_luminosity.py:
  - _get_bubble_ODE RHS: dudt -> fA*dudt when T < _T_INTERFACE_BAND (edit site 1);
  - _bubble_luminosity: L_conduction, L_intermediate -> fA*(...) (edit site 2).
"""
import json
import os

import numpy as np
import pytest
import scipy.interpolate

import trinity.bubble_structure.bubble_luminosity as BL
import trinity.cooling.non_CIE.read_cloudy as non_CIE
from trinity.cooling import net_coolingcurve
from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param
from trinity._input.registry import SPECS

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE = os.path.join(_REPO, "param", "simple_cluster.param")
_FIXTURE = os.path.join(_REPO, "test", "data", "residual_resample_fixture.json")
_PATH_SKIP = {"path_cooling_CIE", "path_cooling_nonCIE", "sps_path", "path2output"}


def _build_full_params(fixture_path):
    """Full params (cooling cubes rebuilt) from a distilled state fixture.
    Mirrors test_dR2min_magic_number._build_params."""
    with open(fixture_path) as fh:
        fixture = json.load(fh)
    params = read_param(os.path.join(_REPO, fixture["base_param"]))
    for k, v in fixture["param_values"].items():
        if k not in _PATH_SKIP and k in params:
            params[k].value = v
    logT, logL = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    if "cStruc_cooling_CIE_logLambda" in params:
        params["cStruc_cooling_CIE_logLambda"].value = logL
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logL, kind="linear")
    cN, hN, net = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cN
    params["cStruc_heating_nonCIE"].value = hN
    params["cStruc_net_nonCIE_interpolation"].value = net
    return params


# --- registry -----------------------------------------------------------------

def test_registry_default_present_and_run_const():
    p = read_param(_BASE)
    assert "cooling_boost_fA" in p
    assert p["cooling_boost_fA"].value == 1.0
    spec = next(s for s in SPECS if s.name == "cooling_boost_fA")
    assert spec.run_const is True
    assert spec.exclude_from_snapshot is True


@pytest.mark.parametrize("bad", ["0", "-1", "-3.5"])
def test_rejects_nonpositive(tmp_path, bad):
    f = tmp_path / f"fa_{bad}.param"
    f.write_text(open(_BASE).read() + f"\ncooling_boost_fA    {bad}\n")
    with pytest.raises(ParameterFileError):
        read_param(str(f))


def test_cross_knob_warning(tmp_path, caplog):
    # fA != 1 with mode != none must warn (double-boost); fA=1 must NOT warn.
    f = tmp_path / "cross.param"
    f.write_text(open(_BASE).read()
                 + "\ncooling_boost_fA    4\ncooling_boost_mode    multiplier\n"
                 "cooling_boost_fmix    2\n")
    with caplog.at_level("WARNING"):
        read_param(str(f))
    assert any("SINGLE knob" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING"):
        read_param(_BASE)  # default fA=1
    assert not any("SINGLE knob" in r.message for r in caplog.records)


# --- edit site 1: band-limited RHS boost --------------------------------------

def test_rhs_band_limiting(monkeypatch):
    # get_dudt -> constant so no cooling cubes are needed; isolates the band gate.
    monkeypatch.setattr(net_coolingcurve, "get_dudt", lambda *a, **k: -1.0e3)
    p = read_param(_BASE)
    p["t_now"].value = 1.0
    r, Pb, v, dTdr = 5.0, 1.0e6, 10.0, -1.0e4

    def rhs(fa, T):
        p["cooling_boost_fA"].value = fa
        return BL._get_bubble_ODE(r, [v, T, dTdr], p, Pb)

    # above the band (T > 10^5.5): boost branch not entered -> identical triplet
    a1, a2 = rhs(1.0, 1.0e6), rhs(2.0, 1.0e6)
    assert a1 == a2

    # in the band (T < 10^5.5): dvdr and dTdr unchanged, dTdrr carries the boost
    b1, b2 = rhs(1.0, 1.0e5), rhs(2.0, 1.0e5)
    assert b1[0] == b2[0]        # dvdr does not depend on dudt
    assert b1[1] == b2[1]        # dTdr is the input, untouched
    assert b1[2] != b2[2]        # dTdrr contains dudt -> differs


def test_rhs_default_is_inert(monkeypatch):
    # fA=1.0 must give the byte-identical triplet the boost-free code produced.
    monkeypatch.setattr(net_coolingcurve, "get_dudt", lambda *a, **k: -1.0e3)
    p = read_param(_BASE)
    p["t_now"].value = 1.0
    p["cooling_boost_fA"].value = 1.0
    got = BL._get_bubble_ODE(5.0, [10.0, 1.0e5, -1.0e4], p, 1.0e6)
    # recompute the production expression WITHOUT the guard branch (fA absent)
    T, r, Pb, v, dTdr = 1.0e5, 5.0, 1.0e6, 10.0, -1.0e4
    ndens = Pb / ((p["mu_convert"].value / p["mu_ion"].value) * p["k_B"].value * T)
    dudt = -1.0e3
    v_term = p["cool_alpha"].value * r / p["t_now"].value
    dTdrr = (Pb / (p["cooling_boost_kappa"].value * p["C_thermal"].value * T ** (5 / 2)) * (
        (p["cool_beta"].value + 2.5 * p["cool_delta"].value) / p["t_now"].value
        + 2.5 * (v - v_term) * dTdr / T - dudt / Pb
    ) - 2.5 * dTdr ** 2 / T - 2 * dTdr / r)
    dvdr = ((p["cool_beta"].value + p["cool_delta"].value) / p["t_now"].value
            + (v - v_term) * dTdr / T - 2 * v / r)
    assert got[0] == dvdr and got[1] == dTdr and got[2] == dTdrr


# --- edit site 2: interface-band loss scaling ---------------------------------

def test_loss_component_scaling(monkeypatch):
    # ~3s: one full bubble solve x2. Freeze the profile (band top -> 0 disables edit
    # site 1) so ONLY edit site 2's
    # component scaling varies: L2, L3 scale by fA exactly; L1 (CIE) is unchanged.
    monkeypatch.setattr(BL, "_T_INTERFACE_BAND", 0.0)
    p = _build_full_params(_FIXTURE)

    def solve(fa):
        p["cooling_boost_fA"].value = fa
        p["bubble_dMdt"].value = float("nan")
        bp = BL.get_bubbleproperties_pure(p)
        return bp.bubble_L1Bubble, bp.bubble_L2Conduction, bp.bubble_L3Intermediate

    l1a, l2a, l3a = solve(1.0)
    l1b, l2b, l3b = solve(2.0)
    assert l1b == l1a                              # L1 (CIE interior) NOT scaled
    assert l2b == pytest.approx(2.0 * l2a, rel=1e-12)
    assert l3b == pytest.approx(2.0 * l3a, rel=1e-12)


# --- band-edge pin (audit G9) -------------------------------------------------

def test_band_edge_pinned_to_cooling_table():
    # _T_INTERFACE_BAND must equal the non-CIE cutoff of the default bundle (in LOG10
    # space -- _noncie_cutoffs returns log10 grid values, compared against np.log10(T)).
    # A cooling-table swap that moves the cutoff then fails here loudly, instead of
    # silently splitting the f_A band from the L2 mask.
    p = _build_full_params(_FIXTURE)
    cutoff_log10, _tmin = net_coolingcurve._noncie_cutoffs(p["cStruc_cooling_nonCIE"].value)
    assert cutoff_log10 == pytest.approx(np.log10(BL._T_INTERFACE_BAND), abs=1e-12)
