"""Tests for cooling_scale (f_cool), the suppression counterpart of the cooling_boost_* family.

cooling_scale (docs/dev/cooling-enhance-suppress/PLAN.md) multiplies the net radiative source
dudt inside the bubble-structure ODE at EVERY temperature (no band) and the resolved L1+L2+L3
loss integrals. Default 1.0 = byte-identical; 0 = the conduction-only Weaver structure; > 1 refused.
Two production edit sites in trinity/bubble_structure/bubble_luminosity.py:
  - _get_bubble_ODE RHS: dudt -> f_cool*dudt after the f_A band block (edit site E2a);
  - _bubble_luminosity: L_bubble, L_conduction, L_intermediate -> f_cool*(...) (edit site E2b).
Mirrors test/test_fA_source_boost.py. This file must live at test/test_cooling_scale.py: _REPO is
derived from its own location, so collected from patches/ it cannot find the repo.
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

# The fixture is tracked, but its ``base_param`` points into docs/dev, which is untracked
# (local-only). Only the two full-solve tests need it; the rest of this module runs normally.
with open(_FIXTURE) as _fh:
    _BASE_PARAM = os.path.join(_REPO, json.load(_fh)["base_param"])
_needs_base_param = pytest.mark.skipif(
    not os.path.isfile(_BASE_PARAM),
    reason="docs/dev is untracked (local-only); fixture base_param unavailable",
)


def _build_full_params(fixture_path):
    """Full params (cooling cubes rebuilt) from a distilled state fixture.
    Mirrors test_fA_source_boost._build_full_params."""
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
    assert "cooling_scale" in p
    assert p["cooling_scale"].value == 1.0
    spec = next(s for s in SPECS if s.name == "cooling_scale")
    assert spec.run_const is True
    assert spec.exclude_from_snapshot is True


@pytest.mark.parametrize("bad", ["-0.5", "1.5", "2", "abc", "true"])
def test_rejects_out_of_range(tmp_path, bad):
    # "true" parses to a bool before the validators run (read_param.py:85-89); float(True) would
    # silently read as 1.0 = off, so booleans are refused by name.
    f = tmp_path / "cs_bad.param"
    f.write_text(open(_BASE).read() + f"\ncooling_scale    {bad}\n")
    with pytest.raises(ParameterFileError):
        read_param(str(f))


@pytest.mark.parametrize("ok", ["0", "0.3", "1"])
def test_accepts_closed_unit_interval(tmp_path, ok):
    f = tmp_path / "cs_ok.param"
    f.write_text(open(_BASE).read() + f"\ncooling_scale    {ok}\n")
    assert read_param(str(f))["cooling_scale"].value == float(ok)


def test_cross_knob_warning(tmp_path, caplog):
    # f_cool != 1 with an active boost knob must warn; the default must not.
    f = tmp_path / "cross.param"
    f.write_text(open(_BASE).read() + "\ncooling_scale    0.5\ncooling_boost_fA    2\n")
    with caplog.at_level("WARNING"):
        read_param(str(f))
    assert any("cooling_scale" in r.message and "SINGLE knob" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING"):
        read_param(_BASE)
    assert not any("cooling_scale" in r.message for r in caplog.records)


# --- edit site E2a: the RHS, no band ------------------------------------------

def _production_rhs_without_scale(p, r, T, Pb, v, dTdr, dudt):
    """The RHS triplet as the unscaled production expression computes it."""
    v_term = p["cool_alpha"].value * r / p["t_now"].value
    dTdrr = (Pb / (p["cooling_boost_kappa"].value * p["C_thermal"].value * T ** (5 / 2)) * (
        (p["cool_beta"].value + 2.5 * p["cool_delta"].value) / p["t_now"].value
        + 2.5 * (v - v_term) * dTdr / T - dudt / Pb
    ) - 2.5 * dTdr ** 2 / T - 2 * dTdr / r)
    dvdr = ((p["cool_beta"].value + p["cool_delta"].value) / p["t_now"].value
            + (v - v_term) * dTdr / T - 2 * v / r)
    return [dvdr, dTdr, dTdrr]


def test_rhs_default_is_inert(monkeypatch):
    monkeypatch.setattr(net_coolingcurve, "get_dudt", lambda *a, **k: -1.0e3)
    p = read_param(_BASE)
    p["t_now"].value = 1.0
    p["cooling_scale"].value = 1.0
    for T in (1.0e5, 1.0e6):                       # below and above the f_A band
        got = BL._get_bubble_ODE(5.0, [10.0, T, -1.0e4], p, 1.0e6)
        want = _production_rhs_without_scale(p, 5.0, T, 1.0e6, 10.0, -1.0e4, -1.0e3)
        assert got[0] == want[0] and got[1] == want[1] and got[2] == want[2]


def test_rhs_scales_everywhere(monkeypatch):
    # unlike f_A there is no band: dTdrr changes at T = 1e6 K as well as at 1e5 K; dvdr, dTdr never.
    monkeypatch.setattr(net_coolingcurve, "get_dudt", lambda *a, **k: -1.0e3)
    p = read_param(_BASE)
    p["t_now"].value = 1.0

    def rhs(f, T):
        p["cooling_scale"].value = f
        return BL._get_bubble_ODE(5.0, [10.0, T, -1.0e4], p, 1.0e6)

    for T in (1.0e5, 1.0e6):
        a, b = rhs(1.0, T), rhs(0.5, T)
        assert a[0] == b[0] and a[1] == b[1] and a[2] != b[2]
        # the scaled triplet equals the production expression with dudt -> 0.5*dudt
        want = _production_rhs_without_scale(p, 5.0, T, 1.0e6, 10.0, -1.0e4, 0.5 * -1.0e3)
        assert b[2] == want[2]


def test_rhs_zero_is_the_conduction_only_structure(monkeypatch):
    monkeypatch.setattr(net_coolingcurve, "get_dudt", lambda *a, **k: -1.0e3)
    p = read_param(_BASE)
    p["t_now"].value = 1.0
    p["cooling_scale"].value = 0.0
    got = BL._get_bubble_ODE(5.0, [10.0, 1.0e5, -1.0e4], p, 1.0e6)
    want = _production_rhs_without_scale(p, 5.0, 1.0e5, 1.0e6, 10.0, -1.0e4, 0.0)
    assert got[2] == want[2]


# --- edit site E2b: every component scales ------------------------------------

@_needs_base_param
def test_loss_components_scale_together(monkeypatch):
    # ~3 s each: full bubble solves. Freeze the profile (net_coolingcurve.get_dudt untouched by
    # the scale: set it to the identity path by using f = 1 in the ODE via a monkeypatched
    # params read) -- simpler: compare the components of two solves whose ODE sees the SAME
    # source by scaling only after the solve. The RHS factor is tested above in isolation.
    p = _build_full_params(_FIXTURE)
    rhs_orig = BL._get_bubble_ODE

    def rhs_unscaled(r, y, params, Pb):
        saved = params["cooling_scale"].value
        params["cooling_scale"].value = 1.0
        try:
            return rhs_orig(r, y, params, Pb)
        finally:
            params["cooling_scale"].value = saved

    monkeypatch.setattr(BL, "_get_bubble_ODE", rhs_unscaled)

    def solve(f):
        p["cooling_scale"].value = f
        p["bubble_dMdt"].value = float("nan")
        bp = BL.get_bubbleproperties_pure(p)
        return bp.bubble_L1Bubble, bp.bubble_L2Conduction, bp.bubble_L3Intermediate, bp.bubble_LTotal

    l1a, l2a, l3a, lta = solve(1.0)
    l1b, l2b, l3b, ltb = solve(0.25)
    assert l1b == pytest.approx(0.25 * l1a, rel=1e-12)      # L1 scales too (f_A leaves it alone)
    assert l2b == pytest.approx(0.25 * l2a, rel=1e-12)
    assert l3b == pytest.approx(0.25 * l3a, rel=1e-12)
    assert ltb == pytest.approx(0.25 * lta, rel=1e-12)
    l1z, l2z, l3z, ltz = solve(0.0)
    assert l1z == 0.0 and l2z == 0.0 and l3z == 0.0 and ltz == 0.0


@_needs_base_param
def test_full_solve_runs_at_zero():
    # the structure solve itself (RHS scaled, f = 0) must return a finite profile: the limit the
    # ladder's S000 arm relies on. A solver failure here is a plan bug, not a physics result.
    p = _build_full_params(_FIXTURE)
    p["cooling_scale"].value = 0.0
    p["bubble_dMdt"].value = float("nan")
    bp = BL.get_bubbleproperties_pure(p)
    assert bp.bubble_LTotal == 0.0
    assert np.isfinite(bp.bubble_Tavg) and bp.bubble_Tavg > 0
    assert np.isfinite(bp.bubble_dMdt)
