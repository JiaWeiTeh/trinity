"""cover_law sigma (docs/dev/leaky-reservoir/CF_LAW_PLAN.md, rulings D-1 to D-9 of 2026-11-26): after the vent the
closed fraction of the leaky reservoir follows the shell's column, C_f = min(1, Phi(-(y + a)/sigma) / Phi(-a/sigma)),
y = ln(Sigma_v/Sigma), 1 at the vent, 1/2 at h = cf_half_radius vent radii at fixed mass. These tests pin the law (its
limits, the plan's erfc form and table, the log-normal picture it comes from), the column and the hook, the keys and
validators, and that the phase-1c rows and right-hand side of a reservoir run use it."""

from __future__ import annotations

import json
import logging
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import erfc

import trinity._functions.unit_conversions as cvt
import trinity.bubble_structure.get_bubbleParams as G
import trinity.phase1c_transition.run_transition_phase as TP
import trinity.phase2_momentum.swept as S
from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param
from trinity._input.registry import SPECS, _validate_cf_half_radius, _validate_cover_law, _validate_vent_model
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp
from trinity.sps import read_sps
from trinity.sps.update_feedback import get_current_sps_feedback

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
H_RANGE = (1.01, 1.2, 1.5, 2.0, 3.0, 10.0, 1e6)


def _ns(**kw):
    """Hand-built params: {key: object with .value}, as the validators and helpers read them."""
    return {k: SimpleNamespace(value=v) for k, v in kw.items()}


# ---------------------------------------------------------------- the law (CF_LAW_PLAN §1)

@pytest.mark.parametrize("h", H_RANGE)
def test_law_is_one_at_the_vent_and_half_at_h_vent_radii(h):
    assert G.cf_law(1.0, h) == 1.0                          # exactly: no jump at the vent
    assert G.cf_law(h**2, h) == pytest.approx(0.5, abs=1e-12)   # fixed mass: Sigma_v/Sigma = (R/R_v)^2


def test_law_equals_the_plans_erfc_form_and_its_table():
    """The code's normal-CDF form against CF_LAW_PLAN §1's erfc form, s = sqrt(2) sigma, where erfc does not
    underflow, and the §1 table (sigma 1.5, fixed mass)."""
    s = math.sqrt(2.0) * G.CF_SIGMA
    table = {1.5: (0.753, 0.5, 0.266, 0.089, 0.015), 2.0: (0.876, 0.711, 0.5, 0.254, 0.078),
             3.0: (0.950, 0.868, 0.731, 0.5, 0.235)}
    for h, row in table.items():
        a = G.cf_offset(h)
        for R, cf in zip((1.2, 1.5, 2.0, 3.0, 5.0), row):
            y = 2.0 * math.log(R)
            assert G.cf_law(R**2, h) == pytest.approx(min(1.0, erfc((y + a) / s) / erfc(a / s)), rel=1e-12)
            assert G.cf_law(R**2, h) == pytest.approx(cf, abs=5e-4)


def test_law_is_monotone_clipped_at_one_and_opens_fully():
    ratios = np.logspace(-3.0, 8.0, 2001)
    for h in (1.5, 2.0, 3.0):
        c = np.array([G.cf_law(x, h) for x in ratios])
        assert np.all(np.diff(c) <= 0.0) and np.all((c >= 0.0) & (c <= 1.0))
        assert np.all(c[ratios <= 1.0] == 1.0)              # a column that grew again: closed (re-closing)
        assert G.cf_law(1e12, h) < 1e-6
    for x in (10.0, 1e3, 1e6):                              # the G1 limit: h = 1e6 keeps the sealed reservoir
        assert G.cf_law(x, 1e6) == 1.0


def test_law_is_the_conditional_probability_of_lognormal_columns():
    """The picture of §1: ln(Sigma_i/Sigma) = sigma z_i over the sky, a direction closed while its column is above a
    threshold; at fixed mass per direction every column scales with the mean. The fraction of the directions closed at
    the vent that are still closed once the mean has fallen by exp(y) is the law (a Monte Carlo of 2e6 directions,
    fixed seed; bar 5 binomial standard deviations)."""
    z = np.random.default_rng(20261126).standard_normal(2_000_000)
    for h in (1.5, 2.0, 3.0):
        a = G.cf_offset(h)
        closed_v = G.CF_SIGMA * z > a
        n = int(closed_v.sum())
        for y in (0.2, 0.7, 1.5, 2.5):
            p = float(np.count_nonzero(G.CF_SIGMA * z[closed_v] > a + y)) / n
            assert abs(p - G.cf_law(math.exp(y), h)) < 5.0 * math.sqrt(max(p * (1.0 - p), 1.0 / n) / n)


# ---------------------------------------------------------------- the closed fraction in force, the column, the hook

def _vented(law, **kw):
    base = dict(vent_model="reservoir", transition_channel="blowout", coverFraction=1.0, cover_law=law,
                cf_half_radius=2.0, _cf_Sigma_vent=0.0, rs_dyn=12.0, shell_mass=3.0e4, bubble_mass=40.0,
                current_phase="transition", shell_mass_net=True, _shell_mass_net_on=True)
    base.update(kw)
    return _ns(**base)


def test_constant_law_is_the_shipped_cover_fraction():
    for law in (None, "constant"):
        kw = {} if law is None else {"cover_law": law}
        assert G.cover_fraction(_ns(vent_model="reservoir", transition_channel="blowout", coverFraction=0.6, **kw)) == 0.6
        assert G.cover_fraction(_ns(vent_model="reservoir", transition_channel="", coverFraction=0.6, **kw)) == 1.0
        assert G.cover_fraction(_ns(coverFraction=0.7, **kw), 5.0, 1.0) == 0.7     # r, m ignored
    assert G.cover_law({}) == "constant"


def test_sigma_law_is_sealed_until_the_column_at_the_vent_is_stored():
    assert G.cover_fraction(_vented("sigma", transition_channel="")) == 1.0           # before the vent
    assert G.cover_fraction(_vented("sigma")) == 1.0                                  # vented, nothing stored yet
    p = _vented("sigma")
    G.store_vent_column(p)
    Sv = p["_cf_Sigma_vent"].value
    assert Sv == (3.0e4 - 40.0) / (4.0 * np.pi * 12.0**2)                             # the shell's own mass over 4 pi r_s^2
    assert G.cover_fraction(p) == 1.0                                                 # exactly 1 at the vent
    p["rs_dyn"].value = 12.0 * 1.5                                                    # the row moves out at fixed mass
    assert G.cover_fraction(p) == G.cf_law(1.5**2, 2.0) == pytest.approx(0.711, abs=5e-4)
    m = 2.0e4                                                                         # explicit (r, m): the RHS's call
    assert G.cover_fraction(p, 15.0, m) == G.cf_law(Sv / (m / (4.0 * np.pi * 15.0**2)), 2.0)
    p["shell_mass"].value = 9.0e4                                                     # the hook stores once
    G.store_vent_column(p)
    assert p["_cf_Sigma_vent"].value == Sv


def test_column_and_hook_outside_the_law():
    p = _vented("constant")
    G.store_vent_column(p)
    assert p["_cf_Sigma_vent"].value == 0.0
    G.store_vent_column({})                                                           # no keys at all: a no-op
    q = _vented("sigma", current_phase="momentum")                                    # the bubble's gas is back
    assert G.shell_column(q) == 3.0e4 / (4.0 * np.pi * 12.0**2)


# ---------------------------------------------------------------- keys and validators

def test_keys_and_defaults():
    spec = {s.name: s for s in SPECS}
    law, h, sv = spec["cover_law"], spec["cf_half_radius"], spec["_cf_Sigma_vent"]
    assert law.default == "sigma" and law.validator is _validate_cover_law   # the default since C1D_PLAN §10.3
    assert h.default == "2" and h.validator is _validate_cf_half_radius
    for s in (law, h):
        assert s.run_const is True and s.exclude_from_snapshot is True and s.category == "input_solver"
    assert sv.default == 0.0 and sv.exclude_from_snapshot is True and sv.category == "runtime_control"


def test_cover_law_validator():
    ok = _ns(vent_model="reservoir", coverFraction=1.0)
    for name in ("constant", "sigma"):
        _validate_cover_law(name, ok)
    _validate_cover_law("sigma", _ns(vent_model="reservoir"))                         # coverFraction at its default
    _validate_cover_law("constant", _ns(vent_model="vacuum", coverFraction=0.5))
    for bad in ("Sigma", "lognormal", "", None, 1.0):
        with pytest.raises(ParameterFileError):
            _validate_cover_law(bad, ok)
    for model in ("vacuum", "ambient"):   # the law acts in the reservoir only: elsewhere accepted, a no-op (C1D §10.3)
        _validate_cover_law("sigma", _ns(vent_model=model, coverFraction=1.0))
        _validate_cover_law("sigma", _ns(vent_model=model, coverFraction=0.77))
    _validate_cover_law("sigma", {})                                                  # vent_model falls back to vacuum
    with pytest.raises(ParameterFileError):                                           # one source of C_f
        _validate_cover_law("sigma", _ns(vent_model="reservoir", coverFraction=0.5))


def test_cf_half_radius_validator():
    for good in (1.01, 2, 2.0, 1e6):
        _validate_cf_half_radius(good, {})
    for bad in (1.0, 0.5, -2.0, 0, float("inf"), float("nan"), True, "2", None):
        with pytest.raises(ParameterFileError):
            _validate_cf_half_radius(bad, {})


def test_vent_model_warning_is_skipped_under_the_law(caplog):
    ok = dict(phii_scheme="swept", transition_trigger="cooling_balance,blowout", coverFraction=1.0)
    with caplog.at_level(logging.WARNING):
        _validate_vent_model("reservoir", _ns(**ok))
    assert "stays sealed after the vent" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _validate_vent_model("reservoir", _ns(cover_law="sigma", **ok))
    assert "stays sealed after the vent" not in caplog.text


# ---------------------------------------------------------------- the runner (the transition-entry state of
# test_vent_reservoir.py: simple_cluster, R2 = 0.6 rCloud, 10 km/s at 0.5 Myr, P_th = 3 P_ram), entered as a swept 1b
# hands it over (rs_dyn > 0), from the state the runner's own hand-off would have built

def _entry(tmp_path, stop_t=0.52):
    txt = re.sub(r"^(path2output\s+).*$", lambda m: m.group(1) + str(tmp_path),
                 (REPO / "param" / "simple_cluster.param").read_text(), flags=re.M)
    if "path2output" not in txt:
        txt += f"\npath2output    {tmp_path}\n"
    (tmp_path / "sc.param").write_text(txt)
    p = read_param(str(tmp_path / "sc.param"))
    get_InitCloudProp(p)
    sps = read_sps.read_sps(p["mCluster"].value / p["sps_refmass"].value, p)
    p["sps_data"].value, p["sps_f"].value = sps, read_sps.get_interpolation(sps)
    t0, R2 = 0.5, 0.6 * p["rCloud"].value
    fb = get_current_sps_feedback(t0, p)
    P_ram = G.pRam(R2, fb.Lmech_total, fb.v_mech_total)
    p["t_now"].value, p["stop_t"].value, p["current_phase"].value = t0, stop_t, "transition"
    p["R2"].value, p["v2"].value, p["rShell"].value = R2, 10.0 * cvt.v_kms2au, 1.2 * R2
    p["shell_mass"].value = S._scalar(S.mass_profile.get_mass_profile(R2, p))
    p["Eb"].value = 3.0 * P_ram * (4 * math.pi / 3) * R2**3 / (p["gamma_adia"].value - 1)
    p["bubble_LTotal"].value, p["bubble_Tavg"].value = 0.3 * fb.Lmech_total, 3e6
    p["cool_alpha"].value = t0 / R2 * p["v2"].value
    p["phii_scheme"].value = "swept"
    p["vent_model"].value, p["transition_channel"].value = "reservoir", "blowout"
    p["cover_law"].value, p["cf_half_radius"].value = "sigma", 2.0
    S.updateDict(p, fb)                                     # the feedback at t0, as the runner sets it before its solves
    rs, Ms, u = S.initial_state(p, fb, p["Eb"].value)     # nothing stored yet, so its shell solves see C_f = 1
    p["rs_dyn"].value, p["rShell"].value, p["shell_mass"].value, p["v2"].value = rs, rs, Ms, u
    # evaporated gas in the bubble (shell_mass_net, latched): the shell's own mass is Ms - 0.1 Ms, and the law must see it
    p["bubble_mass"].value, p["_shell_mass_net_on"].value = 0.1 * Ms, True
    return p


def _rows(p, tmp_path):
    p.flush()
    return [json.loads(ln) for f in tmp_path.rglob("dictionary.jsonl") for ln in f.read_text().splitlines()
            if ln.strip()]


def _column(r):
    m_ev = r["bubble_mass"] if r["current_phase"] == "transition" and r["bubble_mass"] < r["shell_mass"] else 0.0
    return (r["shell_mass"] - m_ev) / (4.0 * math.pi * r["rs_dyn"] ** 2)


def test_first_reservoir_row_stores_the_column_and_is_sealed(tmp_path):
    p = _entry(tmp_path)
    TP.run_phase_transition(p)
    tr = [r for r in _rows(p, tmp_path) if r["current_phase"] == "transition"]
    assert len(tr) >= 1
    Sv = p["_cf_Sigma_vent"].value
    assert Sv == pytest.approx(_column(tr[0]), rel=1e-14)
    assert tr[0]["shell_fLeak"] == 0.0 and tr[0]["bubble_Leak"] == 0.0
    for r in tr:
        assert r["shell_fLeak"] == pytest.approx(1.0 - G.cf_law(Sv / _column(r), 2.0), abs=1e-12)


def test_reservoir_rows_and_rhs_follow_the_law(tmp_path, monkeypatch):
    """A column at the vent three times the entry's (as if the shell had already thinned): C_f ~ 0.6 from the first row.
    Each row's shell_fLeak and leak use the law at the row's column; the integrated dEb/dt at the first segment start
    is L_mech - L_cool - 4 pi R2^2 P_b U - L_leak with that leak; the RHS asks for C_f at its own (r, m); the stored
    column is not overwritten. It runs under the registry defaults, cover_law sigma and hole_forces on (C1D_PLAN.md
    §10.3), so the RHS's C_f calls come from hole_mass and the leak line."""
    p = _entry(tmp_path)
    m_b = p["bubble_mass"].value
    Sv = 3.0 * ((p["shell_mass"].value - m_b) / (4.0 * math.pi * p["rs_dyn"].value ** 2))
    p["_cf_Sigma_vent"].value = Sv
    g = lambda k: p[k].value  # noqa: E731
    L0 = g("bubble_LTotal")
    starts, calls, real_ivp, real_cf = [], [], S.scipy.integrate.solve_ivp, G.cover_fraction

    def spy_ivp(*a, **kw):                   # the runner's segment solves: their RHS at the segment start
        if not a and len(kw.get("y0", ())) == 3 and "events" in kw:
            n0 = len(calls)                  # the C_f calls of this RHS evaluation only (hole_mass asks too)
            d = kw["fun"](kw["t_span"][0], kw["y0"])
            starts.append((kw["y0"].copy(), d, [c for c in calls[n0:] if c[0] is not None]))
        return real_ivp(*a, **kw)

    def spy_cf(params, r=None, m=None):
        calls.append((r, m))
        return real_cf(params, r, m)

    monkeypatch.setattr(S.scipy.integrate, "solve_ivp", spy_ivp)
    monkeypatch.setattr(G, "cover_fraction", spy_cf)
    TP.run_phase_transition(p)
    monkeypatch.undo()
    tr = [r for r in _rows(p, tmp_path) if r["current_phase"] == "transition"]
    assert len(tr) >= 1 and len(starts) >= 1
    assert p["_cf_Sigma_vent"].value == Sv
    assert any(r is not None for r, _ in calls)                                     # D-7: per RHS call
    for r in tr:
        cf = G.cf_law(Sv / _column(r), 2.0)
        assert r["shell_fLeak"] == pytest.approx(1.0 - cf, abs=1e-12)
        leak = G.get_leak_luminosity(cf, r["R2"], r["Pb"], r["c_sound"], g("gamma_adia"), r["press_HII_in"])
        assert r["bubble_Leak"] == pytest.approx(leak, rel=1e-9, abs=1e-30)
    r, (y0, d, rhs_calls) = tr[0], starts[0]
    assert 1.0 - r["shell_fLeak"] < 0.9                                             # the law acts in this test
    assert (r["rs_dyn"], r["v2"], r["Eb"]) == tuple(y0)
    # the calls made inside the spy's RHS evaluation at the first segment start: C_f at the RHS's own (r, m), which
    # equals the row's up to the hand-off mass fixed point (1e-10); under the defaults hole_mass and the leak line both
    # ask, at the same (r, m); the energy equation is checked with that C_f
    assert rhs_calls
    r_rhs, m_rhs = rhs_calls[0]
    assert all(c == (r_rhs, m_rhs) for c in rhs_calls)
    assert r_rhs == y0[0]
    # D-8: the RHS hands the law the shell's own mass, its mass minus the evaporated mass (SWEEP: M(<r); COAST: Ms,
    # which equals M(<r) to the hand-off's 1e-10 at this entry); m alone would be 10 % off
    assert m_rhs == pytest.approx(S._scalar(S.mass_profile.get_mass_profile(r_rhs, p)) - m_b, rel=1e-9)
    cf_rhs = G.cf_law(Sv / (m_rhs / (4.0 * math.pi * r_rhs**2)), 2.0)
    assert cf_rhs == pytest.approx(1.0 - r["shell_fLeak"], abs=1e-8)
    # D-7: C_f is evaluated per RHS call, so a call at another radius gives another C_f (held, it would not)
    assert any(abs(G.cf_law(Sv / (mm / (4.0 * math.pi * x**2)), 2.0) - cf_rhs) > 1e-9 for x, mm in calls
               if x is not None)
    leak_rhs = G.get_leak_luminosity(cf_rhs, r["R2"], r["Pb"], r["c_sound"], g("gamma_adia"), r["press_HII_in"])
    Ed = r["Lmech_total"] - L0 - 4 * math.pi * r["R2"] ** 2 * r["P_drive"] * r["rs_dot"] - leak_rhs
    assert d[2] == pytest.approx(Ed, rel=1e-9)
