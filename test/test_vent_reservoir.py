"""vent_model (docs/dev/leaky-reservoir/PLAN.md, rulings L-0 to L-13 of 2026-11-25): what the hot gas does once the
shell has vented. 'vacuum' (the default) is the code as it was; 'ambient' holds a vented shell's interior at the
outside pressure; 'reservoir' lets the bubble leak through 1 - coverFraction against that pressure in phase 1c.
These tests pin the helpers (the nozzle factor, the leak with a back-pressure, who counts as vented, the floor), the
validator, and that the phase-1c rows of a reservoir run describe the right-hand side that is integrated."""

from __future__ import annotations

import json
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
import trinity.phase1c_transition.run_transition_phase as TP
import trinity.phase2_momentum.swept as S
from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param
from trinity._input.registry import SPECS, _validate_vent_model
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp
from trinity.sps import read_sps
from trinity.sps.update_feedback import get_current_sps_feedback

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
GAMMA = 5.0 / 3.0


def _ns(**kw):
    """Hand-built params: {key: object with .value}, as the validators and helpers read them."""
    return {k: SimpleNamespace(value=v) for k, v in kw.items()}


# ---------------------------------------------------------------- the nozzle factor psi (PLAN 4.2)

def test_vent_factor_limits_and_tabulated_values():
    """1 while choked (x <= x* = 0.487), 0 at x >= 1, the isentropic flux over its choked value between: values worked
    out by hand and by a standalone script for gamma = 5/3 (PLAN 4.2)."""
    f = get_bubbleParams.vent_factor
    x_star = (2.0 / (GAMMA + 1.0)) ** (GAMMA / (GAMMA - 1.0))
    assert x_star == pytest.approx(0.4871392896, rel=1e-9)
    for x in (0.0, 0.1, x_star):
        assert f(x, GAMMA) == 1.0
    for x in (1.0, 1.5, 100.0):
        assert f(x, GAMMA) == 0.0
    for x, psi in ((0.5, 0.999665977778), (0.7, 0.906477772497), (0.9, 0.587209220646), (0.99, 0.193866676931)):
        assert f(x, GAMMA) == pytest.approx(psi, rel=1e-9)


def test_vent_factor_is_continuous_monotone_and_linear_near_one():
    f = get_bubbleParams.vent_factor
    xs = np.linspace(0.3, 1.0, 2001)
    y = np.array([f(x, GAMMA) for x in xs])
    assert np.all(np.diff(y) <= 1e-15) and np.all((y >= 0.0) & (y <= 1.0))
    assert np.max(np.abs(np.diff(y))) < 0.04          # no jump, at x* or at the start of the linear piece
    eps = get_bubbleParams.VENT_EPS
    edge = f(1.0 - eps, GAMMA)
    assert edge == pytest.approx(0.0615563071974, rel=1e-9)   # psi(0.999), the top of the linear piece
    assert f(1.0 - 0.25 * eps, GAMMA) == pytest.approx(0.25 * edge, rel=1e-9)


# ---------------------------------------------------------------- the leak with a back-pressure

def test_leak_without_back_pressure_is_the_shipped_expression_bit_for_bit():
    Cf, R2, Pb, cs = 0.8, 12.5, 3.7e4, 311.0
    shipped = GAMMA / (GAMMA - 1.0) * (1.0 - Cf) * 4.0 * np.pi * R2**2 * Pb * cs
    assert get_bubbleParams.get_leak_luminosity(Cf, R2, Pb, cs, GAMMA) == shipped
    assert get_bubbleParams.get_leak_luminosity(Cf, R2, Pb, cs, GAMMA, 0.0) == shipped
    assert get_bubbleParams.get_leak_luminosity(Cf, R2, Pb, cs, GAMMA, P_out=-1.0) == shipped


def test_leak_is_throttled_by_the_outside_pressure_and_stops_at_it():
    Cf, R2, Pb, cs = 0.5, 20.0, 1.0e4, 300.0
    leak = get_bubbleParams.get_leak_luminosity
    choked = leak(Cf, R2, Pb, cs, GAMMA)
    assert leak(Cf, R2, Pb, cs, GAMMA, 0.3 * Pb) == choked                      # choked: P_b > 2.05 P_out
    assert leak(Cf, R2, Pb, cs, GAMMA, 0.9 * Pb) == pytest.approx(0.587209220646 * choked, rel=1e-9)
    for P_out in (Pb, 1.0001 * Pb, 50.0 * Pb):
        assert leak(Cf, R2, Pb, cs, GAMMA, P_out) == 0.0
    assert leak(1.0, R2, Pb, cs, GAMMA, 0.9 * Pb) == 0.0                         # sealed
    assert leak(Cf, R2, -Pb, cs, GAMMA, 0.9 * Pb) == 0.0                         # no hot gas: never injects


def test_leak_supported_state_is_krumholz_matzner_eq29_with_their_sound_speed():
    """Steady state, no cooling, no work: L_w = L_leak. With Krumholz & Matzner's (2009) own isothermal c_X,
    c_X/v_w = [(1 - 1.045 C_f)/(5 (1 - C_f))]^(1/2) (their Eqs. 27-28), the bubble pressure is their Eq. 29,
    P = P_ram / [5 (1 - C_f)(1 - 1.045 C_f)]^(1/2): 0.915 P_ram at C_f = 0.5 (PLAN 4.4). The shipped product
    (5/2) P c is their enthalpy flux when c is their c_X."""
    L_w, v_w, R = 1.7e8, 2045.0, 15.0
    P_ram = get_bubbleParams.pRam(R, L_w, v_w)
    for Cf in (0.2, 0.5, 0.8):
        c_X = v_w * math.sqrt((1.0 - 1.045 * Cf) / (5.0 * (1.0 - Cf)))
        P = P_ram / math.sqrt(5.0 * (1.0 - Cf) * (1.0 - 1.045 * Cf))
        assert get_bubbleParams.get_leak_luminosity(Cf, R, P, c_X, GAMMA) == pytest.approx(L_w, rel=1e-12)
    assert 1.0 / math.sqrt(5.0 * 0.5 * (1.0 - 1.045 * 0.5)) == pytest.approx(0.9153, abs=1e-4)


# ---------------------------------------------------------------- who is vented, and the closed fraction in force

@pytest.mark.parametrize("model, channel, vented, cf", [
    ("vacuum", "blowout", False, 0.6), ("vacuum", "", False, 0.6), ("ambient", "", False, 0.6),
    ("ambient", "cooling_balance", False, 0.6), ("ambient", "blowout", True, 0.6),
    ("reservoir", "", False, 1.0), ("reservoir", "cooling_balance", False, 1.0), ("reservoir", "blowout", True, 0.6),
])
def test_vented_and_cover_fraction_by_mode_and_channel(model, channel, vented, cf):
    p = _ns(vent_model=model, transition_channel=channel, coverFraction=0.6)
    assert get_bubbleParams.vented(p) is vented
    assert get_bubbleParams.cover_fraction(p) == cf
    assert get_bubbleParams.vent_model(p) == model


def test_helpers_take_params_without_the_new_keys():
    """Hand-built params of older tests and harnesses have no vent_model or transition_channel: the code as it was."""
    assert get_bubbleParams.vented({}) is False
    assert get_bubbleParams.vent_model({}) == "vacuum"
    assert get_bubbleParams.cover_fraction({}) == 1.0
    assert get_bubbleParams.cover_fraction(_ns(coverFraction=0.7)) == 0.7
    assert "blowout" in get_bubbleParams.VENT_CHANNELS


# ---------------------------------------------------------------- the validator

def test_default_is_vacuum_and_only_the_three_names_pass():
    spec = next(s for s in SPECS if s.name == "vent_model")
    assert spec.default == "vacuum" and spec.validator is _validate_vent_model
    assert spec.run_const is True and spec.exclude_from_snapshot is True
    ok = _ns(phii_scheme="swept", transition_trigger="cooling_balance,blowout", coverFraction=0.5)
    for name in ("vacuum", "ambient", "reservoir"):
        _validate_vent_model(name, ok)
    _validate_vent_model("vacuum", {})
    for bad in ("Vacuum", "leaky", "", None, 1.0):
        with pytest.raises(ParameterFileError):
            _validate_vent_model(bad, ok)


@pytest.mark.parametrize("model", ["ambient", "reservoir"])
def test_a_mode_that_could_never_act_is_refused(model):
    with pytest.raises(ParameterFileError):      # not the swept scheme
        _validate_vent_model(model, _ns(phii_scheme="c3c", transition_trigger="cooling_balance,blowout"))
    with pytest.raises(ParameterFileError):      # no venting trigger: no shell would ever count as vented
        _validate_vent_model(model, _ns(phii_scheme="swept", transition_trigger="cooling_balance"))
    with pytest.raises(ParameterFileError):      # no energy phases at all
        _validate_vent_model(model, _ns(phii_scheme="swept", transition_trigger="blowout", start_scheme="stromgren"))
    _validate_vent_model(model, _ns(phii_scheme="swept", transition_trigger="r1"))     # r1 = blowout + ebpeak


# ---------------------------------------------------------------- the reservoir's cooling (PLAN 4.1, L-8)

def test_reservoir_cooling_follows_pressure_squared_times_volume():
    L0, E0, R2, R1 = 4.0e7, 2.0e9, 10.0, 3.0
    V0 = R2**3 - R1**3
    f = S.reservoir_cooling
    assert f(L0, E0, V0, E0, R2, R1) == pytest.approx(L0, rel=1e-15)
    assert f(L0, E0, V0, 0.5 * E0, R2, R1) == pytest.approx(0.25 * L0, rel=1e-15)                 # P halves at fixed V
    assert f(L0, E0, V0, E0, 2.0 * R2, 2.0 * R1) == pytest.approx(L0 / 8.0, rel=1e-15)            # E fixed, V x 8
    for E, r2, r1 in ((0.0, R2, R1), (-1.0, R2, R1), (E0, R2, R2)):
        assert f(L0, E0, V0, E, r2, r1) == 0.0                                                    # no hot gas
    assert f(L0, 0.0, V0, E0, R2, R1) == 0.0 and f(L0, E0, 0.0, E0, R2, R1) == 0.0


# ---------------------------------------------------------------- the runner (a transition-entry state, as
# test_phii_swept_transition.py: simple_cluster, R2 = 0.6 rCloud, 10 km/s at 0.5 Myr, P_th = f P_ram)

def _entry(tmp_path, f_thermal, stop_t):
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
    P_ram = get_bubbleParams.pRam(R2, fb.Lmech_total, fb.v_mech_total)
    p["t_now"].value, p["stop_t"].value, p["current_phase"].value = t0, stop_t, "transition"
    p["R2"].value, p["v2"].value, p["rShell"].value = R2, 10.0 * cvt.v_kms2au, 1.2 * R2
    p["shell_mass"].value = S._scalar(S.mass_profile.get_mass_profile(R2, p))
    p["Eb"].value = f_thermal * P_ram * (4 * math.pi / 3) * R2**3 / (p["gamma_adia"].value - 1)
    p["bubble_LTotal"].value, p["bubble_Tavg"].value = 0.3 * fb.Lmech_total, 3e6
    p["cool_alpha"].value = t0 / R2 * p["v2"].value
    p["phii_scheme"].value = "swept"
    return p, fb


def _rows(p, tmp_path):
    p.flush()
    return [json.loads(ln) for f in tmp_path.rglob("dictionary.jsonl") for ln in f.read_text().splitlines()
            if ln.strip()]


def test_floor_is_zero_until_vented_and_the_outside_pressure_after(tmp_path):
    p, fb = _entry(tmp_path, f_thermal=3.0, stop_t=0.52)
    # after read_param PISM is P/k in code units (K pc^-3); 1e8 K cm^-3 is well above the wind's ram pressure out here.
    # Past 1.2 rCloud the bridge weight is exactly 1.
    p["PISM"].value = 1.0e8 * cvt.ndens_cgs2au
    r, R2 = 1.5 * p["rCloud"].value, 1.4 * p["rCloud"].value
    P_ram = get_bubbleParams.pRam(R2, fb.Lmech_total, fb.v_mech_total)
    for model, channel in (("vacuum", "blowout"), ("ambient", "cooling_balance"), ("reservoir", "")):
        p["vent_model"].value, p["transition_channel"].value = model, channel
        assert S.vent_floor(r, p) == 0.0 and S.vent_floor(r, p, 1.0) == 0.0
        assert S._p_conf(R2, None, fb, p, rs=r)[0] == P_ram          # the wind's ram pressure, bit for bit
    assert S._p_conf(R2, None, fb, p)[0] == P_ram
    for model in ("ambient", "reservoir"):
        p["vent_model"].value, p["transition_channel"].value = model, "blowout"
        floor = S.vent_floor(r, p, 1.0)                              # photons trapped: P_ISM alone out here
        assert floor == S.p_ext(r, 1.0, p) == pytest.approx(p["PISM"].value * p["k_B"].value, rel=1e-12)
        assert floor > P_ram                                         # so the max below is the floor, not P_ram
        assert S.vent_floor(r, p, 0.5) == S.p_ext(r, 0.5, p) > floor     # photons escape: the ionised gas ahead too
        p["shell_fAbsorbedIon"].value = 1.0                          # the last accepted solve's flag (the root-find)
        assert S.vent_floor(r, p) == floor
        assert S._p_conf(R2, None, fb, p, rs=r) == (max(P_ram, floor), R2)
        assert S._p_conf(R2, None, fb, p)[0] == P_ram                # no outer radius given: as before


def test_reservoir_rows_describe_the_integrated_energy_equation(tmp_path, monkeypatch):
    """Phase 1c under vent_model reservoir, vented, C_f = 0.9: at each segment start the row's leak is the leak with the
    back-pressure, its loss is leak + L_cool0 (E/E0)^2 V0/V, and on the first segment (zeta = 1) the integrated dEb/dt
    is L_mech - L_cool - 4 pi R2^2 P_b U - L_leak, with no sound-crossing drain. In this state (P_th = 3 P_ram, cooling
    0.3 L_mech, a leak of at most ~0.2 L_mech) the balance is positive and the drain -Eb c_s/R2 ~ -0.4 L_mech, so a
    runner that kept the shipped min() would fail here."""
    p, _ = _entry(tmp_path, f_thermal=3.0, stop_t=0.52)
    p["vent_model"].value, p["transition_channel"].value, p["coverFraction"].value = "reservoir", "blowout", 0.9
    p["cover_law"].value = "constant"   # a fixed C_f: cover_law defaults to sigma since C1D_PLAN.md §10.3
    g = lambda k: p[k].value  # noqa: E731
    L0 = g("bubble_LTotal")
    starts, real = [], S.scipy.integrate.solve_ivp

    def spy(*a, **kw):                       # the runner's segment solves: their RHS at the segment start
        if not a and len(kw.get("y0", ())) == 3 and "events" in kw:
            starts.append((kw["y0"].copy(), kw["fun"](kw["t_span"][0], kw["y0"])))
        return real(*a, **kw)

    monkeypatch.setattr(S.scipy.integrate, "solve_ivp", spy)
    TP.run_phase_transition(p)
    monkeypatch.undo()
    tr = [r for r in _rows(p, tmp_path) if r["current_phase"] == "transition"]
    assert len(tr) >= 1 and len(starts) >= 1
    E0, V0 = tr[0]["Eb"], tr[0]["R2"] ** 3 - tr[0]["R1"] ** 3
    for r in tr:
        assert r["shell_fLeak"] == pytest.approx(0.1, abs=1e-12)
        assert r["Pb"] >= r["P_ram"] and r["P_drive"] == pytest.approx(r["Pb"], rel=1e-12)   # P_b = P_ram(R1) >= P_ram(R2)
        leak = get_bubbleParams.get_leak_luminosity(0.9, r["R2"], r["Pb"], r["c_sound"], g("gamma_adia"),
                                                    r["press_HII_in"])
        assert r["bubble_Leak"] == pytest.approx(leak, rel=1e-9, abs=1e-30)
        cool = L0 * (r["Eb"] / E0) ** 2 * V0 / (r["R2"] ** 3 - r["R1"] ** 3)
        assert r["bubble_Lloss"] - r["bubble_Leak"] == pytest.approx(cool, rel=1e-9)
    r, (y0, d) = tr[0], starts[0]
    assert (r["rs_dyn"], r["v2"], r["Eb"]) == tuple(y0)
    assert r["bubble_Lloss"] - r["bubble_Leak"] == pytest.approx(L0, rel=1e-12)
    Ed = r["Lmech_total"] - L0 - 4 * math.pi * r["R2"] ** 2 * r["P_drive"] * r["rs_dot"] - r["bubble_Leak"]
    assert d[2] == pytest.approx(Ed, rel=1e-9)
    assert Ed > 0 > -r["Eb"] * r["c_sound"] / r["R2"]     # the shipped min() would have taken the sound-crossing drain


def test_vacuum_run_of_the_same_state_keeps_the_sound_crossing_drain(tmp_path, monkeypatch):
    """The twin of the test above with vent_model left at vacuum and the same C_f, open from birth as before: the
    integrated dEb/dt is min(balance, -Eb c_s/R2)."""
    p, _ = _entry(tmp_path, f_thermal=3.0, stop_t=0.52)
    p["transition_channel"].value, p["coverFraction"].value = "blowout", 0.5
    g = lambda k: p[k].value  # noqa: E731
    assert g("vent_model") == "vacuum"
    starts, real = [], S.scipy.integrate.solve_ivp

    def spy(*a, **kw):
        if not a and len(kw.get("y0", ())) == 3 and "events" in kw:
            starts.append((kw["y0"].copy(), kw["fun"](kw["t_span"][0], kw["y0"])))
        return real(*a, **kw)

    monkeypatch.setattr(S.scipy.integrate, "solve_ivp", spy)
    TP.run_phase_transition(p)
    monkeypatch.undo()
    r = [x for x in _rows(p, tmp_path) if x["current_phase"] == "transition"][0]
    Ed = (r["Lmech_total"] - r["bubble_LTotal"] - 4 * math.pi * r["R2"] ** 2 * r["P_drive"] * r["rs_dot"]
          - get_bubbleParams.get_leak_luminosity(0.5, r["R2"], r["P_drive"], r["c_sound"], g("gamma_adia")))
    assert starts[0][1][2] == pytest.approx(min(Ed, -r["Eb"] * r["c_sound"] / r["R2"]), rel=1e-12)
