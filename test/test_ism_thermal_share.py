"""ism_thermal_share (docs/dev/ism-double-count/PLAN.md, rulings D-5..D-7 of 2026-11-27): once ionising photons escape,
the photoionised gas ahead of the shell replaces the ambient's thermal pressure P_th0, so the pressure ahead is
P_i + (P_ISM - P_th0) w(r) instead of P_i + P_ISM w(r).

  * 'fit' (the default): P_th0 = min(s P_ISM, (mu_convert/mu_atom) nISM k T_i), s = min(1, 0.354 (P/k / 1e4)^-0.337);
  * 'off', every photon absorbed, P_ISM 0 and r <= 0.8 rCloud give the sum without it bit for bit;
  * the subtraction follows the B3 bridge w(r), and every site that adds P_ISM subtracts it (the force functions
    and the three ODE right-hand sides by value);
  * the cap keeps a hand-typed hot pair (Rosette's PISM 1e6 / nISM 0.1) at P_out >= P_ISM.
On simple_cluster's own parameters (as test_pism_bridge.py), PISM typed in K cm^-3 and read by read_param.
Unit-level: no TRINITY run.
"""
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import trinity._functions.unit_conversions as cvt
import trinity.phase1_energy.energy_phase_ODEs as ODEs
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as IMP
import trinity.phase1c_transition.run_transition_phase as TR
import trinity.phase2_momentum.run_momentum_phase as MOM
import trinity.phase2_momentum.swept as S
from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param
from trinity.cloud_properties import density_profile, mass_profile
from trinity.cloud_properties.density_profile import get_press_ism_thermal
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp

REPO = Path(__file__).resolve().parents[1]
_SITES = (ODEs.get_ODE_Edot_pure, ODEs.compute_derived_quantities, IMP.compute_forces_pure, TR.compute_forces_pure,
          MOM.compute_forces_momentum_pure, MOM.get_ODE_momentum_pure, S.p_ext)


def _params(tmp_path, **over):
    """simple_cluster.param plus the keys in over, read as run.py reads it (read_param, then the cloud)."""
    keys = {"log_console": "False", "path2output": str(tmp_path / "out"), **{k: str(v) for k, v in over.items()}}
    par = tmp_path / "ts.param"
    par.write_text((REPO / "param" / "simple_cluster.param").read_text() + "\n"
                   + "\n".join(f"{k} {v}" for k, v in keys.items()) + "\n")
    P = read_param(str(par))
    get_InitCloudProp(P)
    P["Lmech_total"].value, P["v_mech_total"].value = 3e38 * cvt.L_cgs2au, 2000 * cvt.v_kms2au   # P_ram (bridge test's)
    return P


def _pk(P):
    """P_ISM k_B in code units per K cm^-3: divide a code pressure by it to get P/k in K cm^-3."""
    return P["k_B"].value * cvt.ndens_cgs2au


def _forces_P_ext(P, rShell, fabs):
    """press_HII_in (= P_ext) of the three c3c force functions at rShell: 1b, 1c, momentum."""
    sh = SimpleNamespace(rShell=rShell, shell_fAbsorbedIon=fabs, n_IF=0.0, R_IF=0.0, isDissolved=True)
    R2, mShell = 0.9 * rShell, S._scalar(mass_profile.get_mass_profile(rShell, P))
    Pb = 1e7 * cvt.CGS.k_B * cvt.Pb_cgs2au
    L, v = P["Lmech_total"].value, P["v_mech_total"].value
    return (IMP.compute_forces_pure(R2, mShell, Pb, sh, P).press_HII_in,
            TR.compute_forces_pure(R2, mShell, Pb, sh, P).press_HII_in,
            MOM.compute_forces_momentum_pure(R2, mShell, L, v, sh, P).press_HII_in)


@pytest.mark.parametrize("pk, share", [(1e4, 0.3540), (1e6, 0.0750), (1e8, 0.0159)])
def test_the_share_on_fit_c_ambients(tmp_path, pk, share):
    P = _params(tmp_path, PISM=pk)                          # nISM from sigma_eff_ISM fit: the cap does not bind
    assert P["ism_thermal_share"].value == "fit"
    r = 2.0 * P["rCloud"].value
    got = get_press_ism_thermal(r, 0.5, P) / (P["PISM"].value * P["k_B"].value)
    assert got == pytest.approx(share, abs=5e-5)
    assert got == pytest.approx(0.354 * (pk / 1e4) ** -0.337, rel=1e-12)


def test_the_share_is_one_below_its_crossing(tmp_path):
    """s reaches 1 at P/k = 1e4 (1/0.354)^(1/-0.337) ~ 459 K cm^-3: below it the whole ambient is thermal."""
    for pk, whole in ((300.0, True), (600.0, False)):
        P = _params(tmp_path, PISM=pk)
        r = 2.0 * P["rCloud"].value
        pism = P["PISM"].value * P["k_B"].value
        cap = P["mu_convert"].value / P["mu_atom"].value * P["nISM"].value * P["k_B"].value * P["TShell_ion"].value
        assert cap > pism                                    # fit C: the cap stays out of the way
        assert (get_press_ism_thermal(r, 0.5, P) == pytest.approx(pism, rel=1e-12)) is whole


def test_off_and_the_untouched_cases_are_bit_identical(tmp_path):
    P = _params(tmp_path, PISM=1e4)
    rc = P["rCloud"].value

    def all_sites(r, fabs):
        return (S.p_ext(r, fabs, P),) + _forces_P_ext(P, r, fabs)

    cases = [(x * rc, f) for x in (0.5, 0.79, 0.95, 1.0, 1.05, 1.21, 2.0) for f in (1.0, 0.5)]
    P["ism_thermal_share"].value = "off"
    off = [all_sites(r, f) for r, f in cases]
    assert all(get_press_ism_thermal(r, f, P) == 0.0 for r, f in cases)
    P["ism_thermal_share"].value = "fit"
    fit = [all_sites(r, f) for r, f in cases]
    for (r, f), a, b in zip(cases, off, fit):
        if f >= 1.0 or r <= 0.8 * rc:                        # photons held, or inside the bridge: nothing changes
            assert a == b
        else:
            assert all(y < x for x, y in zip(a, b))
    P0 = _params(tmp_path)                                    # P_ISM 0 (simple_cluster as shipped): nothing to subtract
    assert P0["PISM"].value == 0.0 and P0["ism_thermal_share"].value == "fit"
    assert all(get_press_ism_thermal(x * P0["rCloud"].value, 0.5, P0) == 0.0 for x in (1.0, 2.0))
    assert get_press_ism_thermal(2.0 * rc, float("nan"), P) == 0.0   # get_press_ion's gate: NaN adds no P_i
    del P["ism_thermal_share"]                                # hand-built params without the key: off
    assert get_press_ism_thermal(2.0 * rc, 0.5, P) == 0.0


def test_every_site_subtracts_it_with_the_bridge(tmp_path):
    P = _params(tmp_path, PISM=1e6)
    rc = P["rCloud"].value
    full = get_press_ism_thermal(2.0 * rc, 0.5, P)
    for x in (1.0, 1.05, 1.1, 1.5):                       # at 0.9 rCloud w ~ 2e-9: below P_i's rounding
        r = x * rc
        sub = get_press_ism_thermal(r, 0.5, P)
        assert sub == pytest.approx(density_profile.ism_weight(r, rc) * full, rel=1e-12)
        P["ism_thermal_share"].value = "off"
        off = (S.p_ext(r, 0.5, P),) + _forces_P_ext(P, r, 0.5)
        P["ism_thermal_share"].value = "fit"
        fit = (S.p_ext(r, 0.5, P),) + _forces_P_ext(P, r, 0.5)
        for a, b in zip(off, fit):
            assert a - b == pytest.approx(sub, rel=1e-9)
    for fn in _SITES:                    # by source: the three ODE sites are checked by value in the next test
        assert "get_press_ism_thermal(" in inspect.getsource(fn), f"{fn.__module__}.{fn.__qualname__}"


def test_the_ode_right_hand_sides_subtract_it_by_value(tmp_path, monkeypatch):
    """The sites that move the shell: phase 1a's RHS (also c3c 1b/1c's), its derived P_ext, and the c3c momentum RHS.
    With the shell mass frozen (isCollapse), fit - off in dv/dt is exactly 4 pi R2^2 P_th0 w / M; a wrong sign,
    fabs or radius at a site fails here (a source check would not see it)."""
    import dataclasses
    import numpy as np
    P = _params(tmp_path, PISM=1e6)
    rc = P["rCloud"].value
    r, R2 = 2.0 * rc, 1.8 * rc
    sub = get_press_ism_thermal(r, 0.5, P)
    assert sub > 0.0
    fb = SimpleNamespace(Lmech_total=P["Lmech_total"].value, v_mech_total=P["v_mech_total"].value)
    monkeypatch.setattr(ODEs, "get_current_sps_feedback", lambda t, p: fb)
    monkeypatch.setattr(MOM, "get_current_sps_feedback", lambda t, p: fb)
    mS = S._scalar(mass_profile.get_mass_profile(r, P))
    sh = SimpleNamespace(rShell=r, shell_fAbsorbedIon=0.5, n_IF=0.0, R_IF=0.0, isDissolved=True)
    snap = ODEs.create_ODE_snapshot(P, sh)
    snap = dataclasses.replace(snap, shell_fAbsorbedIon=0.5, rShell=r, shell_mass=mS, isCollapse=True,
                               phii_scheme="c3c", P_HII=0.0, coverFraction=1.0)
    msnap = MOM.create_momentum_snapshot(P, sh, mS, 0.0)
    y3, y2 = [R2, 10.0, 1e7], np.array([R2, 10.0])

    def rhs():
        return (ODEs.get_ODE_Edot_pure(0.1, y3, snap, P)[1],
                ODEs.compute_derived_quantities(0.1, y3, snap, P).press_HII_in,
                MOM.get_ODE_momentum_pure(0.1, y2, msnap, P)[1])

    P["ism_thermal_share"].value = "off"
    off = rhs()
    P["ism_thermal_share"].value = "fit"
    fit = rhs()
    dv = 4.0 * np.pi * R2 ** 2 * sub / mS
    assert fit[0] - off[0] == pytest.approx(dv, rel=1e-6)
    assert off[1] - fit[1] == pytest.approx(sub, rel=1e-9)
    assert fit[2] - off[2] == pytest.approx(dv, rel=1e-6)


def test_the_cap_keeps_a_hand_typed_hot_ambient_above_p_ism(tmp_path):
    """Rosette's PISM 1e6 / nISM 0.1 (both typed, used as given): the share would take 7.5e4 K cm^-3, more than
    P_i = 1760; the cap, the ambient as neutral gas at T_i, takes 1.1 x 0.1 x 8000 = 880 K cm^-3."""
    P = _params(tmp_path, PISM=1e6, nISM=0.1)
    r = 2.0 * P["rCloud"].value
    assert get_press_ism_thermal(r, 0.5, P) / _pk(P) == pytest.approx(880.0, rel=1e-9)
    # inside the bridge the cap is still the ambient's own nISM (times w), not P_i(r)/2 on the blended cloud density
    rb = 1.05 * P["rCloud"].value
    want = density_profile.ism_weight(rb, P["rCloud"].value) * 880.0
    assert get_press_ism_thermal(rb, 0.5, P) / _pk(P) == pytest.approx(want, rel=1e-9)
    pism = P["PISM"].value * P["k_B"].value
    for got in (S.p_ext(r, 0.5, P),) + _forces_P_ext(P, r, 0.5):
        assert got >= pism
        assert got / _pk(P) == pytest.approx(1e6 + 1760.0 - 880.0, rel=1e-9)


@pytest.mark.parametrize("bad", ["Fit", "ok22", "on"])
def test_unknown_tokens_are_refused(tmp_path, bad):
    with pytest.raises(ParameterFileError, match="ism_thermal_share"):
        _params(tmp_path, ism_thermal_share=bad)
