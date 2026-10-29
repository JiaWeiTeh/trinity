"""The cloud gas's own pressure inside the cloud (TCloud thermal, default 10 K since 2026-10-29; alphaVirCloud turbulent, default 0).

density_profile.get_press_cloud is added after the photoionised and PISM terms at every place the outside pressure is
built (docs/dev/phii-identity/weak-wind/PLAN.md section 11). Checked here against independent cgs arithmetic.
"""
import math

import pytest

import trinity._functions.unit_conversions as cvt
import trinity.phase2_momentum.swept as S
from trinity._input.read_param import read_param
from trinity.cloud_properties import density_profile
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp

N_CORE = 1e4             # cm^-3: a uniform cloud, 241 Msun of gas after star formation (257.4 input), rCloud ~ 0.55 pc
R_IN, R_OUT = 0.3, 2.0   # pc: inside and outside the cloud
G_CGS = 6.6743e-8


def _turb_cgs(alpha_vir, r_pc):
    """rho sigma^2 [cgs] in the uniform cloud: sigma^2 = alpha G M(<r)/(5 r), M(<r) = (4 pi/3) rho r^3 (gas only)."""
    rho = 1.4 * cvt.CGS.m_H * N_CORE
    r = r_pc * cvt.pc2cm
    return rho * alpha_vir * G_CGS * (4 * math.pi / 3) * rho * r ** 3 / (5 * r)


def _params(tmp_path, **over):
    keys = {"model_name": "cp", "mCloud": "257.4", "sfe": f"{16.4 / 257.4:.10g}", "dens_profile": "densPL",
            "densPL_alpha": "0", "rCore": "0.03", "nCore": f"{N_CORE:g}", "nISM": "100", "PISM": "1e4",
            "log_console": "False", "path2output": str(tmp_path / "out")}
    keys.update({k: str(v) for k, v in over.items()})
    par = tmp_path / "cp.param"
    par.write_text("\n".join(f"{k} {v}" for k, v in keys.items()) + "\n")
    P = read_param(str(par))
    get_InitCloudProp(P)
    return P


def test_defaults_give_the_10K_thermal_term_and_no_turbulence(tmp_path):
    P = _params(tmp_path)
    assert P["TCloud"].value == 10.0 and P["alphaVirCloud"].value == 0.0
    want = 0.6 * N_CORE * cvt.CGS.k_B * 10.0 * cvt.Pb_cgs2au                    # (mu_convert/mu_mol) n k_B 10 K
    assert density_profile.get_press_cloud(R_IN, 1.0, P) == pytest.approx(want, rel=1e-9)
    assert density_profile.get_press_cloud(R_IN, 0.5, P) == 0.0                # photons escape: the photoionised term
    P0 = _params(tmp_path, TCloud=0)                                            # switched off: as before 2026-10-29
    assert density_profile.get_press_cloud(R_IN, 1.0, P0) == 0.0
    assert S.p_ext(R_IN, 1.0, P0) == 0.0


def test_thermal_term_is_molecular_gas_at_TCloud(tmp_path):
    P = _params(tmp_path, TCloud=10)
    got = density_profile.get_press_cloud(R_IN, 1.0, P)
    mu_ratio = 1.4 / (14 / 6)                                # mu_convert / mu_mol at x_He = 0.1: 0.6
    want = mu_ratio * N_CORE * cvt.CGS.k_B * 10.0 * cvt.Pb_cgs2au
    assert got == pytest.approx(want, rel=1e-9)
    assert S.p_ext(R_IN, 1.0, P) == pytest.approx(got, rel=1e-12)


def test_virial_term_is_rho_sigma_sq_minus_the_thermal_part(tmp_path):
    """Bertoldi & McKee's sigma includes the thermal motion: the cloud gas totals rho sigma^2 while photons are all
    absorbed, and keeps the non-thermal part rho sigma^2 - P_th when they escape (the photoionised term then applies)."""
    P = _params(tmp_path, TCloud=10, alphaVirCloud=2)
    turb = _turb_cgs(2, R_IN) * cvt.Pb_cgs2au                   # sigma ~ 0.47 km/s at 0.3 pc
    therm = 0.6 * N_CORE * cvt.CGS.k_B * 10.0 * cvt.Pb_cgs2au     # c_th ~ 0.19 km/s
    assert turb > 2 * therm
    assert density_profile.get_press_cloud(R_IN, 1.0, P) == pytest.approx(turb, rel=1e-6)
    assert density_profile.get_press_cloud(R_IN, 0.5, P) == pytest.approx(turb - therm, rel=1e-6)
    P0 = _params(tmp_path, TCloud=0, alphaVirCloud=2)               # no thermal term: the virial term alone
    assert density_profile.get_press_cloud(R_IN, 0.5, P0) == pytest.approx(turb, rel=1e-6)


def test_virial_term_grows_as_r_squared_in_a_uniform_cloud(tmp_path):
    """sigma^2 ~ M(<r)/r ~ r^2 with the gas mass alone; the cluster's mass would make it grow toward the centre."""
    P = _params(tmp_path, TCloud=0, alphaVirCloud=1)
    r1, r2 = 0.1, 0.4
    ratio = density_profile.get_press_cloud(r2, 0.5, P) / density_profile.get_press_cloud(r1, 0.5, P)
    assert ratio == pytest.approx((r2 / r1) ** 2, rel=1e-6)


def test_subthermal_virial_dispersion_adds_nothing(tmp_path):
    P = _params(tmp_path, TCloud=10, alphaVirCloud=0.01)
    therm = 0.6 * N_CORE * cvt.CGS.k_B * 10.0 * cvt.Pb_cgs2au
    assert _turb_cgs(0.01, R_IN) * cvt.Pb_cgs2au < therm
    assert density_profile.get_press_cloud(R_IN, 1.0, P) == pytest.approx(therm, rel=1e-9)
    assert density_profile.get_press_cloud(R_IN, 0.5, P) == 0.0


def test_nothing_outside_the_cloud(tmp_path):
    P = _params(tmp_path, TCloud=10, alphaVirCloud=2)
    assert R_OUT > P["rCloud"].value
    assert density_profile.get_press_cloud(R_OUT, 1.0, P) == 0.0
    assert S.p_ext(R_OUT, 1.0, P) == pytest.approx(1e4 * cvt.CGS.k_B * cvt.Pb_cgs2au, rel=1e-9)   # PISM alone


def test_edge_speed_uses_the_cloud_gas_sound_speed(tmp_path):
    P = _params(tmp_path, TCloud=10)
    c0_sq = S.p_ext(R_IN, 1.0, P) / (P["mu_convert"].value * density_profile.get_density_profile(R_IN, P))
    c0_cgs = math.sqrt(cvt.CGS.k_B * 10.0 / ((14 / 6) * cvt.CGS.m_H))                       # 0.19 km/s
    assert math.sqrt(c0_sq) == pytest.approx(c0_cgs / 1e5 * cvt.v_kms2au, rel=1e-6)
    u = 1.0                                                                                  # pc/Myr
    assert S.edge_speed(u, R_IN, 1.0, P) == pytest.approx(0.5 * (u + math.sqrt(u * u + 4 * c0_sq)), rel=1e-12)


def test_edge_speed_ignores_the_turbulent_term(tmp_path):
    """The isothermal jump uses the thermal sound speed only; the force balance (p_ext) carries rho sigma^2 too."""
    P = _params(tmp_path, TCloud=10, alphaVirCloud=2)
    rho = P["mu_convert"].value * density_profile.get_density_profile(R_IN, P)
    therm = density_profile.get_press_cloud(R_IN, 1.0, P, turbulent=False)
    assert S.p_ext(R_IN, 1.0, P) > therm                                    # the force sees the turbulent term
    u = 1.0
    assert S.edge_speed(u, R_IN, 1.0, P) == pytest.approx(0.5 * (u + math.sqrt(u * u + 4 * therm / rho)), rel=1e-12)
