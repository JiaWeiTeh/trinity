"""The cloud gas's own pressure inside the cloud (TCloud thermal, sigmaCloud turbulent; both 0 by default).

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

N_CORE = 1e4             # cm^-3: a uniform 257 Msun cloud, rCloud ~ 0.56 pc
R_IN, R_OUT = 0.3, 2.0   # pc: inside and outside the cloud


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


def test_defaults_give_no_cloud_pressure(tmp_path):
    P = _params(tmp_path)
    assert P["TCloud"].value == 0.0 and P["sigmaCloud"].value == 0.0
    assert density_profile.get_press_cloud(R_IN, 1.0, P) == 0.0
    assert S.p_ext(R_IN, 1.0, P) == 0.0                      # inside a cold cloud, as before


def test_thermal_term_is_molecular_gas_at_TCloud(tmp_path):
    P = _params(tmp_path, TCloud=10)
    got = density_profile.get_press_cloud(R_IN, 1.0, P)
    mu_ratio = 1.4 / (14 / 6)                                # mu_convert / mu_mol at x_He = 0.1: 0.6
    want = mu_ratio * N_CORE * cvt.CGS.k_B * 10.0 * cvt.Pb_cgs2au
    assert got == pytest.approx(want, rel=1e-9)
    assert S.p_ext(R_IN, 1.0, P) == pytest.approx(got, rel=1e-12)


def test_thermal_term_off_while_photons_escape_turbulent_term_stays(tmp_path):
    P = _params(tmp_path, TCloud=10, sigmaCloud=2)
    rho_cgs = 1.4 * cvt.CGS.m_H * N_CORE
    turb = rho_cgs * (2e5) ** 2 * cvt.Pb_cgs2au                # rho sigma^2, sigma = 2 km/s
    assert density_profile.get_press_cloud(R_IN, 0.5, P) == pytest.approx(turb, rel=1e-6)
    therm = 0.6 * N_CORE * cvt.CGS.k_B * 10.0 * cvt.Pb_cgs2au
    assert density_profile.get_press_cloud(R_IN, 1.0, P) == pytest.approx(turb + therm, rel=1e-6)


def test_nothing_outside_the_cloud(tmp_path):
    P = _params(tmp_path, TCloud=10, sigmaCloud=2)
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
    P = _params(tmp_path, TCloud=10, sigmaCloud=2)
    rho = P["mu_convert"].value * density_profile.get_density_profile(R_IN, P)
    therm = density_profile.get_press_cloud(R_IN, 1.0, P, turbulent=False)
    assert S.p_ext(R_IN, 1.0, P) > therm                                    # the force sees the turbulent term
    u = 1.0
    assert S.edge_speed(u, R_IN, 1.0, P) == pytest.approx(0.5 * (u + math.sqrt(u * u + 4 * therm / rho)), rel=1e-12)
