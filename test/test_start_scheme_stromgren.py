"""start_scheme 'weaver' | 'stromgren' (docs/dev/phii-identity/weak-wind/PLAN.md, Phase 5; rulings D6, D12, D15, D17).

The cloud is the weak-wind test cloud of the plan: NU Ori's ionising output (Q = 10^47.2 s^-1, Vink x0.05 wind at
v_inf = 2666 km/s, written here as an SPS table) in a uniform 1e4 cm^-3 core, mCluster = sps_refmass = 16.4 Msun.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
import trinity.phase0_init.get_InitPhaseParam as IP
from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param
from trinity._input.registry import SPECS, _validate_start_scheme
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp
from trinity.sps import read_sps

REPO = Path(__file__).resolve().parents[1]
Q_NUORI, LBOL, LI, LN = 1.584893e47, 1.030320e38, 4.174574e36, 9.885738e37      # s^-1, erg/s
LW_X005, PDOT_X005 = 1.184922e33, 8.890495e24                                     # erg/s, g cm s^-2 (Vink x0.05)


def _write_cloud(tmp, name="ww", eps=0.05, radiation=True, **over):
    """A .param + SPS table for the test cloud; eps scales the wind (x Vink) at fixed v_inf; radiation=False takes
    radiation pressure out as the benchmarks do (Li x 1e-20, Ln = 0, Lbol = Li, dust off); over replaces keys."""
    k = eps / 0.05
    li, ln, lb = (LI, LN, LBOL) if radiation else (LI * 1e-20, 0.0, LI * 1e-20)
    sps = tmp / f"{name}_sps.csv"
    sps.write_text("t,Qi,Lbol,Li,Ln,Lmech_W,pdot_W,Lmech_SN\n" + "".join(
        f"{t:.6e},{Q_NUORI:.6e},{lb:.6e},{li:.6e},{ln:.6e},{LW_X005 * k:.6e},{PDOT_X005 * k:.6e},0\n"
        for t in (0.0, 5e4, 1e5, 2e5, 3e5, 5e5)))                      # the M43 tables' time grid (yr)
    keys = {"model_name": name, "mCloud": "257.4", "sfe": f"{16.4 / 257.4:.10g}", "dens_profile": "densPL",
            "densPL_alpha": "0", "rCore": "0.03", "nCore": "1e4", "nISM": "100", "PISM": "1e4", "TShell_neu": "10",
            "phii_scheme": "swept", "include_PHII": "True", "log_console": "False", "path2output": str(tmp / "out"),
            "sps_path": str(sps), "sps_refmass": "16.4"}
    if not radiation:
        keys["dust_sigma"] = "0"
    keys.update({kk: str(v) for kk, v in over.items()})
    cols = ["sps_col_t t yr linear", "sps_col_Qi Qi cgs linear", "sps_col_Lbol Lbol cgs linear", "sps_col_Li Li cgs linear",
            "sps_col_Ln Ln cgs linear", "sps_col_Lmech_W Lmech_W cgs linear", "sps_col_pdot_W pdot_W cgs linear",
            "sps_col_Lmech_SN Lmech_SN cgs linear"]
    par = tmp / f"{name}.param"
    par.write_text("\n".join([f"{kk} {v}" for kk, v in keys.items()] + cols + [""]))
    return par


def _params(tmp, nCore=1e4, **kw):
    """read_param + main.py steps 1-2 (cloud, SPS), as start_expansion does."""
    P = read_param(str(_write_cloud(tmp, nCore=nCore, **kw)))
    get_InitCloudProp(P)
    sps = read_sps.read_sps(P["mCluster"].value / P["sps_refmass"].value, P)
    P["sps_data"].value = sps
    P["sps_f"].value = read_sps.get_interpolation(sps)
    return P


def _uniform_cgs(n, T=8000.0):
    """R_St [pc] and c_i [km/s] by an independent cgs route (x_He 0.1, He singly ionised in the shell)."""
    mu_i, chi = 1.4 / 2.2, 1.1
    aB = 2.59e-13 * (T / 1e4) ** -0.83
    R = (3 * Q_NUORI / (4 * math.pi * chi * aB * n * n)) ** (1 / 3) / cvt.pc2cm
    c = math.sqrt(1.380649e-16 * T / (mu_i * 1.6735575e-24)) / 1e5
    return R, c


# --- the key -----------------------------------------------------------------------------------------------------

def test_validator_accepts_the_two_starts_and_rejects_others():
    for ok in ("weaver", "stromgren"):
        _validate_start_scheme(ok, None)
    for bad in ("auto", "WEAVER", "spitzer", "", "stroemgren"):          # no 'auto' until the zeta ladders (D17)
        with pytest.raises(ParameterFileError):
            _validate_start_scheme(bad, None)


def test_registry_default_is_weaver():
    spec = next(s for s in SPECS if s.name == "start_scheme")
    assert spec.default == "weaver" and spec.validator is not None and spec.run_const is True


def test_phi_ceiling_matches_every_shipped_cube():
    cubes = sorted((REPO / "lib" / "default" / "opiate").glob("*_cube.npy"))
    assert cubes
    for c in cubes:
        log_phi = np.load(c, allow_pickle=True)[2]
        assert 10.0 ** float(np.max(log_phi)) == pytest.approx(IP.NONCIE_PHI_MAX_CGS, rel=1e-12), c.name


# --- D6: the Weaver start refuses below the phi ceiling --------------------------------------------------------------

def test_weaver_start_refuses_below_the_phi_ceiling(tmp_path):
    # NU Ori's Q at Vink x0.05 in a 2e6 core: phi(r0) ~ 2e21 > 1e21 (PLAN section 1, L1: the limit is 1.01e6)
    with pytest.raises(ValueError, match="start-up limit"):
        IP.get_y0(_params(tmp_path, nCore=2e6))


def test_weaver_start_unchanged_above_the_ceiling(tmp_path):
    t0, r0, v0, E0, T0 = IP.get_y0(_params(tmp_path, nCore=1e4))
    assert r0 > 1.151e-6 and v0 * cvt.v_au2kms == pytest.approx(2 * LW_X005 / PDOT_X005 / 1e5, rel=1e-9)


# --- the Stroemgren state -------------------------------------------------------------------------------------------

def test_stromgren_state_matches_independent_closed_forms(tmp_path):
    P = _params(tmp_path, nCore=1e4)
    zMD, zED, R_St, R_eq, c_i = IP.stromgren_zeta(P)
    R_ref, c_ref = _uniform_cgs(1e4)
    assert R_St == pytest.approx(R_ref, rel=1e-6)
    assert c_i * cvt.v_au2kms == pytest.approx(c_ref, rel=1e-4)
    assert zMD == pytest.approx(0.05222, rel=1e-3) and zED / zMD == pytest.approx(7.934, rel=1e-3)
    t_R, R_R, t_rec, beta = IP.rtype_duration(R_St, c_i, P)
    assert t_rec * 1e6 == pytest.approx(9.242, rel=1e-3) and t_R * 1e6 == pytest.approx(37.63, rel=1e-3)
    assert 0.99 < R_R / R_St < 1.0
    t0, rs, u0, Ms, R2g = IP.get_y0_stromgren(P)
    assert rs == R_St and u0 == pytest.approx(2 * c_i / math.sqrt(3), rel=1e-12) and R2g == R_eq
    assert t0 == pytest.approx(P["tSF"].value + t_R, rel=1e-12)
    rho = P["mu_convert"].value * P["nCore"].value
    assert Ms == pytest.approx(4 * math.pi / 3 * rho * R_St ** 3, rel=1e-6)


def test_stromgren_radius_on_a_power_law_envelope(tmp_path):
    """D15: n = nCore (r/rCore)^-1 outside rCore makes the recombination integral linear in R, so
    R_St = rCore + (R_uni^3 - rCore^3) / (3 rCore^2) exactly; the uniform-core formula would give R_uni."""
    P = _params(tmp_path, densPL_alpha=-1, rCore=0.02)
    Qi = float(P["sps_f"].value["fQi"](P["tSF"].value))
    R_uni = (3 * Qi / (4 * math.pi * P["chi_e_shell"].value * P["caseB_alpha"].value * P["nCore"].value ** 2)) ** (1 / 3)
    rc = P["rCore"].value
    assert R_uni > rc
    assert IP.stromgren_radius(P, Qi) == pytest.approx(rc + (R_uni ** 3 - rc ** 3) / (3 * rc ** 2), rel=1e-6)


def test_stromgren_refuses_when_the_sphere_leaves_the_cloud(tmp_path):
    # an M43-like small clump (20 Msun, M* 16.4) at 100 cm^-3: R_St ~ 0.72 pc > rCloud ~ 0.63 pc (D15)
    P = _params(tmp_path, mCloud=20.0, sfe=0.82, nCore=100, nISM=10, PISM=1e3)
    with pytest.raises(ValueError, match="exceeds rCloud"):
        IP.get_y0_stromgren(P)


def test_stromgren_refuses_a_strong_wind_and_a_c3c_run(tmp_path):
    with pytest.raises(ValueError, match="does not apply"):             # Vink x20: zeta_MD ~ 1.04
        IP.get_y0_stromgren(_params(tmp_path, eps=20.0, name="strong"))
    with pytest.raises(ValueError, match="needs phii_scheme 'swept'"):
        IP.get_y0_stromgren(_params(tmp_path, phii_scheme="c3c", name="c3c"))


# --- the start check (weak-wind/start-selection/PLAN.md, Phase 1) -----------------------------------------------------

def test_start_check_on_the_test_cloud(tmp_path):
    P = _params(tmp_path, nCore=1e4)
    zMD, zED, zbar, x, y, phi, why_w, why_s = IP.start_check(P)
    assert zMD == pytest.approx(0.05222, rel=1e-3) and phi / 1e21 == pytest.approx(0.00989, rel=1e-3)
    assert why_w is None and why_s is None
    assert x == pytest.approx(zMD ** 2, rel=1e-9)          # R_St inside the flat core: F_w/F_g = zeta_MD^2
    assert zED / zMD == pytest.approx(7.934, rel=1e-3)     # fixed v_inf: [sqrt(7) v_inf / (11 c_i)]^(1/2)
    assert zbar == pytest.approx(zMD, rel=1e-2)            # uniform cloud: the two conventions agree (edge smoothing aside)
    assert phi == IP._phi_cgs(IP.get_y0(P)[1], float(P["sps_f"].value["fQi"](P["tSF"].value)))   # get_y0's r0, bit for bit


def test_start_check_names_an_input_with_no_admissible_start(tmp_path):
    P = _params(tmp_path, nCore=2e6, phii_scheme="c3c", name="nostart")
    *_, why_w, why_s = IP.start_check(P)
    assert why_w is not None and why_s is not None
    with pytest.raises(ValueError, match="no admissible start"):
        IP.get_y0(P)


# --- a phase-2-only run (D12's segment control included) -------------------------------------------------------------

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wwstart")
    par = _write_cloud(tmp, name="wwstart", radiation=False, start_scheme="stromgren", stop_t=0.012, stop_r=10)
    res = subprocess.run([sys.executable, str(REPO / "run.py"), str(par)], cwd=tmp, capture_output=True, text=True,
                         timeout=1800)
    assert res.returncode == 0, res.stderr[-3000:]
    rows = [json.loads(ln) for ln in next((tmp / "out").rglob("dictionary.jsonl")).read_text().splitlines()
            if ln.strip()]
    return rows, _params(tmp, name="ref", radiation=False)


def test_stromgren_run_is_momentum_only_from_the_closed_form_state(run):
    rows, P = run
    zMD, zED, R_St, R_eq, c_i = IP.stromgren_zeta(P)
    t_R = IP.rtype_duration(R_St, c_i, P)[0]
    assert rows and all(r["current_phase"] == "momentum" for r in rows)
    first = rows[0]
    assert first["t_now"] == pytest.approx(P["tSF"].value + t_R, rel=1e-9)
    assert first["rs_dyn"] == pytest.approx(R_St, rel=1e-9)
    assert first["v2"] == pytest.approx(2 * c_i / math.sqrt(3), rel=1e-9)
    assert first["R2"] < 0.2 * first["rs_dyn"]


def test_stromgren_segments_follow_the_dynamical_time(run):
    rows, _ = run
    t = np.array([r["t_now"] for r in rows])
    rs, u = np.array([r["rs_dyn"] for r in rows]), np.array([r["v2"] for r in rows])
    dt = np.diff(t)
    import trinity.phase2_momentum.swept as S
    assert dt[0] <= S.STROMGREN_DT_FIRST * rs[0] / abs(u[0]) * (1 + 1e-9)
    assert np.all(dt[1:] <= S.STROMGREN_DT_FRAC * rs[1:-1] / np.abs(u[1:-1]) * (1 + 1e-9))


def test_stromgren_run_follows_hosokawa_inutsuka(run):
    """The outer edge against HI06 with radiation pressure off (gravity, the ambient pressure and the 10 K rind on):
    G-WW3's 3 % bar; the gates proper are G-WW1/G-WW3 on Helix."""
    rows, P = run
    zMD, zED, R_St, R_eq, c_i = IP.stromgren_zeta(P)
    t0 = rows[0]["t_now"]
    for r in rows:
        x = r["rs_dyn"] / R_St
        if 1.2 <= x <= 3.0:
            R_HI = R_St * (1 + 7 / 4 * math.sqrt(4 / 3) * c_i * (r["t_now"] - t0) / R_St) ** (4 / 7)
            assert r["rs_dyn"] / R_HI == pytest.approx(1.0, abs=0.03)


def _gravity_reference(t, R_St, c_i, rho0, G, M_star):
    """Hosokawa & Inutsuka's equation of motion plus the cluster and shell gravity (the swept runner's term), from
    (R_St, 2 c_i/sqrt 3): d(M u)/dt = 4 pi r^2 rho0 c_i^2 (R_St/r)^(3/2) - G M (M_* + M/2)/r^2, M = (4 pi/3) rho0 r^3."""
    from scipy.integrate import solve_ivp

    def rhs(_, y):
        r, p = y
        M = 4 * math.pi / 3 * rho0 * r ** 3
        return [p / M, 4 * math.pi * r ** 2 * rho0 * c_i ** 2 * (R_St / r) ** 1.5 - G * M * (M_star + M / 2) / r ** 2]

    M0 = 4 * math.pi / 3 * rho0 * R_St ** 3
    return solve_ivp(rhs, (0.0, float(t[-1])), [R_St, M0 * 2 * c_i / math.sqrt(3)], t_eval=t, rtol=1e-11,
                     atol=1e-16 * M0, method="DOP853").y[0]


def test_stromgren_dense_core_follows_hi06_with_gravity(tmp_path):
    """The regime the start exists for (the M43 cores that lose phase 1a): 2e6 cm^-3, R_St ~ 1e-3 pc, where the
    D12 segments (~1e-7 Myr first) are shorter than LSODA's ODE_MIN_STEP and the star's gravity is first order
    (G M_*/(R_St c_i^2) ~ 0.7; HI06 alone is 15 % off). Measured 2026-10-28 on Helix: 3.9e-4 against this reference."""
    par = _write_cloud(tmp_path, name="dense", radiation=False, nCore="2e6", start_scheme="stromgren", stop_t=2e-3,
                       stop_r=10)
    res = subprocess.run([sys.executable, str(REPO / "run.py"), str(par)], cwd=tmp_path, capture_output=True,
                         text=True, timeout=1800)
    assert res.returncode == 0, res.stderr[-3000:]
    rows = [json.loads(ln) for ln in next((tmp_path / "out").rglob("dictionary.jsonl")).read_text().splitlines()
            if ln.strip()]
    P = _params(tmp_path, nCore=2e6, name="denseref", radiation=False)
    zMD, zED, R_St, R_eq, c_i = IP.stromgren_zeta(P)
    t = np.array([r["t_now"] - rows[0]["t_now"] for r in rows])
    rs = np.array([r["rs_dyn"] for r in rows])
    ref = _gravity_reference(t, R_St, c_i, P["mu_convert"].value * P["nCore"].value, P["G"].value,
                             P["mCluster"].value)
    w = (rs / R_St >= 1.5)
    assert w.sum() > 20 and rs[-1] / R_St > 5
    assert np.max(np.abs(rs[w] / ref[w] - 1)) < 3e-3
