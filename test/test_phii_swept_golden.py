#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phii_scheme 'swept' in the momentum phase (docs/dev/phii-identity PLAN.md 0.0.11, Step 4; 0.0.11.10 T2).

Unit checks on trinity/phase2_momentum/swept.py, on simple_cluster's own parameters:
  * the outer edge's speed (ruling D-T2 (b)): one speed in a cold cloud and for u <= 0; the isothermal jump at the
    gas ahead where 0 < c0 <= c_i, which conserves the shock-frame mass and momentum fluxes; one speed again
    when the gas ahead is hotter than photoionised gas;
  * the mass rule (D-T1): never decreasing, frozen while u <= 0;
  * the shell identity 4 pi R2^2 P_ram + K_s + F_rad,sh = 4 pi r_s^2 P(r_s) (L8) and the R2 root-find, which must
    recover the R2 that produced a given r_s, inside and outside the cloud.
Goldens from momentum-phase replays of two archived runs (harness/momentum_replay.py; skipped where the gitignored
docs/dev is absent): ROS for 0.05 Myr inside its cloud, and B3MW1 to 3 Myr, whose shell leaks its ionising photons
from 2.59 Myr and then runs supersonically into photoionised gas -- the one regime where the sweep speed matters.
Every momentum snapshot must close L8 to 1e-6 and keep rs_dyn on the shell solve's rShell.
"""
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import trinity._functions.unit_conversions as cvt
import trinity.phase2_momentum.swept as S
from trinity._input.read_param import read_param
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "docs" / "dev" / "phii-identity"


@pytest.fixture(scope="module")
def params():
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    get_InitCloudProp(p)
    # a plausible young cluster: Q = 1e51 s^-1, L_bol = 3e40 erg/s, L_mech = 3e38 erg/s at v_w = 2000 km/s
    p["Qi"].value = 1e51 / cvt.s2Myr
    p["Li"].value, p["Ln"].value, p["Lbol"].value = (x * cvt.L_cgs2au for x in (1e40, 2e40, 3e40))
    return p


def _c_i_sq(p):
    return p["k_B"].value * p["TShell_ion"].value / p["mu_ion_shell"].value


def test_edge_speed_one_speed_in_a_cold_cloud_and_when_falling_back(params):
    r = 0.5 * params["rCloud"].value
    for u in (0.3, 3.0, 30.0):
        assert S.edge_speed(u, r, 1.0, params) == u                      # photons held, P_ext = 0
    for u in (0.0, -2.0):
        assert S.edge_speed(u, r, 0.5, params) == u                      # D-T1


def test_edge_speed_is_the_isothermal_jump_when_photons_escape_inside_the_cloud(params):
    r, u = 0.5 * params["rCloud"].value, 20.0
    c0_sq = _c_i_sq(params)                                              # P_ext / rho = k T_ion / mu_ion_shell
    U = S.edge_speed(u, r, 0.5, params)
    assert U == pytest.approx(0.5 * (u + math.sqrt(u * u + 4 * c0_sq)), rel=1e-12)
    rho1 = U / (U - u)                                                   # rho0 = 1: mass flux
    assert (rho1 * ((U - u) ** 2 + c0_sq)) / (U ** 2 + c0_sq) == pytest.approx(1.0, rel=1e-12)   # momentum flux


def test_edge_speed_past_the_cloud_follows_the_ambient(params):
    r, u = 2.0 * params["rCloud"].value, 20.0
    rho = params["mu_convert"].value * S._scalar(S.density_profile.get_density_profile(S.np.array([r]), params))
    kB, saved = params["k_B"].value, params["PISM"].value
    try:
        params["PISM"].value = 0.25 * _c_i_sq(params) * rho / kB          # warm ambient, c0 = c_i / 2
        assert S.edge_speed(u, r, 1.0, params) == pytest.approx(0.5 * (u + math.sqrt(u * u + _c_i_sq(params))),
                                                                rel=1e-12)
        params["PISM"].value = 4.0 * _c_i_sq(params) * rho / kB           # hotter than photoionised gas
        assert S.edge_speed(u, r, 1.0, params) == u
    finally:
        params["PISM"].value = saved


def test_mass_never_decreases_and_freezes_while_falling_back(params):
    r = 0.5 * params["rCloud"].value
    m_r = S._scalar(S.mass_profile.get_mass_profile(r, params))
    assert S.swept_mass(r, 1.0, 2 * m_r, False, params) == (2 * m_r, 0.0)
    assert S.swept_mass(r, 1.0, 0.5 * m_r, True, params) == (0.5 * m_r, 0.0)
    m, md = S.swept_mass(r, 1.0, 0.5 * m_r, False, params)
    assert m == pytest.approx(m_r) and md > 0


@pytest.mark.parametrize("R2", [1.0, 5.0], ids=["inside_cloud", "outside_cloud"])
def test_root_recovers_R2_and_the_shell_identity_closes(params, R2):
    fb = SimpleNamespace(Lmech_total=3e38 * cvt.L_cgs2au, v_mech_total=2000 * cvt.v_kms2au, Qi=params["Qi"].value)
    Ms = S._scalar(S.mass_profile.get_mass_profile(1.4 * R2, params))
    sp, Pc = S._shell_at(R2, 1.4 * R2, Ms, fb, params)
    root = S.root_R2(float(sp.rShell), Ms, fb, params)
    assert root["ok"], root.get("why")
    assert root["R2"] == pytest.approx(R2, rel=1e-8)
    assert abs(root["resid"]) < 1e-9
    assert abs(root["L8"]) < 1e-6
    # the bracket from a guess (Step 4c) finds the same root, and the same guess finds it bit-identically -- the
    # runner's reconciliation after a broken-off segment relies on that to dedupe its snapshot
    g1 = S.root_R2(float(sp.rShell), Ms, fb, params, guess=1.3 * R2)
    g2 = S.root_R2(float(sp.rShell), Ms, fb, params, guess=1.3 * R2)
    assert g1["R2"] == pytest.approx(R2, rel=1e-8)
    assert g1["R2"] == g2["R2"]


def test_swept_runner_invariants_on_simple_cluster(tmp_path):
    """The runner itself, with no docs/dev: a momentum-entry state inside simple_cluster's cloud, 0.02 Myr.

    Every snapshot must close the shell identity, keep rs_dyn on rShell, never lose mass, sweep at
    4 pi r_s^2 rho U, carry the c3c gravity, and report the injected wind+SN momentum as F_ram_wind + F_ram_SN;
    and between snapshots the momentum M_s u must change by the impulse of the drive (the segment-held K_s and
    F_rad, the live terms trapezoided).
    """
    import re
    import trinity.phase2_momentum.run_momentum_phase as MP
    from trinity.sps import read_sps
    txt = re.sub(r"^(path2output\s+).*$", lambda m: m.group(1) + str(tmp_path),
                 (REPO / "param" / "simple_cluster.param").read_text(), flags=re.M)
    if "path2output" not in txt:
        txt += f"\npath2output    {tmp_path}\n"
    (tmp_path / "sc.param").write_text(txt)
    p = read_param(str(tmp_path / "sc.param"))
    get_InitCloudProp(p)
    sps = read_sps.read_sps(p["mCluster"].value / p["sps_refmass"].value, p)
    p["sps_data"].value, p["sps_f"].value = sps, read_sps.get_interpolation(sps)
    g = lambda k: p[k].value  # noqa: E731
    R2 = 0.6 * g("rCloud")                                   # inside the cloud, 10 km/s, 0.5 Myr
    p["t_now"].value, p["stop_t"].value, p["current_phase"].value = 0.5, 0.52, "momentum"
    p["R2"].value, p["v2"].value, p["rShell"].value = R2, 10.0 * cvt.v_kms2au, 1.2 * R2
    p["shell_mass"].value = S._scalar(S.mass_profile.get_mass_profile(R2, p))
    p["phii_scheme"].value = "swept"
    MP.run_phase_momentum(p)
    p.flush()
    rows = [json.loads(ln) for f in tmp_path.rglob("dictionary.jsonl") for ln in f.read_text().splitlines()
            if ln.strip()]
    mom = [r for r in rows if r["current_phase"] == "momentum"]
    assert len(mom) >= 4
    for r in mom:
        rs = r["rs_dyn"]
        outer = 4 * math.pi * r["rShell"] ** 2 * r["P_outer"]
        assert abs((4 * math.pi * r["R2"] ** 2 * r["Pb"] + r["K_s"] + r["F_rad_sh"] - outer) / outer) < 1e-6
        assert abs(rs / r["rShell"] - 1) < 1e-9
        rho = g("mu_convert") * S._scalar(S.density_profile.get_density_profile(S.np.array([rs]), p))
        assert r["shell_massDot"] == pytest.approx(4 * math.pi * rs ** 2 * rho * r["rs_dot"], rel=1e-9)
        assert r["F_grav"] == pytest.approx(g("G") * r["shell_mass"] / rs ** 2
                                            * (g("mCluster") + 0.5 * r["shell_mass"]), rel=1e-12)
        assert r["F_ram"] == pytest.approx(r["F_ram_wind"] + r["F_ram_SN"], rel=1e-6)
    for a, b in zip(mom[:-1], mom[1:]):
        assert b["shell_mass"] >= a["shell_mass"]
        imp = (a["K_s"] + a["F_rad"] + 0.5 * ((a["F_ram"] - a["F_ion_in"] - a["F_grav"])
                                              + (b["F_ram"] - b["F_ion_in"] - b["F_grav"]))) * (b["t_now"] - a["t_now"])
        dp = b["shell_mass"] * b["v2"] - a["shell_mass"] * a["v2"]
        assert dp == pytest.approx(imp, rel=1e-3)


# --- goldens from archived momentum-entry states ---------------------------------------------------------------
REPLAYS = {
    # cfg: (archive, dt [Myr], final values)
    "ROS": ("data/hpc/c_20261020_160344Z/c3c/ROS", 0.05,
            dict(t_now=0.2541449275435451, rs_dyn=7.803146704777075, R2=2.8182449184169838, v2=0.4751732523377449,
                 shell_mass=48782.04930761799, rs_dot=0.4751732523377449)),
    "B3MW1": ("data/hpc/c_20261020_160350Z/c3c/B3MW1", 2.5949,
              dict(t_now=3.000005544939368, rs_dyn=56.5944666794103, R2=29.77295416523202, v2=23.640990774125516,
                   shell_mass=126267.9094900973, rs_dot=27.574867019049176)),
}


@pytest.mark.parametrize("cfg", list(REPLAYS))
def test_momentum_replay_golden(cfg, tmp_path):
    rel, dt, golden = REPLAYS[cfg]
    need = [BUNDLE / "harness" / f for f in ("momentum_replay.py", "option3_offline.py", "_stamp.py")]
    need += [BUNDLE / rel / "dictionary.jsonl"]
    if not all(f.exists() for f in need):
        pytest.skip("archived run or replay harness absent (docs/dev is gitignored)")
    out = tmp_path / "swept.jsonl"
    # a separate process: trinity keeps module-level state between runs in one interpreter
    subprocess.run([sys.executable, str(BUNDLE / "harness" / "momentum_replay.py"), str(BUNDLE / rel),
                    "--scheme", "swept", "--dt", str(dt), "--out", str(out)], cwd=REPO, check=True,
                   capture_output=True, text=True, timeout=1800)
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    mom = [r for r in rows if r["current_phase"] == "momentum"]
    assert mom
    for r in mom:
        outer = 4 * math.pi * r["rShell"] ** 2 * r["P_outer"]
        L8 = (4 * math.pi * r["R2"] ** 2 * r["Pb"] + r["K_s"] + r["F_rad_sh"] - outer) / outer
        assert abs(L8) < 1e-6, (r["t_now"], L8)
        assert abs(r["rs_dyn"] / r["rShell"] - 1) < 1e-9
        assert r["rs_dot"] >= r["v2"] or r["v2"] <= 0
    for key, value in golden.items():
        assert mom[-1][key] == pytest.approx(value, rel=1e-6), key
