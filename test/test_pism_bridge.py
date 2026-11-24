"""B3 of docs/dev/turnaround-stall/PLAN.md (re-planned 2026-11-23 on main ef0ec37a; harness/apply_b3_v2.py): P_ISM bridged
at the cloud edge with the density's own tanh weight, at every place the outside pressure is built (ruling D3 = turb).

  * density_profile.ism_weight is get_density_profile's own bridge, and exactly the old step outside 0.8-1.2 rCloud;
  * P_ISM is continuous across rCloud, no site keeps the step, and P_ISM 0 / alphaVirCloud 0 (the defaults) give
    ef0ec37a's values bit for bit;
  * with alphaVirCloud > 0 the cloud's own pressure fades with 1 - ism_weight out to 1.2 rCloud (D3 = turb), and its
    sigma^2 counts the total mass inside r as McKee & Tan (2003, eq. A3), M_gas(<r) / (1 - sfe) (A1);
  * swept's edge_speed takes c0^2 from the gas ahead's own thermal pressure, without P_ISM (D6).
On simple_cluster's own parameters (1e5 Msun, sfe 0.3, a uniform 1e5 cm^-3 cloud of rCloud ~ 1.7 pc), with
P/k = 1e6 K cm^-3 (Paper II's highest P_ISM slice) and alphaVirCloud 1 (D4, a virialised cloud) where a test needs them.
Unit-level: no TRINITY run. Lives in test/ (copy as test/test_pism_bridge.py).
"""
import inspect
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import trinity
import trinity._functions.unit_conversions as cvt
import trinity.phase1_energy.energy_phase_ODEs as ODEs
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as IMP
import trinity.phase1c_transition.run_transition_phase as TR
import trinity.phase2_momentum.run_momentum_phase as MOM
import trinity.phase2_momentum.swept as S
from trinity._input.read_param import read_param
from trinity.cloud_properties import density_profile, mass_profile
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp

REPO = Path(__file__).resolve().parents[1]
PKG = Path(trinity.__file__).resolve().parent
PISM_K = 1e6       # P/k [K cm^-3]


def _params(tmp_path, **over):
    """simple_cluster.param plus the keys in over, read as run.py reads it (read_param, then the cloud)."""
    keys = {"log_console": "False", "path2output": str(tmp_path / "out"), **{k: str(v) for k, v in over.items()}}
    par = tmp_path / "pb.param"
    par.write_text((REPO / "param" / "simple_cluster.param").read_text() + "\n"
                   + "\n".join(f"{k} {v}" for k, v in keys.items()) + "\n")
    P = read_param(str(par))
    get_InitCloudProp(P)
    return P


def _c_i_sq(P):
    return P["k_B"].value * P["TShell_ion"].value / P["mu_ion_shell"].value


def test_the_weight_is_the_density_bridge_and_the_step_outside_its_band(tmp_path):
    P = _params(tmp_path)
    rc, w = P["rCloud"].value, density_profile.ism_weight
    assert w(rc, rc) == 0.5
    for r in (0.0, 0.5 * rc, 0.8 * rc * (1 - 1e-12), float("nan")):   # NaN: the step's r >= rCloud was False
        assert w(r, rc) == 0.0
    for r in (1.2 * rc * (1 + 1e-12), 2.0 * rc, float("inf")):
        assert w(r, rc) == 1.0
    # get_density_profile's own weight: n = nCore (1 - w) + nISM w in this uniform cloud
    nC, nI = P["nCore"].value, P["nISM"].value
    rr = np.linspace(0.9 * rc, 1.1 * rc, 2001)
    w_dens = (nC - density_profile.get_density_profile(rr, P)) / (nC - nI)
    ws = np.array([w(float(r), rc) for r in rr])
    assert np.max(np.abs(ws - w_dens)) < 1e-12
    assert np.all(np.diff(ws) >= 0.0)


def _forces_P_ext(P, rShell):
    """press_HII_in (= P_ext) of the three c3c force functions at rShell: 1b, 1c, momentum."""
    sh = SimpleNamespace(rShell=rShell, shell_fAbsorbedIon=1.0, n_IF=0.0, R_IF=0.0, isDissolved=True)
    R2, mShell = 0.9 * rShell, S._scalar(mass_profile.get_mass_profile(rShell, P))
    Pb = 1e7 * cvt.CGS.k_B * cvt.Pb_cgs2au
    L, v = P["Lmech_total"].value, P["v_mech_total"].value
    return (IMP.compute_forces_pure(R2, mShell, Pb, sh, P).press_HII_in,
            TR.compute_forces_pure(R2, mShell, Pb, sh, P).press_HII_in,
            MOM.compute_forces_momentum_pure(R2, mShell, L, v, sh, P).press_HII_in)


def test_pism_is_bridged_at_the_cloud_edge(tmp_path):
    P = _params(tmp_path, PISM=PISM_K, TCloud=0)            # no cloud-gas pressure: the outside pressure is P_ISM alone
    P["Lmech_total"].value, P["v_mech_total"].value = 3e38 * cvt.L_cgs2au, 2000 * cvt.v_kms2au   # 1c's P_ram
    rc, pism = P["rCloud"].value, P["PISM"].value * P["k_B"].value
    assert pism == pytest.approx(PISM_K * cvt.CGS.k_B * cvt.Pb_cgs2au, rel=1e-9)
    for r, share in ((0.5 * rc, 0.0), (rc, 0.5), (2.0 * rc, 1.0)):
        for got in (S.p_ext(r, 1.0, P),) + _forces_P_ext(P, r):
            assert got == pytest.approx(share * pism, rel=1e-12, abs=0.0)
    ps = np.array([S.p_ext(float(r), 1.0, P) for r in np.linspace(0.9 * rc, 1.1 * rc, 4001)])
    assert np.max(np.abs(np.diff(ps))) < 0.01 * pism        # the old step was all of pism at rCloud


# the step as ef0ec37a wrote it at all seven sites: `if r(Shell) >= ...rCloud...:` and then `... += ...PISM...`
_STEP = re.compile(r"if\s+\w+\s*>=\s*[^:\n]*rCloud[^:\n]*:\s*\n\s*\w+\s*\+=[^\n]*PISM")
_SITES = (ODEs.get_ODE_Edot_pure, ODEs.compute_derived_quantities, IMP.compute_forces_pure, TR.compute_forces_pure,
          MOM.compute_forces_momentum_pure, MOM.get_ODE_momentum_pure, S.p_ext)


def test_no_site_keeps_the_pism_step():
    hits = [f"{p.relative_to(PKG)}: {m.group(0)!r}" for p in sorted(PKG.rglob("*.py"))
            for m in _STEP.finditer(p.read_text(encoding="utf-8"))]
    assert hits == []
    for fn in _SITES:
        assert "ism_weight(" in inspect.getsource(fn), f"{fn.__module__}.{fn.__qualname__}"


def test_defaults_are_bit_identical(tmp_path):
    """P_ISM 0 and alphaVirCloud 0 (simple_cluster as shipped): the cloud pressure and p_ext are ef0ec37a's, bit for bit,
    in and around the bridge; with P_ISM > 0 the same holds outside 0.8-1.2 rCloud."""
    P = _params(tmp_path)
    assert P["PISM"].value == 0.0 and P["alphaVirCloud"].value == 0.0 and P["TCloud"].value == 10.0
    rc = P["rCloud"].value
    for r in (x * rc for x in (0.5, 0.79, 0.81, 0.95, 0.99, 1.0, 1.01, 1.05, 1.19, 1.21, 2.0)):
        n = float(np.atleast_1d(density_profile.get_density_profile(np.atleast_1d(r), P))[0])
        P_th = P["mu_convert"].value / P["mu_mol"].value * n * P["k_B"].value * P["TCloud"].value if r < rc else 0.0
        assert density_profile.get_press_cloud(r, 1.0, P) == P_th                     # the step at rCloud stays
        for fabs in (1.0, 0.5):
            ion = S._scalar(S.get_press_ion(r, P)) if fabs < 1.0 else 0.0
            assert S.p_ext(r, fabs, P) == ion + density_profile.get_press_cloud(r, fabs, P)   # P_ISM 0 adds nothing
    Pp = _params(tmp_path, PISM=PISM_K)
    pism = Pp["PISM"].value * Pp["k_B"].value
    for r in (x * Pp["rCloud"].value for x in (0.5, 0.79)):
        assert S.p_ext(r, 1.0, Pp) == density_profile.get_press_cloud(r, 1.0, Pp)        # no P_ISM inside 0.8 rCloud
    for r in (x * Pp["rCloud"].value for x in (1.21, 2.0)):
        assert S.p_ext(r, 1.0, Pp) == pism                                                # all of it past 1.2 rCloud


def test_turbulent_cloud_pressure_fades_with_the_bridge(tmp_path):
    """D3 = turb: with alphaVirCloud > 0 the cloud's whole pressure is weighted by 1 - ism_weight, out to 1.2 rCloud;
    the thermal call edge_speed makes (turbulent=False) keeps the step at rCloud."""
    P = _params(tmp_path, alphaVirCloud=1)                  # D4: virialised; TCloud 10 K as shipped
    rc, a, G, sfe = P["rCloud"].value, P["alphaVirCloud"].value, P["G"].value, P["sfe"].value
    for r in (x * rc for x in (0.5, 0.9, 0.99, 1.0, 1.01, 1.1)):
        n = float(np.atleast_1d(density_profile.get_density_profile(np.atleast_1d(r), P))[0])
        P_th = P["mu_convert"].value / P["mu_mol"].value * n * P["k_B"].value * P["TCloud"].value
        M_tot = S._scalar(mass_profile.get_mass_profile(r, P)) / (1.0 - sfe)
        rho_sig2 = P["mu_convert"].value * n * a * G * M_tot / (5.0 * r)
        want = max(P_th, rho_sig2) * (1.0 - density_profile.ism_weight(r, rc))
        assert density_profile.get_press_cloud(r, 1.0, P) == pytest.approx(want, rel=1e-9)
    assert density_profile.get_press_cloud(rc, 1.0, P) > 0.0                        # no cut at rCloud
    assert density_profile.get_press_cloud(1.2 * rc * (1 + 1e-12), 1.0, P) == 0.0
    ps = [density_profile.get_press_cloud(float(r), 1.0, P) for r in np.linspace(0.9 * rc, 1.1 * rc, 8001)]
    assert np.max(np.abs(np.diff(ps))) < 0.01 * max(ps)
    assert density_profile.get_press_cloud(rc, 1.0, P, turbulent=False) == 0.0     # edge_speed's call: the step
    assert density_profile.get_press_cloud(0.99 * rc, 1.0, P, turbulent=False) > 0.0


def test_turbulent_sigma_counts_the_stars_as_mckee_tan(tmp_path):
    """A1 (ruled 2026-11-20, b5-turbulent-pressure.md): sigma^2 = alpha_vir G M(<r) / (5 r) with McKee & Tan's (2003,
    eq. A3) total mass, the stars spread like the gas: M(<r) = M_gas(<r) / (1 - sfe). simple_cluster's cloud is uniform,
    so M_gas(<r) = 4/3 pi r^3 rho in closed form and mass_profile is not reused."""
    P = _params(tmp_path, TCloud=0, alphaVirCloud=1)        # TCloud 0: the cloud gas pushes with rho sigma^2 alone
    assert P["sfe"].value == pytest.approx(0.3)             # sigma^2 is 1/0.7 of the gas-only value
    r = 0.5 * P["rCloud"].value
    rho = P["mu_convert"].value * density_profile.get_density_profile(r, P)
    M_tot = 4.0 / 3.0 * math.pi * r ** 3 * rho / (1.0 - P["sfe"].value)
    want = rho * P["alphaVirCloud"].value * P["G"].value * M_tot / (5.0 * r)
    assert density_profile.get_press_cloud(r, 1.0, P) == pytest.approx(want, rel=1e-9)


def test_edge_speed_c0_leaves_pism_out(tmp_path):
    """D6: in the bridge inside the cloud the force sees P_ISM w, but the jump's c0^2 stays the gas ahead's own thermal
    pressure over rho: TCloud's while photons are held, c_i^2 where they escape (with P_ISM in it c0^2 > c_i^2 there,
    and the jump would switch off). For u > 0 and S7's sweep branch (u < 0, sweep=True)."""
    P = _params(tmp_path, PISM=PISM_K)                      # TCloud 10 K as shipped
    rc = P["rCloud"].value
    r = 0.99 * rc
    w, pism = density_profile.ism_weight(r, rc), P["PISM"].value * P["k_B"].value
    th = density_profile.get_press_cloud(r, 1.0, P, turbulent=False)
    rho = P["mu_convert"].value * density_profile.get_density_profile(r, P)
    assert 0.05 < w < 0.2 and w * pism > 0.1 * th            # P_ISM is not small against the 10 K gas here
    assert S.p_ext(r, 1.0, P) == pytest.approx(th + w * pism, rel=1e-12)     # the force feels it
    for fabs, c0_sq in ((1.0, th / rho), (0.5, _c_i_sq(P))):
        for u, sweep in ((20.0, False), (1.0, False), (-1.0, True)):
            jump = 0.5 * (u + math.sqrt(u * u + 4.0 * c0_sq))
            assert S.edge_speed(u, r, fabs, P, sweep=sweep) == pytest.approx(jump, rel=1e-12)
