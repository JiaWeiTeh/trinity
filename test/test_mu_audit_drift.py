"""Anti-drift validation for the n = n_H (mu / chi_e) audit — Phases 0-2.

Every refined operation is validated against the ORIGINAL pre-fix operation
and value (pre-fix commit 7321fef), so any *silent drift* — a reverted
factor, a changed mu, a dropped chi_e, an unintended collateral edit — fails
loudly here.  See docs/dev/archive/n-consistency/audit.md and
docs/dev/archive/n-consistency/implementation-plan.md.

Composition at defaults: x_He = 0.1, Z_He = 2  =>
    mu_H/mu_p = mu_convert/mu_ion = 2.3 ,   chi_e = 1 + Z_He*x_He = 1.2 .
The ORIGINAL code used the pure-hydrogen limits:
    ionised pressure factor 2.0   (= mu_H/mu_p for mu_H=1, mu_p=1/2),
    bubble n  = Pb/(2 kB T),   rho = n * mu_ion,   CIE = n^2 * Lambda  (no chi_e).
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

import trinity._functions.unit_conversions as cvt
from trinity._input.read_param import read_param

REPO = Path(__file__).resolve().parents[1]
PARAM = str(REPO / "param" / "cloud_example_PL.param")


def _p():
    return read_param(PARAM)


def _src(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# =====================================================================
# Phase 0 — mu_* derived from x_He/Z_He must equal the ORIGINAL values
# =====================================================================
def test_phase0_mu_bit_identical_to_original():
    """Derived mu_* reproduce the pre-fix default.param values (string ->
    float(Fraction) * m_H) bit-for-bit.  Any drift in the Step-6 derivation
    breaks this."""
    p = _p()
    mH = cvt.convert2au("m_H")
    original = {
        "mu_convert": 1.4 * mH,
        "mu_atom": float(Fraction(14, 11)) * mH,
        "mu_ion": float(Fraction(14, 23)) * mH,
        "mu_mol": float(Fraction(14, 6)) * mH,
    }
    for k, orig in original.items():
        assert p[k].value == orig, (
            f"{k} drifted from the original pre-fix value: "
            f"{p[k].value!r} != {orig!r}"
        )


def test_phase0_chi_e_is_new_and_correct():
    """chi_e is the only genuinely new constant: n_e/n_H = 1 + Z_He*x_He."""
    p = _p()
    assert p["chi_e"].value == 1.0 + p["Z_He"].value * p["x_He"].value
    assert p["chi_e"].value == 1.2


# =====================================================================
# Phase 1 — ionised-gas pressure prefactor: original 2.0 -> mu_H/mu_p
# =====================================================================
def test_phase1_pressure_factor_vs_original():
    """The ~1e4 K HII / shell pressure uses the SINGLY-ionised shell factor
    mu_H/mu_p,shell = mu_convert/mu_ion_shell = 2.2 -- NOT the bubble's
    doubly-ionised 2.3, and NOT the original pure-H 2.0."""
    p = _p()
    f_new = p["mu_convert"].value / p["mu_ion_shell"].value  # implemented HII factor
    f_orig = 2.0  # pre-fix pure-hydrogen value
    assert abs(f_new - 2.2) < 1e-12                                  # singly mu_H/mu_p,shell
    assert abs(p["mu_convert"].value / p["mu_ion"].value - 2.3) < 1e-12  # bubble stays doubly
    assert np.isclose(f_new / f_orig, 1.1, rtol=1e-12)              # intended change vs original


def test_phase1_all_eleven_sites_refined_and_no_original_remains():
    """All 11 ionised-pressure sites carry the refined factor; not one of the
    original `* 2.0 *` operations survives."""
    files = [
        "trinity/phase1_energy/energy_phase_ODEs.py",
        "trinity/phase1_energy/run_energy_phase.py",
        "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
        "trinity/phase1c_transition/run_transition_phase.py",
        "trinity/phase2_momentum/run_momentum_phase.py",
    ]
    factor = "(params['mu_convert'].value / params['mu_ion_shell'].value)"
    total = 0
    for rel in files:
        s = _src(rel)
        for orig in ("P_ion = 2.0 *", "P_HII = 2.0 *", "P_ext = 2.0 *",
                     "P_HII_f = 2.0 *"):
            assert orig not in s, f"{rel}: original op '{orig}' reverted"
        # the doubly-ionised bubble factor must NOT leak into the HII/shell sites
        assert "params['mu_convert'].value / params['mu_ion'].value" not in s, (
            f"{rel}: HII pressure must use mu_ion_shell (singly), not mu_ion")
        total += s.count(factor)
    assert total == 11, f"expected 11 refined HII-pressure sites, found {total}"


# =====================================================================
# Phase 2 — bubble interior: n -> n_H, rho -> mu_H*n, CIE -> chi_e*n^2*Lambda
# =====================================================================
def test_phase2_bubble_n_rho_cie_vs_original():
    p = _p()
    kB = p["k_B"].value
    mi = p["mu_ion"].value
    mc = p["mu_convert"].value
    chi = p["chi_e"].value
    Pb, T, Lam = 1.234e-3, 1.0e6, 5.0e-7  # arbitrary positive test point

    # ORIGINAL operations (pre-fix 7321fef)
    n_orig = Pb / (2 * kB * T)
    rho_orig = n_orig * mi
    cie_orig = n_orig**2 * Lam

    # REFINED operations (current committed code)
    n_new = Pb / ((mc / mi) * kB * T)
    rho_new = n_new * mc
    cie_new = chi * n_new**2 * Lam

    # Intended factors vs original (no other drift).
    assert np.isclose(n_new / n_orig, 2.0 / (mc / mi), rtol=1e-12)        # ~0.8696
    assert np.isclose(rho_new / rho_orig, 2.0, rtol=1e-12)               # factor-2 fix
    assert np.isclose(cie_new / cie_orig, chi * (2.0 / (mc / mi)) ** 2,  # ~0.9074
                      rtol=1e-12)

    # Refined values are the physically-correct ones; original were not.
    ntot = Pb / (kB * T)
    assert np.isclose(n_new, (mi / mc) * ntot, rtol=1e-12)   # n_H = (mu_p/mu_H) n_tot
    assert np.isclose(rho_new, mi * ntot, rtol=1e-12)        # rho = mu_p * n_tot (correct)
    assert np.isclose(rho_orig, 0.5 * mi * ntot, rtol=1e-12)  # original rho = 0.5x (deficit)


def test_phase2_no_original_operations_remain():
    """Bubble + cooling source: every original op is gone, every refined op
    present with the expected multiplicity."""
    bub = _src("trinity/bubble_structure/bubble_luminosity.py")
    assert "Pb / (2 * params[" not in bub
    assert "n[::-1] * params['mu_ion'].value" not in bub
    assert "integrand_bubble = n_bubble**2" not in bub
    assert "integrand_int = n_interm[mask]**2" not in bub
    assert bub.count(
        "Pb / ((params['mu_convert'].value / params['mu_ion'].value)"
    ) == 5
    assert "n[::-1] * params['mu_convert'].value" in bub
    assert bub.count("params['chi_e'].value * n_bubble**2") == 1
    assert bub.count("params['chi_e'].value * n_interm[mask]**2") == 1

    cool = _src("trinity/cooling/net_coolingcurve.py")
    assert "dudt = ndens**2 * Lambda_CIE" not in cool
    assert "dudt_CIE = (ndens**2 * Lambda)" not in cool
    assert "params_dict['chi_e'].value * ndens**2 * Lambda_CIE" in cool
    assert "params_dict['chi_e'].value * ndens**2 * Lambda)" in cool


def test_phase2_conduction_sites_untouched():
    """Weaver conduction/evaporation still uses mu_ion as the mean-mass-per-
    particle mu_p (Mbdot / Tprofile / vprofile) — must NOT have drifted."""
    bub = _src("trinity/bubble_structure/bubble_luminosity.py")
    assert "* mu_ion / k_B" in bub                       # _get_init_dMdt
    assert "constant = (25/4 * k_B / mu_ion / C_thermal)" in bub
    assert "* k_B * T / mu_ion / Pb)" in bub


# =====================================================================
# Phase 3 — shell to n_H: validate the real get_shellODE against the
# ORIGINAL and the REFINED formula, plus structural coefficient guards.
# =====================================================================
def _shell_params():
    """Params with runtime feedback values (default 0) set to positive test
    values so get_shellODE's radiation/recombination terms are non-trivial."""
    p = _p()
    p["Ln"].value = 1.0e3
    p["Li"].value = 2.0e3
    p["Qi"].value = 1.0e5
    return p


def test_phase3_shellODE_ion_vs_original():
    """Ionised shell ODE uses the SINGLY-ionised shell composition: prefactor
    mu_p,shell/mu_H + chi_e_shell recombination -- NOT the pre-audit (mu_ion/
    mu_atom, no chi_e), and NOT the bubble's doubly-ionised mu_ion/chi_e."""
    from trinity.shell_structure.get_shellODE import get_shellODE
    p = _shell_params()
    n, phi, tau, r = 1.0e3, 0.5, 0.3, 5.0
    dndr, dphidr, dtaudr = get_shellODE([n, phi, tau], r, 1.0, True, p)

    mu_n = p["mu_atom"].value
    mu_p_shell = p["mu_ion_shell"].value   # shell HII is singly-ionised
    mu_ion = p["mu_ion"].value             # bubble (doubly) -- must NOT drive the shell
    mu_H = p["mu_convert"].value
    chi_sh = p["chi_e_shell"].value
    kB = p["k_B"].value
    c = p["c_light"].value
    sd = p["dust_sigma"].value
    aB = p["caseB_alpha"].value
    Ln, Li, Qi = p["Ln"].value, p["Li"].value, p["Qi"].value
    tion = p["TShell_ion"].value

    net = np.exp(-tau)
    dust = n * sd / (4 * np.pi * r**2 * c) * (Ln * net + Li * phi)
    recomb = n**2 * aB * Li / Qi / c
    dndr_refined = mu_p_shell / mu_H / (kB * tion) * (dust + chi_sh * recomb)
    dndr_original = mu_ion / mu_n / (kB * tion) * (dust + recomb)  # pre-audit

    assert np.isclose(dndr, dndr_refined, rtol=1e-12)      # function == refined (singly)
    assert not np.isclose(dndr, dndr_original, rtol=1e-6)  # and != pre-audit

    dphidr_refined = -4 * np.pi * r**2 * chi_sh * aB * n**2 / Qi - n * sd * phi
    assert np.isclose(dphidr, dphidr_refined, rtol=1e-12)
    assert np.isclose(dtaudr, n * sd * 1.0, rtol=1e-12)    # dust term: invariant


def test_phase3_shellODE_neutral_vs_original():
    """Neutral shell ODE: prefactor 1 -> mu_n/mu_H (dust-only RHS)."""
    from trinity.shell_structure.get_shellODE import get_shellODE
    p = _shell_params()
    n, tau, r = 1.0e3, 0.3, 5.0
    dndr, dtaudr = get_shellODE([n, tau], r, 1.0, False, p)

    mu_n = p["mu_atom"].value
    mu_H = p["mu_convert"].value
    kB = p["k_B"].value
    c = p["c_light"].value
    sd = p["dust_sigma"].value
    Ln = p["Ln"].value
    tneu = p["TShell_neu"].value

    net = np.exp(-tau)
    dust = n * sd / (4 * np.pi * r**2 * c) * (Ln * net)
    dndr_refined = mu_n / mu_H / (kB * tneu) * dust
    dndr_original = 1.0 / (kB * tneu) * dust

    assert np.isclose(dndr, dndr_refined, rtol=1e-12)
    assert not np.isclose(dndr, dndr_original, rtol=1e-6)


def test_phase3_shell_structure_coefficients_and_no_original():
    """shell_structure.py uses the SINGLY-ionised shell composition: BC and
    I-front jump on mu_ion_shell; chi_e_shell x3; mass/grav/tau weights
    mu_convert x8; mu_atom only at the jump numerator; the bubble's doubly-
    ionised mu_ion / chi_e never appear in the shell."""
    s = _src("trinity/shell_structure/shell_structure.py")
    assert "params['mu_ion_shell'].value / params['mu_convert'].value" in s  # refined BC (singly)
    assert "params['mu_ion'].value" not in s            # bubble (doubly) mu_ion must not leak
    assert s.count("params['mu_convert'].value") == 8   # 7 mass/grav/tau + BC
    assert s.count("params['mu_atom'].value") == 1      # only the I-front jump numerator
    assert "params['mu_atom'].value / params['mu_ion_shell'].value" in s  # the :298 jump (singly)
    assert s.count("params['chi_e_shell'].value") == 3  # max_shellRadius, n_IF_Str, phi_hydrogen
    assert "params['chi_e'].value" not in s             # bubble chi_e must not leak


def test_phase3_coefficients_reduce_to_original_at_pure_H():
    """Faithful-generalisation check: at x_He=0 (pure H) mu_n=mu_H so the
    refined prefactor mu_p/mu_H collapses onto the original mu_p/mu_n, and
    chi_e=1 (recomb unchanged). This is what lets us attribute the helium-run
    drift to physics rather than to a different/buggy formula."""
    xHe, ZHe = Fraction(0), Fraction(2)
    muH = 1 + 4 * xHe
    mu_n = muH / (1 + xHe)
    chi_e = 1 + ZHe * xHe
    assert mu_n == muH    # => mu_p/mu_H == mu_p/mu_n (original prefactor) at x_He=0
    assert chi_e == 1     # => recomb/Stromgren collapse to the original


# =====================================================================
# Phase 6 — sound-speed docstring (6A) + BE sigma expose (no mu/structure change)
# =====================================================================
def test_phase6a_get_soundspeed_docstring_and_value():
    """6A is docstring-only: 'adiabatic' + 'pc/Myr', and the returned value is
    unchanged (still sqrt(gamma*kB*T/mu_ion) for the hot bubble)."""
    import trinity._functions.operations as ops
    import trinity._functions.unit_conversions as cvt
    doc = ops.get_soundspeed.__doc__.lower()
    assert "adiabatic" in doc and "isothermal" not in doc
    assert "pc/myr" in doc and "myr/pc" not in doc
    p = _p()
    T = 1.0e6
    mu = p["mu_ion"].value * cvt.Msun2g
    expect = np.sqrt(p["gamma_adia"].value * (p["k_B"].value * cvt.k_B_au2cgs)
                     * T / mu) * cvt.v_cms2au
    assert np.isclose(ops.get_soundspeed(T, p), expect, rtol=1e-12)


def test_phase6b_densBE_sigma_exposed_and_Teff_mu_unchanged():
    """6B exposes sigma = c_s [km/s] but does NOT touch the BE EOS mu/gamma:
    densBE_Teff still equals the ORIGINAL mu_convert*c_s^2/(gamma*kB), so the
    cloud structure is provably unchanged (this rejects the abandoned mu_mol
    plan)."""
    from trinity.cloud_properties.bonnorEbertSphere import (
        create_BE_sphere_from_params, K_B_CGS, MSUN_TO_G,
    )
    p = read_param(str(REPO / "param" / "cloud_example_BE.param"))
    res = create_BE_sphere_from_params(p)
    # sigma is the support velocity dispersion c_s, in km/s
    assert np.isclose(p["densBE_sigma"].value, res.c_s / 1.0e5, rtol=1e-12)
    assert p["densBE_sigma"].ori_units == "km/s"
    # densBE_Teff still uses the ORIGINAL mu_convert + gamma_adia (no mu_mol drift)
    Teff_orig = (p["mu_convert"].value * MSUN_TO_G * res.c_s**2
                 / (p["gamma_adia"].value * K_B_CGS))
    assert np.isclose(p["densBE_Teff"].value, Teff_orig, rtol=1e-9)


# =====================================================================
# Phase A — region-dependent He ionisation (bubble doubly / shell singly)
# =====================================================================
def test_phaseA_region_dependent_ionisation():
    """Hot bubble is doubly-ionised (Z_He=2); the ~1e4 K shell/HII region is
    singly-ionised (Z_He_shell=1). The two composition sets must not cross-leak."""
    import trinity._functions.unit_conversions as cvt
    p = _p()
    mH = cvt.convert2au("m_H")
    # bubble (doubly): Z_He=2 -> mu_ion=14/23, chi_e=1.2
    assert p["Z_He"].value == 2
    assert np.isclose(p["mu_ion"].value / mH, 14 / 23, rtol=1e-12)
    assert np.isclose(p["chi_e"].value, 1.2, rtol=1e-12)
    # shell/HII (singly): Z_He_shell=1 -> mu_ion_shell=14/22, chi_e_shell=1.1
    assert p["Z_He_shell"].value == 1
    assert np.isclose(p["mu_ion_shell"].value / mH, 14 / 22, rtol=1e-12)
    assert np.isclose(p["chi_e_shell"].value, 1.1, rtol=1e-12)
    # no cross-leak: bubble code keeps mu_ion (doubly) and never uses *_shell;
    # shell code (checked elsewhere) never uses the bubble mu_ion/chi_e.
    bub = _src("trinity/bubble_structure/bubble_luminosity.py")
    assert "params['mu_convert'].value / params['mu_ion'].value" in bub  # bubble n keeps doubly
    assert "mu_ion_shell" not in bub and "chi_e_shell" not in bub
