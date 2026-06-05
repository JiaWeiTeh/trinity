"""Anti-drift validation for the n = n_H (mu / chi_e) audit — Phases 0-2.

Every refined operation is validated against the ORIGINAL pre-fix operation
and value (pre-fix commit 7321fef), so any *silent drift* — a reverted
factor, a changed mu, a dropped chi_e, an unintended collateral edit — fails
loudly here.  See analysis/n-consistency-audit.md and
analysis/n-consistency-implementation-plan.md.

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
    p = _p()
    f_new = p["mu_convert"].value / p["mu_ion"].value  # implemented factor
    f_orig = 2.0  # pre-fix pure-hydrogen value
    assert abs(f_new - 2.3) < 1e-12  # He-aware mu_H/mu_p
    assert abs(f_new / f_orig - 1.15) < 1e-12  # intended change ratio

    # For identical (n, kB, T) the refined pressure is exactly 1.15x original.
    n, kB, T = 3.3e4, p["k_B"].value, p["TShell_ion"].value
    P_orig = f_orig * n * kB * T
    P_new = f_new * n * kB * T
    assert np.isclose(P_new / P_orig, 1.15, rtol=1e-12)


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
    factor = "(params['mu_convert'].value / params['mu_ion'].value)"
    total = 0
    for rel in files:
        s = _src(rel)
        for orig in ("P_ion = 2.0 *", "P_HII = 2.0 *", "P_ext = 2.0 *",
                     "P_HII_f = 2.0 *"):
            assert orig not in s, f"{rel}: original op '{orig}' reverted"
        total += s.count(factor)
    assert total == 11, f"expected 11 refined ionised-pressure sites, found {total}"


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
    """Ionised shell ODE: dndr must equal the REFINED formula (mu_p/mu_H
    prefactor + chi_e recombination) and NOT the original (mu_p/mu_n, no
    chi_e); dphidr recomb gains chi_e; dtaudr (dust) is invariant."""
    from trinity.shell_structure.get_shellODE import get_shellODE
    p = _shell_params()
    n, phi, tau, r = 1.0e3, 0.5, 0.3, 5.0
    dndr, dphidr, dtaudr = get_shellODE([n, phi, tau], r, 1.0, True, p)

    mu_n = p["mu_atom"].value
    mu_p = p["mu_ion"].value
    mu_H = p["mu_convert"].value
    chi = p["chi_e"].value
    kB = p["k_B"].value
    c = p["c_light"].value
    sd = p["dust_sigma"].value
    aB = p["caseB_alpha"].value
    Ln, Li, Qi = p["Ln"].value, p["Li"].value, p["Qi"].value
    tion = p["TShell_ion"].value

    net = np.exp(-tau)
    dust = n * sd / (4 * np.pi * r**2 * c) * (Ln * net + Li * phi)
    recomb = n**2 * aB * Li / Qi / c
    dndr_refined = mu_p / mu_H / (kB * tion) * (dust + chi * recomb)
    dndr_original = mu_p / mu_n / (kB * tion) * (dust + recomb)

    assert np.isclose(dndr, dndr_refined, rtol=1e-12)      # function == refined
    assert not np.isclose(dndr, dndr_original, rtol=1e-6)  # and != original

    dphidr_refined = -4 * np.pi * r**2 * chi * aB * n**2 / Qi - n * sd * phi
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
    """Closed-form coefficients + structural anti-drift for shell_structure.py:
    BC mu_ion/mu_convert; mass/grav/tau weights mu_convert x8; chi_e x3;
    mu_atom survives ONLY at the convention-independent I-front jump (:298)."""
    s = _src("trinity/shell_structure/shell_structure.py")
    assert "params['mu_ion'].value / params['mu_convert'].value" in s   # refined BC
    assert "params['mu_ion'].value / params['mu_atom'].value" not in s  # original BC gone
    assert s.count("params['mu_convert'].value") == 8   # 7 mass/grav/tau + BC
    assert s.count("params['mu_atom'].value") == 1      # only the I-front jump
    assert "params['mu_atom'].value / params['mu_ion'].value" in s      # the :298 jump
    assert s.count("params['chi_e'].value") == 3        # max_shellRadius, n_IF_Str, phi_hydrogen


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
