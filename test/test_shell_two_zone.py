#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The two-zone neutral shell (TShell_mol > 0) against an exact isothermal slab, and the one-zone layer (TShell_mol 0) untouched.

docs/dev/phii-identity/weak-wind/neutral-shell/PLAN.md, gate G-NS1. The state is StarBench-like, as in
test_shell_neutral_slice_cap.py: n = 1e3 cm^-3, R_St = 1 pc, R2 = 1e-3 pc, Ln = 0 (no radiation force on the neutral gas; a
tiny Li keeps the ionised march on), dust ON so that tau grows (the column is counted from the front). The ionised layer's own
dust then stops the front short of R_St: R_IF = 0.527 pc, tau_IF = 2.43 (checked by an independent scipy integration of
dphi/dr = -3 r^2/R_St^3 - n dust_sigma phi, 2026-10-29). Both neutral zones are uniform and the solve has a closed form:
    n_1 = n_IF (mu_atom/mu_ion_shell) (T_ion/T_PDR)               the atomic zone, from R_IF to r_b
    r_b = R_IF + tauShell_mol / (n_1 dust_sigma)                    dtau/dr = n_1 dust_sigma
    n_2 = n_1 (mu_mol/mu_atom) (T_PDR/T_mol)                        the pressure-continuous jump (27.5 at 300 K / 20 K, x_He 0.1)
    r_s^3 = r_b^3 + 3 M_2 / (4 pi mu_H n_2),   M_1 = (4 pi/3) mu_H n_1 (r_b^3 - R_IF^3),   M_2 = M_sh - M_ion - M_1
In units of the test's m_ion (n0 uniform to R_St): M_ion = 0.146, M_1 = 0.553 (n_1 = 5.33e4 cm^-3, zone 1 0.0122 pc thick), so
a shell of 10 m_ion has both zones and a shell of 0.4 m_ion runs out of mass inside the atomic zone (checked as a premise in the
test that uses it). A first version used 2.5 m_ion, from a closed form without the ionised layer's dust; that shell reaches the
molecular zone.
"""
import logging
import math
from pathlib import Path

import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
import trinity.phase2_momentum.swept as S
from trinity._input.read_param import read_param
from trinity.shell_structure.shell_structure import shell_structure_pure

REPO = Path(__file__).resolve().parents[1]
N0_CGS = 1e3                            # cm^-3
H_NU_ERG = 13.6 * 1.602176634e-12
T_PDR, T_MOL, DTAU = 300.0, 20.0, 3.0


def _slab(m_over_m_ion, **keys):
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    g = lambda k: float(p[k].value)  # noqa: E731
    for k, v in keys.items():
        p[k].value = v
    n0 = N0_CGS * cvt.ndens_cgs2au
    Pb = n0 * g("k_B") * g("TShell_ion") * g("mu_convert") / g("mu_ion_shell")     # the solve derives n0 from Pb
    Qi = 4 * math.pi / 3 * g("chi_e_shell") * g("caseB_alpha") * n0**2 * 1.0**3       # R_St = 1 pc
    m_ion = g("mu_convert") * n0 * 4 * math.pi / 3 * 1.0**3                            # the uniform ionised layer
    p["Qi"].value, p["Ln"].value = Qi, 0.0
    p["Li"].value = 1e-20 * Qi * H_NU_ERG * cvt.E_cgs2au
    p["Pb"].value, p["R2"].value, p["rShell"].value = Pb, 1e-3, 1e-3
    p["shell_mass"].value = m_over_m_ion * m_ion
    p["bubble_mass"].value = 0.0
    p["isDissolved"].value = False
    return p, m_ion, shell_structure_pure(p)


def test_defaults_are_the_pdr_two_zone_layer():
    """Since the flip (2026-11-02, PDR_DEFAULT_PLAN.md P5) the defaults are step (b): TShell_neu 'auto', TShell_mol 20 K, tauShell_mol
    3. A numeric TShell_neu with TShell_mol 0 is the one-zone layer of earlier runs: no molecular zone, no PDR temperature."""
    p, _, sp = _slab(10.0)
    assert p["TShell_neu"].value == "auto" and p["TShell_mol"].value == 20.0 and p["tauShell_mol"].value == 3.0
    assert sp.has_neutral and np.isfinite(sp.shell_T_atom) and sp.shell_N_tran > 0
    _, _, one = _slab(10.0, TShell_neu=100.0, TShell_mol=0.0)
    assert one.has_neutral and one.shell_mol_idx == -1
    assert np.isnan(one.shell_T_atom) and np.isnan(one.shell_G0_IF) and np.isnan(one.shell_N_tran)


def test_column_at_the_defaults_is_2e21_per_cm2():
    p, _, _ = _slab(2.5)
    sigma_cgs = float(p["dust_sigma"].value) * cvt.pc2cm**2            # dust_sigma is stored in pc^2
    assert float(p["tauShell_mol"].value) / sigma_cgs == pytest.approx(2.0e21, rel=1e-9)


def test_two_zone_slab_matches_the_closed_form():
    p, m_ion, sp = _slab(10.0, TShell_neu=T_PDR, TShell_mol=T_MOL, tauShell_mol=DTAU)
    g = lambda k: float(p[k].value)  # noqa: E731
    i, j = int(sp.shell_ion_idx), int(sp.shell_mol_idx)
    assert sp.has_neutral and j > i
    r, n = np.asarray(sp.shell_r_arr, float), np.asarray(sp.shell_n_arr, float)
    n1 = float(sp.n_IF) * g("mu_atom") / g("mu_ion_shell") * g("TShell_ion") / T_PDR
    r_b = float(sp.R_IF) + DTAU / (n1 * g("dust_sigma"))
    assert abs(r[j] / r_b - 1) < 1e-6                                    # the landed boundary, atomic side
    assert r[j + 1] == r[j]                                              # the molecular side starts at the same radius
    jump = g("mu_mol") / g("mu_atom") * T_PDR / T_MOL
    assert jump == pytest.approx(27.5, rel=1e-12)                        # (14/6)/(14/11) x 300/20 at x_He 0.1
    assert n[j + 1] / n[j] == pytest.approx(jump, rel=1e-12)
    assert np.all(np.abs(n[i + 1:j + 1] / n1 - 1) < 1e-9)                # zone 1 uniform (Ln = 0)
    assert np.all(np.abs(n[j + 1:] / (n1 * jump) - 1) < 1e-9)            # zone 2 uniform
    m_1 = 4 * math.pi / 3 * g("mu_convert") * n1 * (r_b**3 - float(sp.R_IF)**3)
    m_2 = 10.0 * m_ion - float(sp.shell_mass_ion) - m_1
    assert m_2 > 0
    r_exact = (r_b**3 + 3 * m_2 / (4 * math.pi * g("mu_convert") * n1 * jump)) ** (1 / 3)
    assert abs(float(sp.rShell) / r_exact - 1) < 1e-6
    assert float(sp.shell_mass_neutral) == pytest.approx(10.0 * m_ion - float(sp.shell_mass_ion), rel=1e-6)


def test_mass_running_out_inside_the_atomic_zone_leaves_one_zone():
    p1, m_ion, one = _slab(0.4, TShell_neu=T_PDR)
    g = lambda k: float(p1[k].value)  # noqa: E731
    n1 = float(one.n_IF) * g("mu_atom") / g("mu_ion_shell") * g("TShell_ion") / T_PDR
    r_b = float(one.R_IF) + DTAU / (n1 * g("dust_sigma"))
    m_1 = 4 * math.pi / 3 * g("mu_convert") * n1 * (r_b**3 - float(one.R_IF)**3)
    assert one.has_neutral, "premise: the shell must reach past the front"
    assert 0.4 * m_ion - float(one.shell_mass_ion) < m_1, "premise: the neutral mass must run out before the column"
    _, _, two = _slab(0.4, TShell_neu=T_PDR, TShell_mol=T_MOL, tauShell_mol=DTAU)
    assert two.shell_mol_idx == -1
    assert np.array_equal(np.asarray(one.shell_r_arr), np.asarray(two.shell_r_arr))
    assert np.array_equal(np.asarray(one.shell_n_arr), np.asarray(two.shell_n_arr))
    assert float(one.rShell) == float(two.rShell)


def test_layer_terms_identity_closes_with_a_molecular_zone():
    """L8, 4 pi R2^2 P_c + K_s + F_rad,sh = 4 pi r_s^2 P(r_s), with the zone pressures of swept.layer_terms. The slab is isobaric
    (radiation off), so L8 closes for ANY state once each zone's pressure prefactor is right -- which is what this checks: with
    the atomic prefactor kept beyond the boundary, K_s and P(r_s) would be 27.5x too large there and L8 would be of order 1."""
    p, _, sp = _slab(10.0, TShell_neu=T_PDR, TShell_mol=T_MOL, tauShell_mol=DTAU)
    g = lambda k: float(p[k].value)  # noqa: E731
    j = int(sp.shell_mol_idx)
    assert j > int(sp.shell_ion_idx)
    n = np.asarray(sp.shell_n_arr, float)
    p_neu = g("mu_convert") / g("mu_atom") * g("k_B") * T_PDR
    p_mol = g("mu_convert") / g("mu_mol") * g("k_B") * T_MOL
    assert p_mol * n[j + 1] == pytest.approx(p_neu * n[j], rel=1e-12)       # pressure continuous at the boundary
    K, F_rad_sh, P_out, L8 = S.layer_terms(sp, g("R2"), g("Pb"), p)          # P_c = Pb: the shell's inner edge
    assert P_out == pytest.approx(p_mol * n[-1], rel=1e-12)
    assert abs(L8) < 1e-9


def _auto_slab(m_over_m_ion, g0):
    """The slab with step (b) on (TShell_neu 'auto') and a far-UV field of about g0 Habing at R_IF ~ 1 pc. The dust is a millionth
    of the default (dust_sigma 1.5e-27 cm^2, tauShell_mol 3e-6, so the cold switch stays at 2e21 cm^-2): with the default dust and
    R2 = 1e-3 pc, Ln's radiation force would confine the ionised layer to ~1e-3 pc and pin T_atom at the table's 5000 K corner (the
    package's second cross-check, PDR_DEFAULT_PLAN.md). Premises checked by an independent CGS integration of the same equations
    (2026-11-01): R_IF 1.00 / 1.00 / 0.99 pc and the zones each test asserts."""
    p, m_ion, _ = _slab(m_over_m_ion, TShell_neu="auto", TShell_mol=T_MOL, tauShell_mol=3e-6, dust_sigma=1.5e-27 * cvt.cm2pc**2)
    p["Ln"].value = g0 * 4 * math.pi * cvt.pc2cm**2 * 1.6e-3 / 0.67 * cvt.L_cgs2au
    return p, m_ion, shell_structure_pure(p)


def test_step_b_three_zones_on_the_slab():
    """TShell_neu 'auto' (step (b), PDR_DEFAULT_PLAN.md G-B1) at G0 ~ 100: warm atomic gas at the PDR temperature of the front's
    field and density, warm H2 from Bialy & Sternberg's column, cold gas beyond tauShell_mol; every jump pressure-continuous; L8."""
    import trinity.shell_structure.pdr as pdr
    p, _, sp = _auto_slab(10.0, 1e2)
    g = lambda k: float(p[k].value)  # noqa: E731
    i, j, c = int(sp.shell_ion_idx), int(sp.shell_mol_idx), int(sp.shell_cold_idx)
    assert sp.has_neutral and i < j < c, "premise: three zones"
    n = np.asarray(sp.shell_n_arr, float)
    T = float(sp.shell_T_atom)
    jump_IF = g("mu_atom") / g("mu_ion_shell") * g("TShell_ion")
    assert T == pytest.approx(pdr.t_pdr(float(sp.shell_G0_IF), float(sp.n_IF) * cvt.ndens_au2cgs, jump_IF), rel=1e-12)
    assert 50.0 < float(sp.shell_G0_IF) < 200.0 and T_MOL < T < 1000.0
    assert n[i + 1] / n[i] == pytest.approx(jump_IF / T, rel=1e-12)          # the front
    assert n[j + 1] / n[j] == pytest.approx(g("mu_mol") / g("mu_atom"), rel=1e-12)   # H2 forms: 11/6, same T
    assert n[c + 1] / n[c] == pytest.approx(T / T_MOL, rel=1e-12)            # the gas turns cold: same mu
    K, F_rad_sh, P_out, L8 = S.layer_terms(sp, g("R2"), g("Pb"), p)
    assert P_out == pytest.approx(g("mu_convert") / g("mu_mol") * g("k_B") * T_MOL * n[-1], rel=1e-12)
    assert abs(L8) < 1e-6


def test_step_b_warm_h2_without_a_cold_zone_closes_L8():
    """The mass runs out between the H2 switch and the cold one: beyond shell_mol_idx the gas is warm H2 at the atomic temperature
    (shell_cold_idx -1), and swept.layer_terms must price it so -- as cold gas, K_s and P(r_s) would be off by T_atom/T_mol and
    L8 of order 1 (the package's first review, B2)."""
    p, _, sp = _auto_slab(1.5, 1e2)
    g = lambda k: float(p[k].value)  # noqa: E731
    j, c = int(sp.shell_mol_idx), int(sp.shell_cold_idx)
    assert sp.has_neutral and j > int(sp.shell_ion_idx) and c == -1, "premise: H2 forms, the cold switch is not reached"
    n = np.asarray(sp.shell_n_arr, float)
    K, F_rad_sh, P_out, L8 = S.layer_terms(sp, g("R2"), g("Pb"), p)
    assert P_out == pytest.approx(g("mu_convert") / g("mu_mol") * g("k_B") * float(sp.shell_T_atom) * n[-1], rel=1e-12)
    assert abs(L8) < 1e-6


def test_step_b_cold_switch_first():
    """At G0 ~ 1e5 Bialy & Sternberg's column (~2.8e21 cm^-2) lies beyond the cold switch (2e21): the H2 switch is dropped and the
    gas turns molecular and cold at once, as in the a2 layer (shell_cold_idx == shell_mol_idx), but at the PDR temperature."""
    p, _, sp = _auto_slab(10.0, 1e5)
    g = lambda k: float(p[k].value)  # noqa: E731
    j, c = int(sp.shell_mol_idx), int(sp.shell_cold_idx)
    assert sp.has_neutral and j > int(sp.shell_ion_idx) and c == j, "premise: one switch, molecular and cold together"
    n = np.asarray(sp.shell_n_arr, float)
    T = float(sp.shell_T_atom)
    assert n[j + 1] / n[j] == pytest.approx(g("mu_mol") / g("mu_atom") * T / T_MOL, rel=1e-12)
    K, F_rad_sh, P_out, L8 = S.layer_terms(sp, g("R2"), g("Pb"), p)
    assert abs(L8) < 1e-6


def test_step_b_layer_terms_without_neutral_gas():
    """TShell_neu 'auto' on a shell that is all ionised (no neutral zone, so no PDR temperature in this solve): layer_terms must
    run (it is called on every swept root) and give the ionised layer's pressure (the package's first review, B1)."""
    p, _, sp = _slab(0.05, TShell_neu="auto", TShell_mol=T_MOL, tauShell_mol=DTAU)
    g = lambda k: float(p[k].value)  # noqa: E731
    assert not sp.has_neutral, "premise: 0.05 m_ion is all ionised"
    K, F_rad_sh, P_out, L8 = S.layer_terms(sp, g("R2"), g("Pb"), p)
    n = np.asarray(sp.shell_n_arr, float)
    assert P_out == pytest.approx(g("mu_convert") / g("mu_ion_shell") * g("k_B") * g("TShell_ion") * n[-1], rel=1e-12)


def test_validators_refuse_bad_two_zone_keys():
    """A negative TShell_mol would read as 'off' without a word; tauShell_mol <= 0 would switch at the front itself."""
    from trinity._input.errors import ParameterFileError
    from trinity._input.registry import SPECS, _validate_TShell_mol, _validate_tauShell_mol
    spec = {s.name: s for s in SPECS}
    assert spec["TShell_mol"].validator is _validate_TShell_mol
    assert spec["tauShell_mol"].validator is _validate_tauShell_mol
    _validate_TShell_mol(0.0, None)
    _validate_TShell_mol(20.0, None)
    _validate_tauShell_mol(3.0, None)
    for bad in (-1.0, True, "20"):
        with pytest.raises(ParameterFileError):
            _validate_TShell_mol(bad, None)
    for bad in (0.0, -2.0, True):
        with pytest.raises(ParameterFileError):
            _validate_tauShell_mol(bad, None)
    from trinity._input.registry import _validate_TShell_neu, _validate_fFUV_Ln
    assert spec["TShell_neu"].validator is _validate_TShell_neu and spec["fFUV_Ln"].validator is _validate_fFUV_Ln
    _validate_TShell_neu(100.0, None)
    _validate_TShell_neu("auto", {"TShell_mol": 20.0})
    for bad in (0.0, -1.0, True, "300"):
        with pytest.raises(ParameterFileError):
            _validate_TShell_neu(bad, None)
    with pytest.raises(ParameterFileError, match="set TShell_mol 0 and a numeric TShell_neu"):   # 'auto' needs two zones
        _validate_TShell_neu("auto", {"TShell_mol": 0.0})
    with pytest.raises(ParameterFileError, match="below TShell_mol"):        # D-F6: no atomic zone colder than the molecular one
        _validate_TShell_neu(10.0, {"TShell_mol": 20.0})
    _validate_TShell_neu(10.0, {"TShell_mol": 0.0})                          # one zone at 10 K stays allowed
    with pytest.raises(ParameterFileError, match="Invalid TShell_mol"):      # D-F8: reported as TShell_mol's error
        _validate_TShell_neu("auto", {"TShell_mol": -1.0})
    _validate_fFUV_Ln(0.67, None)
    for bad in (0.0, 1.5, -0.1, True):
        with pytest.raises(ParameterFileError):
            _validate_fFUV_Ln(bad, None)


def test_step_b_outputs_are_registered_and_stored():
    """D-F2 (P5): shell_T_atom, shell_G0_IF and shell_N_tran are registered per-row keys; the 'auto' solve fills them (the transition
    column is pdr.n_tran_bs16's) and updateDict copies them into params, so every row of dictionary.jsonl stores them."""
    from trinity._input.dictionary import updateDict
    from trinity._input.registry import SPECS
    keys = ("shell_T_atom", "shell_G0_IF", "shell_N_tran")
    spec = {s.name: s for s in SPECS}
    assert all(spec[k].category == "runtime_shell" and not spec[k].exclude_from_snapshot for k in keys)
    p, _, sp = _auto_slab(10.0, 1e2)
    assert np.isfinite(sp.shell_T_atom) and sp.shell_G0_IF > 0 and sp.shell_N_tran > 0
    updateDict(p, sp)
    assert all(p[k].value == getattr(sp, k) for k in keys)


def test_flip_warning_and_the_one_zone_recipe(tmp_path, caplog):
    """D-B5 (P5): a .param with a numeric TShell_neu and no TShell_mol is told that it now sets the atomic zone of the two-zone layer;
    TShell_mol 0 with a numeric TShell_neu (the one-zone layer of earlier runs) is not; TShell_mol 0 alone is refused, because the
    default TShell_neu 'auto' needs two zones."""
    from trinity._input import read_param as RP
    from trinity._input.errors import ParameterFileError

    def load(name, **over):
        keys = {"model_name": name, "log_console": "False", "path2output": str(tmp_path / "out")}
        keys.update({k: str(v) for k, v in over.items()})
        par = tmp_path / f"{name}.param"
        par.write_text("\n".join(f"{k} {v}" for k, v in keys.items()) + "\n")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=RP.logger.name):
            P = read_param(str(par))
        return P, [r for r in caplog.records if "Set TShell_mol 0 for the one-zone layer" in r.getMessage()]

    P, warned = load("old", TShell_neu=100)
    assert P["TShell_mol"].value == 20 and len(warned) == 1
    P, warned = load("recipe", TShell_neu=100, TShell_mol=0)
    assert P["TShell_mol"].value == 0 and P["TShell_neu"].value == 100 and not warned
    with pytest.raises(ParameterFileError, match="set TShell_mol 0 and a numeric TShell_neu"):
        load("half", TShell_mol=0)
    with pytest.raises(ParameterFileError, match="below TShell_mol"):        # D-F6: an old 10 K file is refused, not inverted
        load("cold", TShell_neu=10)


def test_sweep_parent_warns_once_for_a_numeric_tshell_neu():
    """D-F7 (P5): a sweep discards its children's stderr, so the parent names the combinations whose numeric TShell_neu has no
    TShell_mol (they now get the two-zone layer); 'auto', a set TShell_mol and an absent TShell_neu are not counted."""
    from trinity._input.sweep_runner import two_zone_flip_warning
    combos = [({"TShell_neu": 100.0}, "a"), ({"TShell_neu": 300.0, "TShell_mol": 0.0}, "b"), ({"TShell_neu": "auto"}, "c"),
              ({}, "d")]
    msg = two_zone_flip_warning(combos)
    assert msg is not None and msg.startswith("WARNING: 1/4 combinations") and "TShell_mol 0" in msg
    assert two_zone_flip_warning(combos[1:]) is None
