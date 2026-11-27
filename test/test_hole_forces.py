"""hole_forces (docs/dev/leaky-reservoir/C1D_PLAN.md, rulings D-1 to D-9 of 2026-11-26 and the v2 rulings of §10.3):
every force on the shell sees the holes in it; 'on' is the default, and under it the cf_law reads the outbound column
(the smallest swept column on the way out, since #796: test_cf_outbound.py) and a run ends as a disrupted shell (end
code 5) once C_f falls below cf_disrupt or the real mass is gone. Under 'on' the swept equation of motion is du/dt = (k F - G M_r/r^2 (M_* + m_ev + M_r/2) - k Mdot u)/M_r
with (k, M_r) = (C_f, C_f m) for holes open from birth and (C_f, m - D) after the vent of a leaky reservoir, D the gas the
holes did not sweep; phase 1a puts its self-gravity on C_f m; the IR depth becomes C_f tau/(1 + (1 - C_f) tau). These
tests pin the helpers (limits, the plan's numbers, the deficit quadrature), that 'off' and C_f = 1 give today's terms bit
for bit, the key and its validator, phase 1a's snapshot, RHS and F_grav column, and that a phase-1c reservoir run's
right-hand side uses the real mass and C_f (per call under cover_law sigma). The momentum RHS and the 1b RHS with holes
from birth are checked on full runs (gates G3, G4b of C1D_PLAN §9.3), not here."""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
import trinity.bubble_structure.get_bubbleParams as G
import trinity.phase1_energy.energy_phase_ODEs as ODEs
import trinity.phase1c_transition.run_transition_phase as TP
import trinity.phase2_momentum.swept as S
from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param
from trinity._input.registry import SPECS, _validate_cf_disrupt, _validate_hole_forces
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp
from trinity.sps import read_sps
from trinity.sps.update_feedback import get_current_sps_feedback

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def _ns(**kw):
    """Hand-built params: {key: object with .value}, as the validators and helpers read them."""
    return {k: SimpleNamespace(value=v) for k, v in kw.items()}


def _vented(cf=0.5, **kw):
    """A vented leaky reservoir under hole_forces on with the deficit's keys at their defaults."""
    base = dict(hole_forces="on", vent_model="reservoir", transition_channel="blowout", coverFraction=cf,
                _hole_D=0.0, _hole_Ms_seg=0.0, _hole_Cf_seg=1.0, _cf_tab_r=None, _cf_tab_S=None)
    base.update(kw)
    return _ns(**base)


# ---------------------------------------------------------------- the IR depth (C1D_PLAN §4.2, D-3)

def test_ir_depth_is_tau_at_cf_one_and_has_the_plans_limits():
    tau = np.logspace(-8.0, 8.0, 20001)
    assert all(G.ir_depth(float(t), 1.0) == float(t) for t in tau)          # exactly: G2 needs no branch
    for cf in (0.1, 0.5, 0.9):
        assert G.ir_depth(1e-8, cf) == pytest.approx(cf * 1e-8, rel=1e-7)   # thin: only the covered sky returns light
        assert G.ir_depth(1e8, cf) == pytest.approx(cf / (1.0 - cf), rel=1e-6)   # thick
        d = np.array([G.ir_depth(float(t), cf) for t in tau])
        assert np.all(np.diff(d) > 0.0) and np.all(d < tau)                 # holes only ever lose IR photons
    # the plan's table at C_f = 0.5 (C1D_PLAN §4.2)
    for t, want in ((0.1, 0.048), (1.0, 0.333), (5.0, 0.714), (20.0, 0.909)):
        assert G.ir_depth(t, 0.5) == pytest.approx(want, abs=5e-4)


def test_ir_depth_is_the_hit_count_through_holes():
    """1 + tau_eff is the number of shell hits of a photon whose first hit is on the covered sky: after each hit it
    leaves outward with chance T = 1/(1 + tau), else it crosses the cavity and hits the shell again with chance C_f, so
    the hits are sum_n (C_f (1 - T))^n = 1/(1 - C_f (1 - T)); C_f = 1 gives today's 1 + tau."""
    for cf in (0.3, 0.7, 1.0):
        for tau in (0.2, 3.0, 40.0):
            T = 1.0 / (1.0 + tau)
            hits = sum((cf * (1.0 - T)) ** n for n in range(5000))
            assert 1.0 + G.ir_depth(tau, cf) == pytest.approx(hits, rel=1e-9)


# ---------------------------------------------------------------- (k, M_r): off, holes from birth, after a vent

def test_off_gives_todays_terms_bit_for_bit():
    """Off (or the key absent): hole_mass gives (1.0, Ms - m_ev). The second loop pins only the IEEE identity the off
    path rests on (x * 1.0, x / 1.0 exact, operation order as in swept.py): it is arithmetic, not the code line; the
    code sites are pinned by the runner tests below and on full runs by the gate G0 (C1D_PLAN §9.3)."""
    rng = np.random.default_rng(20261126)
    for p in ({}, _ns(hole_forces="off", vent_model="reservoir", transition_channel="blowout", coverFraction=0.5)):
        for _ in range(2000):
            Ms, m_ev, r = 10 ** rng.uniform(1, 8), float(rng.choice([0.0, 10 ** rng.uniform(-2, 0)])), 10 ** rng.uniform(-1, 3)
            k, Mr = G.hole_mass(p, r, Ms, m_ev)
            assert k == 1.0 and Mr == Ms - m_ev
    Gc = 4.49e-3
    for _ in range(20000):
        F, md, v = rng.normal() * 10 ** rng.uniform(0, 12), 10 ** rng.uniform(-3, 6), rng.normal() * 10 ** rng.uniform(-3, 3)
        m, m_ev, r, mC = 10 ** rng.uniform(1, 8), 10 ** rng.uniform(-3, 0), 10 ** rng.uniform(-1, 3), 10 ** rng.uniform(2, 7)
        mn = m - m_ev
        k, Mr = 1.0, mn
        old = (F - Gc * mn / r**2 * (mC + m_ev + 0.5 * mn) - md * v) / mn
        new = (k * F - Gc * Mr / r**2 * (mC + m_ev + 0.5 * Mr) - k * md * v) / Mr
        assert new == old


def test_holes_from_birth_keep_the_per_steradian_forces():
    """vent_model vacuum/ambient, coverFraction 0.4: hole_mass gives k = C_f, M_r = C_f m; the identity below (arithmetic,
    not the code line: the 1b RHS is checked on full runs by G3 and the F_grav column by G4b) is why only the
    self-gravity changes, to G (M_* + m_ev + C_f m/2)/r^2 (Harper-Clark & Murray 2009, eq. 9)."""
    for vm in ("vacuum", "ambient"):
        p = _ns(hole_forces="on", vent_model=vm, coverFraction=0.4, transition_channel="blowout")
        k, Mr = G.hole_mass(p, 3.0, 1e5, 1e3)
        assert (k, Mr) == (0.4, 0.4 * (1e5 - 1e3))
        mn, Gc, F, mC = 1e5 - 1e3, 4.49e-3, 7e7, 1e4
        a = (k * F - Gc * Mr / 9.0 * (mC + 1e3 + 0.5 * Mr)) / Mr
        assert a == pytest.approx((F - Gc * mn / 9.0 * (mC + 1e3 + 0.5 * 0.4 * mn)) / mn, rel=1e-13)


def test_reservoir_is_sealed_until_the_vent():
    p = _vented(transition_channel="")                      # 1a/1b of a reservoir: C_f = 1, no deficit
    assert G.hole_mass(p, 3.0, 1e5, 1e3) == (1.0, 1e5 - 1e3)
    G.hole_deficit(p, 1e5)
    assert p["_hole_Ms_seg"].value == 0.0 and p["_hole_D"].value == 0.0


# ---------------------------------------------------------------- the deficit D (C1D_PLAN §4, D-4)

def test_deficit_starts_at_the_vent_and_is_exact_for_constant_cf():
    p = _vented(0.5)
    assert G.hole_mass(p, 3.0, 2e4, 0.0) == (0.5, 2e4)      # the vent row: D = 0 before the first segment start
    G.hole_deficit(p, 2e4)
    assert (p["_hole_D"].value, p["_hole_Ms_seg"].value, p["_hole_Cf_seg"].value) == (0.0, 2e4, 0.5)
    Ms = 2e4
    for dM in (1e3, 0.0, 3.3e3, 7e2):                       # a COAST segment adds nothing
        Ms += dM
        k, Mr = G.hole_mass(p, 3.0, Ms, 50.0)               # inside the segment (and at the next row) before the start
        G.hole_deficit(p, Ms)
        assert Mr == pytest.approx(Ms - 50.0 - 0.5 * (Ms - 2e4), rel=1e-14)
    assert p["_hole_D"].value == pytest.approx(0.5 * (Ms - 2e4), rel=1e-14)


def test_deficit_is_the_plans_quadrature_under_a_varying_cf():
    """The review's example (C1D_PLAN §4; data/c1d_deficit.csv): C_f falls linearly from 1 to 0.5 over r_v..2 r_v in a
    uniform ambient that adds Delta M = M_v. The per-segment sum gives the mass per covered steradian 3.49714, 3.41812
    and 3.39787 M_v for 5, 20 and 100 segments (exact 3.39286)."""
    want = {5: 3.49714, 20: 3.41812, 100: 3.39787}
    for N, per_covered in want.items():
        p = _vented(1.0)
        for s in np.linspace(1.0, 2.0, N + 1):
            p["coverFraction"].value = 1.0 - 0.5 * (s - 1.0)
            G.hole_deficit(p, 1.0 + (s**3 - 1.0) / 7.0)
        k, Mr = G.hole_mass(p, 2.0, 2.0, 0.0)
        assert k == 0.5 and Mr / k == pytest.approx(per_covered, abs=1e-5)


def test_deficit_is_a_no_op_off_and_with_holes_from_birth():
    for p in (_vented(0.5, hole_forces="off"), _vented(0.5, vent_model="ambient"), _vented(0.5, vent_model="vacuum")):
        G.hole_deficit(p, 3e4)
        assert p["_hole_Ms_seg"].value == 0.0 and p["_hole_D"].value == 0.0
    G.hole_deficit({}, 3e4)                                 # no keys at all: off


def test_the_law_reads_the_swept_column_under_on():
    """v2 (ruled 2026-11-26 21:1x, C1D_PLAN.md §10.3; D-4' superseded): under cover_law sigma the closed fraction reads
    the shell's SWEPT column, its swept mass less the evaporated gas, per RHS call (hole_mass) and for the row
    (shell_column), as with the key off: the still-covered directions sweep as every direction does without holes. The
    deficit D still sets the real mass the forces act on."""
    p = _vented(1.0, cover_law="sigma", cf_half_radius=2.0, _cf_Sigma_vent=0.0, rs_dyn=12.0, shell_mass=3.0e4,
                current_phase="transition", shell_mass_net=False)
    G.hole_deficit(p, 3.0e4)                                # the vent row: D = 0, the law not yet stored (C_f = 1)
    assert p["_hole_Cf_seg"].value == 1.0
    G.store_vent_column(p)
    Sv = p["_cf_Sigma_vent"].value
    assert Sv == 3.0e4 / (4.0 * np.pi * 12.0**2)
    p["_hole_Cf_seg"].value = 0.6                           # as if the next rows had opened 40 % of the sky
    p["shell_mass"].value, p["rs_dyn"].value = 4.0e4, 20.0
    swept = 4.0e4 / (4.0 * np.pi * 20.0**2)
    assert G.shell_column(p) == swept                       # the row's column: no deficit in it
    k, Mr = G.hole_mass(p, 20.0, 4.0e4, 0.0)
    assert k == G.cf_law(Sv / swept, 2.0)
    assert Mr == pytest.approx(4.0e4 - 0.4 * 1.0e4, rel=1e-15)   # the real mass: the forces see D
    off = _vented(1.0, hole_forces="off", cover_law="sigma", cf_half_radius=2.0, _cf_Sigma_vent=Sv, rs_dyn=20.0,
                  shell_mass=4.0e4, current_phase="transition", shell_mass_net=False)
    assert G.cover_fraction(off) == k                       # one law for both keys
    assert G.hole_mass(off, 20.0, 4.0e4, 0.0) == (1.0, 4.0e4)


def test_store_vent_column_is_untouched():
    """D-9 as checked against the code (C1D_PLAN §9.1): the deficit starts from vented(), so the cf_law hook keeps its
    signature (params only) and its once-only, cover_law-sigma-only behaviour, also under hole_forces on, and it never
    writes the deficit's keys. Since 2026-11-27 it also starts the outbound table (CF_LAW_PLAN.md §11)."""
    assert list(inspect.signature(G.store_vent_column).parameters) == ["params"]
    p = _vented(0.5, cover_law="constant", _cf_Sigma_vent=0.0, rs_dyn=10.0, shell_mass=1e4)
    G.store_vent_column(p)
    assert p["_cf_Sigma_vent"].value == 0.0 and p["_hole_Ms_seg"].value == 0.0
    p = _vented(1.0, cover_law="sigma", cf_half_radius=2.0, _cf_Sigma_vent=0.0, rs_dyn=10.0, shell_mass=1e4,
                current_phase="transition", shell_mass_net=False)
    G.hole_deficit(p, 1e4)                                  # the vent row, as _forces_to_params ends it
    held = (p["_hole_D"].value, p["_hole_Ms_seg"].value, p["_hole_Cf_seg"].value)
    G.store_vent_column(p)
    Sv = p["_cf_Sigma_vent"].value
    assert Sv == 1e4 / (4.0 * np.pi * 10.0**2)
    assert (p["_cf_tab_r"].value, p["_cf_tab_S"].value) == ([10.0], [Sv])   # the outbound table starts here
    p["shell_mass"].value, p["rs_dyn"].value = 2e4, 15.0
    G.store_vent_column(p)                                  # once only
    assert p["_cf_Sigma_vent"].value == Sv
    assert p["_cf_tab_r"].value == [10.0]
    assert (p["_hole_D"].value, p["_hole_Ms_seg"].value, p["_hole_Cf_seg"].value) == held


# ---------------------------------------------------------------- the radiation force and phase 1a

def test_f_rad_sees_the_holes_only_when_on():
    sp = SimpleNamespace(isDissolved=False, shell_fAbsorbedWeightedTotal=0.8, shell_tauKappaRatio=3.0)
    base = dict(Lbol=1e8, c_light=306.6, dust_KappaIR=2.0)
    old = 0.8 * 1e8 / 306.6 * (1.0 + 3.0 * 2.0)
    assert S.f_rad(sp, _ns(**base)) == old                  # the key absent
    assert S.f_rad(sp, _ns(hole_forces="off", **base), 0.5) == old
    assert S.f_rad(sp, _ns(hole_forces="on", **base), 1.0) == old
    assert S.f_rad(sp, _ns(hole_forces="on", **base), 0.5) == pytest.approx(0.8 * 1e8 / 306.6 * (1.0 + 0.5 * 6.0 / (1.0 + 0.5 * 6.0)), rel=1e-14)


def test_phase_1a_snapshot_holds_the_self_gravity_factor():
    assert ODEs.ODESnapshot.__dataclass_fields__["self_gravity_cf"].default == 1.0


def test_phase_1a_sees_the_holes(tmp_path):
    """Phase 1a through the real create_ODE_snapshot, get_ODE_Edot_pure and compute_derived_quantities (simple_cluster,
    R2 = 0.6 rCloud, 10 km/s at 0.5 Myr, P_th = 3 P_ram, tau_IR = 2): off gives today's F_rad and F_grav to the bit; on,
    with holes from birth (ambient, coverFraction 0.4), the snapshot holds self_gravity_cf = 0.4 and F_rad with the IR
    depth, and the RHS's acceleration and the F_grav column move by exactly the self-gravity change; a reservoir's 1a is
    sealed (C_f = 1), so on gives off's acceleration there."""
    p = _params(tmp_path)
    t0, R2 = 0.5, 0.6 * p["rCloud"].value
    fb = get_current_sps_feedback(t0, p)
    S.updateDict(p, fb)
    p["t_now"].value, p["current_phase"].value, p["phii_scheme"].value = t0, "energy", "swept"
    p["R2"].value = p["rShell"].value = R2
    p["shell_mass"].value = m = S._scalar(S.mass_profile.get_mass_profile(R2, p))
    p["shell_fAbsorbedIon"].value, p["c_sound"].value = 1.0, 100.0 * cvt.v_kms2au
    Eb = 3.0 * G.pRam(R2, fb.Lmech_total, fb.v_mech_total) * (4 * math.pi / 3) * R2**3 / (p["gamma_adia"].value - 1)
    y = [R2, 10.0 * cvt.v_kms2au, Eb]
    sp = SimpleNamespace(isDissolved=False, shell_fAbsorbedWeightedTotal=0.8,
                         shell_tauKappaRatio=2.0 / p["dust_KappaIR"].value)
    tau = sp.shell_tauKappaRatio * p["dust_KappaIR"].value

    def at(**keys):
        for key, v in keys.items():
            p[key].value = v
        snap = ODEs.create_ODE_snapshot(p, sp)
        return snap, ODEs.get_ODE_Edot_pure(t0, y, snap, p), ODEs.compute_derived_quantities(t0, y, snap, p)

    off, a_off, d_off = at(hole_forces="off", vent_model="ambient", coverFraction=0.4)
    on, a_on, d_on = at(hole_forces="on")
    sealed, a_sealed, _ = at(vent_model="reservoir")
    Gc, mC, Fr = p["G"].value, p["mCluster"].value, 0.8 * p["Lbol"].value / p["c_light"].value
    assert d_off.shell_mass == m and off.evaporated_mass == 0.0          # 1a: the shell keeps all its gas
    assert off.self_gravity_cf == 1.0 and off.F_rad == Fr * (1.0 + tau)
    assert d_off.F_grav == Gc * m / R2**2 * (mC + 0.5 * m)
    assert on.self_gravity_cf == 0.4
    assert on.F_rad == pytest.approx(Fr * (1.0 + 0.4 * tau / (1.0 + 0.6 * tau)), rel=1e-14)
    assert d_on.F_grav == pytest.approx(Gc * m / R2**2 * (mC + 0.5 * 0.4 * m), rel=1e-14)
    a_g = ODEs.get_ODE_Edot_pure(t0, y, dataclasses.replace(on, F_rad=off.F_rad), p)   # gravity alone
    assert a_g[1] - a_off[1] == pytest.approx(0.5 * 0.6 * Gc * m / R2**2, rel=1e-6)
    assert a_on[1] - a_g[1] == pytest.approx((on.F_rad - off.F_rad) / m, rel=1e-6)
    assert sealed.self_gravity_cf == 1.0 and sealed.F_rad == off.F_rad and a_sealed[1] == a_off[1]


# ---------------------------------------------------------------- the key and its validator

def test_key_and_runtime_keys():
    spec = {s.name: s for s in SPECS}
    h, d = spec["hole_forces"], spec["cf_disrupt"]
    assert h.default == "on" and h.validator is _validate_hole_forces          # the default since C1D_PLAN §10.3
    assert d.default == "0.1" and d.validator is _validate_cf_disrupt
    for x in (h, d):
        assert x.run_const is True and x.exclude_from_snapshot is True and x.category == "input_solver"
    for name, default in (("_hole_D", 0.0), ("_hole_Ms_seg", 0.0), ("_hole_Cf_seg", 1.0)):
        assert spec[name].default == default and spec[name].exclude_from_snapshot is True
        assert spec[name].category == "runtime_control"
    names = [s.name for s in SPECS]
    assert names.index("hole_forces") == names.index("cf_half_radius") + 1 == names.index("cf_disrupt") - 1
    assert spec["cover_law"].default == "sigma"                               # the default since C1D_PLAN §10.3


def test_validator(caplog):
    """v2: 'on' is the default, so it is accepted everywhere: a no-op outside phii_scheme swept (no refusal) and with
    no holes (no warning); on, a coverFraction below cf_disrupt is refused (the shell would end as disrupted at once)."""
    for v in ("off", "on"):
        _validate_hole_forces(v, _ns(phii_scheme="c3c", coverFraction=0.05))   # c3c: a no-op either way
    for bad in ("yes", True, None, "On"):
        with pytest.raises(ParameterFileError, match="Invalid hole_forces"):
            _validate_hole_forces(bad, _ns())
    with caplog.at_level(logging.WARNING):
        _validate_hole_forces("on", _ns(phii_scheme="swept", vent_model="vacuum", coverFraction=1.0))
        _validate_hole_forces("on", _ns())                                    # every key at its fallback
    assert caplog.text == ""
    with pytest.raises(ParameterFileError, match="below cf_disrupt"):
        _validate_hole_forces("on", _ns(phii_scheme="swept", coverFraction=0.05, cf_disrupt=0.1))
    _validate_hole_forces("off", _ns(phii_scheme="swept", coverFraction=0.05, cf_disrupt=0.1))
    _validate_hole_forces("on", _ns(phii_scheme="swept", coverFraction=0.1, cf_disrupt=0.1))   # at the bar: kept


def test_cf_disrupt_validator():
    for ok in (0.1, 1e-3, 0.5, 0.99):
        _validate_cf_disrupt(ok, _ns())
    for bad in (0.0, 1.0, -0.1, 1.5, True, "0.1", None, float("nan")):
        with pytest.raises(ParameterFileError, match="Invalid cf_disrupt"):
            _validate_cf_disrupt(bad, _ns())


def test_defaults_and_their_no_op_cases():
    """The registry's defaults are hole_forces on and cover_law sigma; hand-built params without the keys read 'off' and
    'constant' (C1D_PLAN.md §10.3). hole_forces acts under phii_scheme swept only. The law acts in the leaky reservoir
    only: under vacuum and ambient, vented or not, cover_fraction is coverFraction with sigma in force (the blocker the
    cf_law session found, 2026-11-26 21:30)."""
    assert G.hole_forces(_ns(hole_forces="on")) and G.hole_forces(_ns(hole_forces="on", phii_scheme="swept"))
    assert not G.hole_forces({}) and not G.hole_forces(_ns(hole_forces="on", phii_scheme="c3c"))
    assert G.cover_law({}) == "constant"
    for vm in ("vacuum", "ambient"):
        for ch in ("", "blowout"):
            p = _ns(cover_law="sigma", vent_model=vm, transition_channel=ch, coverFraction=0.77, cf_half_radius=2.0,
                    _cf_Sigma_vent=5.0, rs_dyn=10.0, shell_mass=1e4, current_phase="transition", shell_mass_net=False)
            assert G.cover_fraction(p) == 0.77 and G.cover_fraction(p, 3.0, 1e3) == 0.77


def test_disrupted():
    """The end of a disrupted shell: on, once the row's closed fraction is below cf_disrupt (0.1 when the key is
    absent); never off."""
    p = _vented(0.05)
    assert G.disrupted(p)
    p["cf_disrupt"] = SimpleNamespace(value=0.01)
    assert not G.disrupted(p)
    assert not G.disrupted(_vented(0.05, hole_forces="off")) and not G.disrupted(_vented(0.1))
    law = _vented(1.0, cover_law="sigma", cf_half_radius=2.0, _cf_Sigma_vent=1e3, rs_dyn=10.0, shell_mass=1e4,
                  current_phase="transition", shell_mass_net=False)          # a column far below the vent's
    assert G.cover_fraction(law) < 0.1 and G.disrupted(law)
    law["_cf_Sigma_vent"].value = 1e4 / (4.0 * np.pi * 100.0)                # the vent's own column: C_f = 1
    assert not G.disrupted(law)



# ---------------------------------------------------------------- guards (the final /xcheck of 2026-11-27, X1-X3)

def _latch_flip():
    """A vented reservoir in 1c whose shell_mass_net latch turns on only after D has grown: the bubble held 310 Msun,
    more than the swept mass, until the shell swept 311 (C_f 0.5, constant law)."""
    p = _vented(0.5, phii_scheme="swept", cover_law="constant", shell_mass_net=True, _shell_mass_net_on=False,
                current_phase="transition", bubble_mass=310.0, shell_mass=100.0, rs_dyn=5.0, cf_disrupt=0.1)
    for Ms in (100.0, 150.0, 200.0, 250.0, 300.0, 310.0):
        p["shell_mass"].value = Ms
        assert G.evaporated_mass(p) == 0.0
        G.hole_deficit(p, Ms)
    p["shell_mass"].value = 311.0
    return p


def test_no_real_mass_ends_as_a_disrupted_shell():
    """X1 (ruled 2026-11-27: the run ends as disrupted): m_ev jumps to the bubble's 310 Msun while D = 105 Msun stays,
    so M_r = 311 - 310 - 105 < 0. hole_mass holds M_r at 0 (no division by a negative mass) and the row is disrupted
    with its own message; the C_f message is unchanged; off, the same state is a shell of 1 Msun."""
    p = _latch_flip()
    m_ev = G.evaporated_mass(p)
    assert m_ev == 310.0
    G.refuse_no_shell(m_ev, 311.0)                                   # shell_mass_net alone sees 1 Msun left
    assert G.hole_mass(p, 6.0, 311.0, m_ev) == (0.5, 0.0)
    assert G.disrupted(p) and G.disrupted_reason(p).startswith("Shell disrupted: no real shell mass left")
    p["current_phase"].value = "momentum"                            # the momentum RHS passes m_ev = 0: a shell
    assert not G.disrupted(p)
    p["hole_forces"].value = "off"
    assert G.hole_mass(p, 6.0, 311.0, m_ev) == (1.0, 1.0) and G.disrupted_reason(p) == ""
    q = _vented(0.05, cf_disrupt=0.1)
    assert G.disrupted_reason(q) == "Shell disrupted: closed fraction C_f = 0.05 < cf_disrupt 0.1"


def test_cf_law_stays_positive_and_unchanged():
    """X2: ln C_f is floored at CF_LN_FLOOR, so a column that thins far in one segment (h just above 1) gives a C_f
    below cf_disrupt but above 0 (k = 0 divided before); everywhere above the floor the law is the min() form bit for
    bit."""
    assert 0.0 < G.cf_law(1.6, 1.0002) < 1e-300 and 0.0 < G.cf_law(1e300, 2.0) < 1e-300
    for h in (1.05, 1.5, 2.0, 3.0, 10.0):
        for x in np.logspace(-2.0, 6.0, 81):
            y = G._cf_log(np.log(float(x)), G.cf_offset(h), G.CF_SIGMA)
            if y > G.CF_LN_FLOOR:
                assert G.cf_law(float(x), h) == float(np.exp(min(0.0, y)))


def test_nan_fails_loudly():
    """X3: a NaN column gives a NaN C_f (min() read it as a sealed shell, 1.0); a NaN radius takes the outer branch of
    the outbound table instead of indexing past it."""
    assert math.isnan(G.cf_law(float("nan"), 2.0))
    tab = _ns(_cf_tab_r=[10.0, 12.0], _cf_tab_S=[5.0, 4.0])
    assert G.outbound_column(tab, float("nan"), 3.0) == 3.0 and G.outbound_column(tab, float("nan"), 6.0) == 4.0

# ---------------------------------------------------------------- the runner: phase 1c of a vented reservoir
# (the transition-entry state of test_cf_law.py: simple_cluster, R2 = 0.6 rCloud, 10 km/s at 0.5 Myr, P_th = 3 P_ram,
# entered as a swept 1b hands it over), cover_law constant, coverFraction 0.5

def _params(tmp_path):
    """simple_cluster.param read, its cloud set up and its SPS tables loaded, output into tmp_path."""
    txt = re.sub(r"^(path2output\s+).*$", lambda m: m.group(1) + str(tmp_path),
                 (REPO / "param" / "simple_cluster.param").read_text(), flags=re.M)
    if "path2output" not in txt:
        txt += f"\npath2output    {tmp_path}\n"
    (tmp_path / "sc.param").write_text(txt)
    p = read_param(str(tmp_path / "sc.param"))
    get_InitCloudProp(p)
    sps = read_sps.read_sps(p["mCluster"].value / p["sps_refmass"].value, p)
    p["sps_data"].value, p["sps_f"].value = sps, read_sps.get_interpolation(sps)
    return p


def _entry(tmp_path, holes, stop_t=0.55):
    p = _params(tmp_path)
    t0, R2 = 0.5, 0.6 * p["rCloud"].value
    fb = get_current_sps_feedback(t0, p)
    P_ram = G.pRam(R2, fb.Lmech_total, fb.v_mech_total)
    p["t_now"].value, p["stop_t"].value, p["current_phase"].value = t0, stop_t, "transition"
    p["R2"].value, p["v2"].value, p["rShell"].value = R2, 10.0 * cvt.v_kms2au, 1.2 * R2
    p["shell_mass"].value = S._scalar(S.mass_profile.get_mass_profile(R2, p))
    p["Eb"].value = 3.0 * P_ram * (4 * math.pi / 3) * R2**3 / (p["gamma_adia"].value - 1)
    p["bubble_LTotal"].value, p["bubble_Tavg"].value = 0.3 * fb.Lmech_total, 3e6
    p["cool_alpha"].value = t0 / R2 * p["v2"].value
    p["phii_scheme"].value = "swept"
    p["vent_model"].value, p["transition_channel"].value = "reservoir", "blowout"
    p["vent_channel"].value = "blowout"   # D-11: the held vent state (PLAN_SWITCH.md)
    p["coverFraction"].value, p["hole_forces"].value = 0.5, holes
    p["cover_law"].value = "constant"          # a fixed C_f (cover_law defaults to sigma since C1D_PLAN §10.3)
    S.updateDict(p, fb)
    rs, Ms, u = S.initial_state(p, fb, p["Eb"].value)
    p["rs_dyn"].value, p["rShell"].value, p["shell_mass"].value, p["v2"].value = rs, rs, Ms, u
    p["bubble_mass"].value, p["_shell_mass_net_on"].value = 0.1 * Ms, True   # evaporated gas: m_ev = 0.1 Ms
    return p


def _run(tmp_path, holes, monkeypatch, sigma_vent=None):
    """One phase-1c run from _entry; the rows of the transition and the RHS at each segment start, keyed by the start
    state (r_s, u): solve_segment re-runs a failed segment with Radau from the same y0 (phase_events.py), so keys pair
    rows with starts where a plain zip would shift. sigma_vent: cover_law sigma with coverFraction 1.0 and the column
    at the vent set to sigma_vent times the entry's real column (store_vent_column then keeps it)."""
    p = _entry(tmp_path, holes)
    if sigma_vent is not None:
        p["cover_law"].value, p["cf_half_radius"].value, p["coverFraction"].value = "sigma", 2.0, 1.0
        p["_cf_Sigma_vent"].value = sigma_vent * ((p["shell_mass"].value - p["bubble_mass"].value)
                                                  / (4.0 * math.pi * p["rs_dyn"].value ** 2))
        p["_cf_tab_r"].value, p["_cf_tab_S"].value = [p["rs_dyn"].value], [p["_cf_Sigma_vent"].value]
    starts, real_ivp = {}, S.scipy.integrate.solve_ivp

    def spy_ivp(*a, **kw):                   # the runner's segment solves: their RHS at the segment start
        if not a and len(kw.get("y0", ())) == 3 and "events" in kw:
            y0 = kw["y0"]
            starts[(float(y0[0]), float(y0[1]))] = kw["fun"](kw["t_span"][0], y0)
        return real_ivp(*a, **kw)

    monkeypatch.setattr(S.scipy.integrate, "solve_ivp", spy_ivp)
    TP.run_phase_transition(p)
    monkeypatch.undo()
    p.flush()
    rows = [json.loads(ln) for f in tmp_path.rglob("dictionary.jsonl") for ln in f.read_text().splitlines()
            if ln.strip()]
    tr = [r for r in rows if r["current_phase"] == "transition"]
    pairs = [(r, starts[(r["rs_dyn"], r["v2"])]) for r in tr if (r["rs_dyn"], r["v2"]) in starts]
    return p, tr, pairs


def _accel(r, k, Mr):
    """The swept RHS's du/dt at a segment start from the row's columns: k (F - F_grav - Mdot u) / M_r with F = F_ram +
    K_s - F_ion_in + F_rad (D-6: the row's F_grav is the gravity on M_r / k)."""
    F = r["F_ram"] + r["K_s"] - r["F_ion_in"] + r["F_rad"]
    return k * (F - r["F_grav"] - r["shell_massDot"] * r["v2"]) / Mr


def test_sigma_reservoir_law_reads_the_swept_column(tmp_path, monkeypatch):
    """cover_law sigma with hole_forces on, a column at the vent three times the entry's (C_f ~ 0.6 from the first
    row, as test_cf_law.py does): each row's shell_fLeak is 1 - C_f of the law at the smallest swept column of the
    rows so far (the outbound table, CF_LAW_PLAN.md §11: rows move out, the column grows, C_f is held), with the
    deficit rebuilt from the rows, D_k = D_{k-1} + (1 - C_f,{k-1}) (M_s,k - M_s,{k-1}), D = 0 at the first row; the
    RHS at each segment start uses the law's C_f at the row's SWEPT column as k (per call, D-5; v2) and the real mass
    M_s - m_ev - D; and the deficit grows."""
    p, tr, pairs = _run(tmp_path, "on", monkeypatch, sigma_vent=3.0)
    assert len(tr) >= 2 and len(pairs) >= 2
    m_b, Sv = p["bubble_mass"].value, p["_cf_Sigma_vent"].value
    assert all(b["rs_dyn"] > a["rs_dyn"] for a, b in zip(tr, tr[1:]))          # every row at a new radius
    D, prev, real, Smin = 0.0, None, {}, Sv
    for r in tr:
        if prev is not None:
            D += prev["shell_fLeak"] * (r["shell_mass"] - prev["shell_mass"])
        Mr = r["shell_mass"] - m_b - D
        Smin = min(Smin, (r["shell_mass"] - m_b) / (4.0 * math.pi * r["rs_dyn"] ** 2))
        cf = G.cf_law(Sv / Smin, 2.0)
        assert r["shell_fLeak"] == pytest.approx(1.0 - cf, abs=1e-9)
        real[(r["rs_dyn"], r["v2"])] = (cf, Mr)
        prev = r
    assert D > 0.0 and p["_hole_D"].value == pytest.approx(D, rel=1e-9)
    for r, d in pairs:
        cf, Mr = real[(r["rs_dyn"], r["v2"])]
        assert d[1] == pytest.approx(_accel(r, cf, Mr), rel=1e-7)


@pytest.mark.parametrize("holes", ["off", "on"])
def test_reservoir_rhs_uses_the_real_mass_and_cf(tmp_path, monkeypatch, holes):
    """At each segment start the RHS's acceleration is k (F_ram + K_s - F_ion_in + F_rad - F_grav - Mdot u) / M_r from
    the row's columns (D-6: F_grav is the gravity on M_r/k), with k = 0.5 and M_r = M_s - m_ev - D under 'on', D summed
    from the vent; k = 1 and M_r = M_s - m_ev under 'off'. The row's F_rad carries the holes' IR depth under 'on'."""
    p, tr, pairs = _run(tmp_path, holes, monkeypatch)
    assert len(tr) >= 2 and len(pairs) >= 2
    k = 0.5 if holes == "on" else 1.0
    Ms_v, m_ev = tr[0]["shell_mass"], p["bubble_mass"].value
    kappa = p["dust_KappaIR"].value
    for r, d in pairs:
        D = (1.0 - k) * (r["shell_mass"] - Ms_v)                # constant C_f: the sum is exact
        assert d[1] == pytest.approx(_accel(r, k, r["shell_mass"] - m_ev - D), rel=1e-7)
        tau = r["shell_tauKappaRatio"] * kappa
        tau_eff = G.ir_depth(tau, 0.5) if holes == "on" else tau
        assert r["F_rad"] == pytest.approx(r["shell_fAbsorbedWeightedTotal"] * r["Lbol"] / p["c_light"].value
                                           * (1.0 + tau_eff), rel=1e-12)
    if holes == "on":
        assert tr[-1]["shell_mass"] > Ms_v                      # the shell swept gas after the vent
        assert p["_hole_D"].value == pytest.approx(0.5 * (p["_hole_Ms_seg"].value - Ms_v), rel=1e-12)
    else:
        assert p["_hole_Ms_seg"].value == 0.0


def test_reservoir_run_ends_as_a_disrupted_shell(tmp_path, monkeypatch):
    """E2 (C1D_PLAN.md §10.3): with hole_forces on and a column at the vent a thousand times the entry's, the law opens
    the sky at once (C_f < 0.1 on the first row): the 1c runner writes that row and ends the run with
    SimulationEndCode.SHELL_DISRUPTED. The same state with the key off runs on (the end is the key's)."""
    from trinity._output.simulation_end import SimulationEndCode
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    p, tr, _ = _run(tmp_path / "on", "on", monkeypatch, sigma_vent=1e3)
    assert tr and 1.0 - tr[-1]["shell_fLeak"] < 0.1
    assert p["SimulationEndCode"].value == SimulationEndCode.SHELL_DISRUPTED.code == 5
    t_end = tr[-1]["t_now"]   # the crossing row, and the 1c exit row the runner writes at the same t after any end
    assert all(r["t_now"] == t_end for r in tr)   # here C_f ~ 4e-5 at once: the first row crosses (the full-run gate
    #                                               G-end checks crossings after many rows, in 1c and in momentum)
    q, tq, _ = _run(tmp_path / "off", "off", monkeypatch, sigma_vent=1e3)
    assert q["SimulationEndCode"].value != SimulationEndCode.SHELL_DISRUPTED.code
    assert SimulationEndCode.SHELL_DISRUPTED.outcome == "shell_disrupted" and SimulationEndCode.SHELL_DISRUPTED.is_clean()


def test_fate_readout_knows_a_disrupted_shell():
    """tools/bubble_fate.py reads end code 5 as a physical end of a disrupted shell, not as a numerical failure (the
    adversarial review of 2026-11-26 found it 'unresolved' before)."""
    import tools.bubble_fate as BF
    v = BF.classify({"end_outcome": "shell_disrupted", "end_code": 5, "v2_final": 10.0, "R2_final": 50.0,
                     "R2_max": 50.0, "t_final": 5.0, "rCloud": 20.0, "isCollapse": False})
    assert (v["shell"], v["stop"], v["report"], v["censored"]) == ("disrupted", "physical", "shell_disrupted", False)
    with pytest.raises(ParameterFileError, match="Invalid cf_disrupt"):   # a bad cf_disrupt gets its own message
        _validate_cf_disrupt(1.5, _ns())
    _validate_hole_forces("on", _ns(phii_scheme="swept", coverFraction=1.0, cf_disrupt=1.5))   # not the key's to report
