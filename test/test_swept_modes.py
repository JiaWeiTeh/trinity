"""B7 (docs/dev/turnaround-stall/PLAN_B7.md, ruled 2026-11-21): the swept mass in two modes, SWEEP and COAST, switched
only at terminal events (S7: sweep_mode, mode_rates, mode_event, edge_speed's sweep branch), and root_R2's thin-shell
limit (S10). On simple_cluster's own parameters, as test_phii_swept_golden.py."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.integrate

import trinity._functions.unit_conversions as cvt
import trinity.phase2_momentum.swept as S
from trinity._input.read_param import read_param
from trinity.cloud_properties import mass_profile
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def params():
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    get_InitCloudProp(p)
    # a plausible young cluster (as test_phii_swept_golden.py): Q = 1e51 s^-1, L_bol = 3e40 erg/s
    p["Qi"].value = 1e51 / cvt.s2Myr
    p["Li"].value, p["Ln"].value, p["Lbol"].value = (x * cvt.L_cgs2au for x in (1e40, 2e40, 3e40))
    return p


def _M(r, p):
    return S._scalar(mass_profile.get_mass_profile(r, p))


def test_sweep_mode_coasts_while_receding_and_sweeps_only_past_unswept_gas(params):
    r = 0.5 * params["rCloud"].value
    M = _M(r, params)
    for carried in (None, True, False):
        assert S.sweep_mode(carried, r, 0.0, M, params) is False          # at rest or receding: COAST
        assert S.sweep_mode(carried, r, -2.0, 0.5 * M, params) is False
    assert S.sweep_mode(True, r, 3.0, 2.0 * M, params) is True            # a carried SWEEP stays one
    assert S.sweep_mode(False, r, 3.0, M, params) is False                # COAST at the swept edge: its event decides
    assert S.sweep_mode(False, r, 3.0, 0.5 * M, params) is True           # COAST past unswept gas (a missed event)
    assert S.sweep_mode(None, r, 3.0, M * (1 + 1e-10), params) is True    # phase entry: the hand-off's fixed point
    assert S.sweep_mode(None, r, 3.0, 2.0 * M, params) is False


def test_mode_rates_hold_the_mass_in_coast_and_sweep_it_at_the_jump_speed(params):
    r = 0.5 * params["rCloud"].value
    M = _M(r, params)
    assert S.mode_rates(3.0, r, 0.7 * M, False, 0.5, params) == (3.0, 0.7 * M, 0.0)
    U, m, md = S.mode_rates(3.0, r, 0.7 * M, True, 0.5, params)
    m2, md2 = mass_profile.get_mass_profile(r, params, return_mdot=True, rdot=U)
    assert U == S.edge_speed(3.0, r, 0.5, params) and m == S._scalar(m2) == M
    assert md == S._scalar(md2) > 0


def test_edge_speed_sweep_branch_continues_the_jump_through_u0(params):
    """In SWEEP the jump continues smoothly through u = 0 (the old step of c0 there pinned LSODA); without sweep the
    edge still moves with its gas for u <= 0 (D-T1). Photons escape inside the cloud, so c0 = c_i (to edge_speed's
    1e-9 allowance)."""
    r = 0.5 * params["rCloud"].value
    c_i_sq = params["k_B"].value * params["TShell_ion"].value / params["mu_ion_shell"].value
    for u in (-1e-3, 0.0, 1e-3):
        jump = 0.5 * (u + math.sqrt(u * u + 4.0 * c_i_sq))
        assert S.edge_speed(u, r, 0.5, params, sweep=True) == pytest.approx(jump, rel=1e-9)
    assert S.edge_speed(0.0, r, 0.5, params) == 0.0 and S.edge_speed(-1e-3, r, 0.5, params) == -1e-3
    assert S.edge_speed(3.0, r, 0.5, params, sweep=True) == S.edge_speed(3.0, r, 0.5, params)


def test_mode_event_ends_sweep_at_u0_and_coast_at_unswept_gas(params):
    r = 0.5 * params["rCloud"].value
    M = _M(r, params)
    ev = S.mode_event(True, M, params)
    assert (ev.terminal, ev.direction, ev.name) == (True, -1, "mode_switch")
    assert ev(0.0, np.array([r, 2.5])) == 2.5
    ev = S.mode_event(False, M, params)
    assert (ev.terminal, ev.direction, ev.name) == (True, 1, "mode_switch")
    assert ev(0.0, np.array([r, 1.0])) < 0.0 < ev(0.0, np.array([1.01 * r, 1.0]))


def test_a_coast_segment_ends_where_the_edge_reaches_unswept_gas(params):
    r1 = 0.5 * params["rCloud"].value
    ev = S.mode_event(False, _M(r1, params), params)
    sol = scipy.integrate.solve_ivp(lambda t, y: np.array([r1, 0.0]), (0.0, 1.0), np.array([0.9 * r1, 1.0]),
                                    events=[ev], rtol=1e-10, atol=1e-12)
    assert sol.status == 1 and sol.t_events[0].size == 1
    assert sol.y_events[0][0][0] == pytest.approx(r1, rel=1e-9)


def test_root_R2_takes_the_thin_shell_limit(params, monkeypatch):
    """S10: the residual rShell(R2) - r_s is >= 0 at R2 = r_s, so one still < 0 at the last-resort upper end
    r_s (1 - 1e-9) puts the root inside the last 1e-9 r_s; root_R2 returns that end instead of 'no bracket'. A shell
    thicker than that keeps brentq's root. The shell solve is replaced by a shell of fixed thickness d."""
    rs = 0.0157                                                           # pc, a dense start (B7c's K_017)
    fb = SimpleNamespace(Lmech_total=0.0, v_mech_total=0.0, Qi=0.0)
    monkeypatch.setattr(S, "layer_terms", lambda sp, R2, Pc, p: (0.0, 0.0, 0.0, 0.0))
    for frac, want, rel in ((1e-12, rs * (1 - 1e-9), 0.0), (1e-6, rs * (1 - 1e-6), 1e-9)):
        monkeypatch.setattr(S, "_shell_at", lambda R2, rs_, Ms, fb_, p, Eb=None, phase="transition", t=None,
                            d=frac * rs: (SimpleNamespace(rShell=R2 + d), 0.0))
        root = S.root_R2(rs, 1.0, fb, params)
        assert root["ok"], root
        assert root["R2"] == pytest.approx(want, rel=rel, abs=0.0)
