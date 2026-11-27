"""The switch package (docs/dev/turnaround-stall/PLAN_SWITCH.md): D-11, vented is a held state (vent_channel) set
where blowout fires, at 1b's exit or at the end of a 1c or momentum segment; R-a, the rt count is carried after 1b;
R-e, a late vent's cooling is scaled from 1c's entry state. Hand-built params for the helpers; the 1c runner on a
sealed shell past the cloud edge (the last two tests, T3: the gate configs never vent inside 1c after 1b)."""
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
import trinity.bubble_structure.get_bubbleParams as G
import trinity.phase1c_transition.run_transition_phase as TP
import trinity.phase2_momentum.swept as S
from trinity._input.read_param import read_param
from trinity._input.registry import SPECS
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp
from trinity.phase1b_energy_implicit.run_energy_implicit_phase import rt_count_step
from trinity.sps import read_sps
from trinity.sps.update_feedback import get_current_sps_feedback

REPO = Path(__file__).resolve().parents[1]
RCLOUD, LN100 = 10.0, math.log(100.0)


def _ns(**kw):
    """Hand-built params: {key: object with .value}, as the helpers read them."""
    return {k: SimpleNamespace(value=v) for k, v in kw.items()}


def _sealed(**kw):
    """A sealed shell under vent_model reservoir with the blowout route on: the keys blowout_after_1b reads."""
    base = dict(vent_model='reservoir', transition_trigger='cooling_balance,blowout', blowout_rule='geometric',
                transition_channel='cooling_balance', vent_channel='', t_vent=np.nan, isVented=False,
                _rt_N=0.0, _rt_on=False, rt_growth_factor=100.0, rCloud=RCLOUD, PISM=1.0, k_B=1.0)
    base.update(kw)
    return _ns(**base)


def test_vented_reads_the_held_state():
    p = _sealed()
    assert G.vented(p) is False
    p['vent_channel'].value = 'blowout'
    assert G.vented(p) is True                  # 1b's exit reason (cooling_balance here) no longer decides
    p['vent_model'].value = 'vacuum'
    assert G.vented(p) is False


def test_hand_built_params_without_the_key_read_the_old_rule():
    assert G.vented(_ns(vent_model='ambient', transition_channel='blowout')) is True
    assert G.vented(_ns(vent_model='ambient', transition_channel='cooling_balance')) is False
    assert G.vented(_ns(vent_model='ambient', transition_channel='blowout', vent_channel='')) is False


def test_mark_vented_sets_the_three_keys():
    p = _sealed()
    G.mark_vented(p, 'blowout', 0.25, 'momentum', 12.0, 11.0)
    assert (p['vent_channel'].value, p['t_vent'].value, p['isVented'].value) == ('blowout', 0.25, True)
    assert G.vented(p) is True


@pytest.mark.parametrize('kw, want', [
    (dict(), True),
    (dict(vent_model='ambient'), True),
    (dict(vent_model='vacuum'), False),                       # D-11: skipped under vacuum (the code as it was)
    (dict(transition_trigger='cooling_balance'), False),      # no blowout route
    (dict(transition_trigger='r1'), True),                    # r1 = blowout,ebpeak
    (dict(vent_channel='blowout'), False),                    # vented already: held, nothing to test
    (dict(start_scheme='auto'), True),                        # R-f guard: the watch ignores the start scheme
])
def test_blowout_watch(kw, want):
    assert S.blowout_watch(_sealed(**kw)) is want


def test_geometric_fires_on_the_end_radius():
    p = _sealed()
    assert S.blowout_after_1b(p, 'momentum', 1.0, 5.0, 11.0, 9.0, 1.0, 1.1, 5.0, 11.5, 0.999 * RCLOUD) is False
    assert (p['vent_channel'].value, p['isVented'].value) == ('', False)
    assert S.blowout_after_1b(p, 'momentum', 1.1, 5.0, 11.5, 9.9, 1.0, 1.2, 5.0, 12.0, 1.001 * RCLOUD) is True
    assert (p['vent_channel'].value, p['t_vent'].value, p['isVented'].value) == ('blowout', 1.2, True)


def test_rt_carries_the_count_and_vents_as_in_1b():
    a, dt = 4.0e3, 0.01                          # a push [pc Myr^-2] over a segment [Myr]
    rs0, R2s = 1.2 * RCLOUD, 1.2 * RCLOUD - 0.05
    delta = max(rs0 - R2s, 1e-6 * R2s)
    p = _sealed(blowout_rule='rt', _rt_N=0.9 * LN100, _rt_on=True)
    N0 = p['_rt_N'].value
    # a decelerating segment adds nothing and does not vent
    assert S.blowout_after_1b(p, 'transition', 1.0, 10.0, rs0, R2s, 2.0, 1.0 + dt, 9.0, rs0, R2s) is False
    assert p['_rt_N'].value == N0
    # a push adds 1b's term sqrt(a / delta) dt (2.83 here) and crosses ln 100 with Pb > PISM k_B and r_s > rCloud
    assert S.blowout_after_1b(p, 'transition', 1.0, 10.0, rs0, R2s, 2.0, 1.0 + dt, 10.0 + a * dt, rs0, R2s) is True
    assert p['_rt_N'].value == rt_count_step(N0, 1.0, 10.0, 1.0 + dt, 10.0 + a * dt, delta)
    assert p['vent_channel'].value == 'blowout' and p['t_vent'].value == 1.0 + dt


def test_rt_needs_the_overpressure_and_the_edge_outside():
    big = dict(blowout_rule='rt', _rt_N=10.0 * LN100, _rt_on=True)
    p = _sealed(**big)                           # Pb 0.5 below PISM k_B = 1
    assert S.blowout_after_1b(p, 'momentum', 1.0, 10.0, 12.0, 11.0, 0.5, 1.01, 10.0, 12.0, 11.0) is False
    p = _sealed(**big)                           # the edge back inside the cloud: the count holds, no vent
    assert S.blowout_after_1b(p, 'momentum', 1.0, 10.0, 9.0, 8.0, 2.0, 1.01, 10.0, 9.0, 8.0) is False
    p = _sealed(blowout_rule='rt')               # not started, edge inside: the count stays off
    assert S.blowout_after_1b(p, 'momentum', 1.0, 10.0, 9.0, 8.0, 2.0, 1.01, 50.0, 9.0, 8.0) is False
    assert p['_rt_on'].value is False and p['_rt_N'].value == 0.0
    assert p['vent_channel'].value == ''


def test_the_new_keys():
    spec = {s.name: s for s in SPECS}
    for k in ('vent_channel', 't_vent', '_rt_N', '_rt_on'):
        assert spec[k].category == 'runtime_control' and spec[k].exclude_from_snapshot, k
    assert spec['isVented'].category == 'runtime_control' and not spec['isVented'].exclude_from_snapshot
    assert spec['isVented'].default is False and spec['vent_channel'].default == ''
    assert math.isnan(spec['t_vent'].default)


def _late_entry(tmp_path, rule):
    """test_vent_reservoir.py's transition-entry state (simple_cluster at 0.5 Myr, 10 km/s), but the bubble's inner face
    2 per cent PAST rCloud, P_th = 10 P_ram, and the shell SEALED: 1b ended on cooling_balance under vent_model
    reservoir with the blowout route; C_f 0.9 from the vent on (cover_law constant); PISM 1e3 K cm^-3, far below Pb.
    rt: the count carried from 1b is past ln G (it passed ln G while the edge was back inside the cloud, where a shell
    keeps its count but does not vent; PLAN_RT.md 2026-11-26), so the first check past the edge vents."""
    txt = re.sub(r"^(path2output\s+).*$", lambda m: m.group(1) + str(tmp_path),
                 (REPO / "param" / "simple_cluster.param").read_text(), flags=re.M)
    if "path2output" not in txt:
        txt += f"\npath2output    {tmp_path}\n"
    (tmp_path / "sc.param").write_text(txt)
    p = read_param(str(tmp_path / "sc.param"))
    get_InitCloudProp(p)
    sps = read_sps.read_sps(p["mCluster"].value / p["sps_refmass"].value, p)
    p["sps_data"].value, p["sps_f"].value = sps, read_sps.get_interpolation(sps)
    t0, R2 = 0.5, 1.02 * p["rCloud"].value
    fb = get_current_sps_feedback(t0, p)
    P_ram = G.pRam(R2, fb.Lmech_total, fb.v_mech_total)
    p["t_now"].value, p["stop_t"].value, p["current_phase"].value = t0, 0.52, "transition"
    p["R2"].value, p["v2"].value, p["rShell"].value = R2, 10.0 * cvt.v_kms2au, 1.2 * R2
    p["shell_mass"].value = S._scalar(S.mass_profile.get_mass_profile(R2, p))
    p["Eb"].value = 10.0 * P_ram * (4 * math.pi / 3) * R2**3 / (p["gamma_adia"].value - 1)
    p["bubble_LTotal"].value, p["bubble_Tavg"].value = 0.3 * fb.Lmech_total, 3e6
    p["cool_alpha"].value = t0 / R2 * p["v2"].value
    p["phii_scheme"].value = "swept"
    p["vent_model"].value, p["transition_trigger"].value = "reservoir", "cooling_balance,blowout"
    p["blowout_rule"].value, p["transition_channel"].value = rule, "cooling_balance"
    p["coverFraction"].value, p["cover_law"].value = 0.9, "constant"
    p["PISM"].value = 1.0e3 * cvt.ndens_cgs2au          # P/k in code units after read_param (K pc^-3)
    if rule == "rt":
        p["_rt_on"].value, p["_rt_N"].value = True, 1.5 * LN100
    return p


@pytest.mark.parametrize("rule", ["geometric", "rt"])
def test_a_sealed_shell_vents_in_1c_and_takes_the_reservoir_branch(tmp_path, rule):
    """D-13: past the edge at 1c's first segment end, the next row is vented; the wall is sealed (C_f 1) before it and
    leaks through 1 - 0.9 from it on; R-e: the cooling of every vented row is L0 (E/E_in)^2 V_in/V with 1c's entry."""
    p = _late_entry(tmp_path, rule)
    L0 = p["bubble_LTotal"].value
    TP.run_phase_transition(p)
    p.flush()
    rows = [json.loads(ln) for f in tmp_path.rglob("dictionary.jsonl") for ln in f.read_text().splitlines()
            if ln.strip()]
    tr = [r for r in rows if r["current_phase"] == "transition"]
    k = next((i for i, r in enumerate(tr) if r["isVented"]), None)
    assert k is not None and k >= 1, "the sealed shell never vented in 1c"
    assert not any(r["isVented"] for r in tr[:k]) and all(r["isVented"] for r in tr[k:])
    assert p["vent_channel"].value == "blowout" and p["t_vent"].value == tr[k]["t_now"]
    assert all(r["shell_fLeak"] == 0.0 for r in tr[:k])                               # sealed before the vent
    assert all(r["shell_fLeak"] == pytest.approx(0.1, abs=1e-12) for r in tr[k:])     # C_f 0.9 from the vent row on
    assert tr[k]["bubble_Leak"] > 0.0                                                 # the leak branch from the vent row
    E_in, V_in = tr[0]["Eb"], tr[0]["R2"] ** 3 - tr[0]["R1"] ** 3                     # R-e: 1c's entry state
    for r in tr[k:]:
        cool = L0 * (r["Eb"] / E_in) ** 2 * V_in / (r["R2"] ** 3 - r["R1"] ** 3)
        assert r["bubble_Lloss"] - r["bubble_Leak"] == pytest.approx(cool, rel=1e-9)
