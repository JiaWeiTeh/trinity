#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phii_scheme 'swept' in the implicit phase (docs/dev/phii-identity PLAN.md 0.0.11, Step 6, registered 2026-10-25).

A full run at production defaults (the stress fixture's config, mCloud 1e4, sfe 0.1, nCore 5e2) under 'swept', to
stop_t 0.0036 Myr, past the 1a -> 1b boundary at 3e-3 Myr; in its own process, since trinity keeps module-level state.
On every implicit row: the shell identity closes with the inner face at P_drive (the bubble's own pressure),
rs_dyn sits on the shell solve's rShell, the outer edge outruns the gas (U >= u), and the beta/delta solve was fed the
wall speed (cool_alpha R2 / t), since Step 7b-1 the implicit-function tangent of the R2 constraint coupled with the energy
balance (test_tangent_* below). The mass never decreases; 1a is c3c's.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_PARAM = ("mCloud 1e4\nsfe 0.1\nnCore 5e2\nstop_t 0.0036\nphii_scheme swept\n"
          "model_name sweptfix\nlog_console False\n")


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("swept1b")
    (tmp / "fixture.param").write_text(_PARAM)
    res = subprocess.run([sys.executable, str(REPO / "run.py"), str(tmp / "fixture.param")], cwd=tmp,
                         capture_output=True, text=True, timeout=1800)
    assert res.returncode == 0, res.stderr[-3000:]
    return [json.loads(ln) for ln in (tmp / "outputs" / "sweptfix" / "dictionary.jsonl").read_text().splitlines()
            if ln.strip()]


def test_swept_implicit_rows_close_the_shell_and_carry_the_wall_speed(rows):
    en = [r for r in rows if r["current_phase"] == "energy"]
    im = [r for r in rows if r["current_phase"] == "implicit"]
    assert en and len(im) >= 3
    assert all(r["rs_dyn"] == 0.0 for r in en)                               # 1a stays c3c
    for r in im:
        outer = 4 * math.pi * r["rShell"] ** 2 * r["P_outer"]
        assert abs((4 * math.pi * r["R2"] ** 2 * r["P_drive"] + r["K_s"] + r["F_rad_sh"] - outer) / outer) < 1e-6
        assert abs(r["rs_dyn"] / r["rShell"] - 1) < 1e-8
        assert r["F_ram"] == pytest.approx(4 * math.pi * r["R2"] ** 2 * r["P_drive"], rel=1e-12)
        assert r["rs_dot"] >= r["v2"] > 0
        assert r["P_HII"] == 0.0 and r["P_ram"] == 0.0
    # the wall speed the beta/delta solve saw (cool_alpha R2 / t) is finite and positive while the shell expands;
    # since Step 7b-1 it is the coupled tangent, checked against the runner's own values in the test below
    for a, b in zip(im[:-1], im[1:]):
        assert math.isfinite(b["cool_alpha"]) and b["cool_alpha"] > 0
        assert b["shell_mass"] >= a["shell_mass"]
    assert len({r["cool_beta"] for r in im}) > 1                              # beta/delta was solved, not held


# The 1b -> 1c hand-off on the no-physical-root exit (PLAN.md Step 7 item (6), gate H1, registered 2026-10-26).
# The runner forces that exit on the second implicit segment and spies on ImplicitSwept.advance; the transition
# runner must start from the last state advance carried, not from the wall speed set at the segment start.
_HANDOFF_DRIVER = r'''
import json, runpy, sys
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as imp
import trinity.phase2_momentum.swept as swept
calls, last, seen = [0], {}, {}
def forced(window):
    calls[0] += 1
    return calls[0] >= 2
imp.unsolvable_handoff = forced
_start = swept.ImplicitSwept.segment_start
def segment_start(self, params, fb, t, Eb):
    root = _start(self, params, fb, t, Eb)
    last['wall'] = self.Rdot2                     # what this segment's start put in v2
    return root
swept.ImplicitSwept.segment_start = segment_start
_advance = swept.ImplicitSwept.advance
def advance(self, params, y, r_max):
    _advance(self, params, y, r_max)
    last.update(rs=self.rs, u=self.u, Ms=self.Ms)
    last.pop('wall', None)
swept.ImplicitSwept.advance = advance
_transition = swept.run_phase_transition_swept
def transition(params):
    if not seen:
        seen.update(rs=params['rs_dyn'].value, u=params['v2'].value, Ms=params['shell_mass'].value, last=dict(last),
                    calls=calls[0], flags=str(params['solver_flags'].value))
        with open('handoff.json', 'w') as f:
            json.dump(seen, f)
    return _transition(params)
swept.run_phase_transition_swept = transition
sys.argv = ['run.py', 'fixture.param']
runpy.run_path(%r, run_name='__main__')
'''


def test_no_root_handoff_carries_the_swept_state(tmp_path):
    (tmp_path / "fixture.param").write_text(_PARAM)
    (tmp_path / "driver.py").write_text(_HANDOFF_DRIVER % str(REPO / "run.py"))
    res = subprocess.run([sys.executable, "driver.py"], cwd=tmp_path, capture_output=True, text=True, timeout=1800,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))})
    assert res.returncode == 0, res.stderr[-3000:]
    h = json.loads((tmp_path / "handoff.json").read_text())
    assert h["last"], "advance never ran before the forced exit"
    assert h["calls"] == 2 and "no_physical_root_handoff" in h["flags"], "1b did not leave by the forced exit"
    assert h["last"]["wall"] != h["last"]["u"], "the segment-start wall speed equals u: the test could not see the bug"
    assert (h["rs"], h["u"], h["Ms"]) == (h["last"]["rs"], h["last"]["u"], h["last"]["Ms"])


# Step 7b-1 (PLAN.md, registered 2026-10-26, gate G7b1 (1)): the implicit-function tangent of the R2 constraint.
# The fixture run spies on swept.tangent_partials (one-sided values against central differences at the same state, signs,
# the trial keys put back) and on ImplicitSwept.to_params (the wall speed and the closure's miss at each snapshot).
_TANGENT_DRIVER = r'''
import atexit, dataclasses, json, runpy, sys
import numpy as np
import trinity.phase2_momentum.swept as S
from trinity.phase1b_energy_implicit.get_betadelta import compute_R1_Pb
rec = {"partials": [], "rows": [], "fallback_calls": [], "live": [], "coupled": []}
FAIL_AT = %d
_tp, calls = S.tangent_partials, [0]
def tp(R2, rs, Ms, Eb, fb, params, t, sp, h=S.TAN_STEP):
    calls[0] += 1
    if calls[0] == FAIL_AT:
        rec["fallback_calls"].append(t)
        return None
    keys = {k: params[k].value for k in S._TRIAL_KEYS}
    fbkeys = {f.name: params[f.name].value for f in dataclasses.fields(fb) if f.name in params}
    out = _tp(R2, rs, Ms, Eb, fb, params, t, sp, h)
    restored = all(params[k].value == v for k, v in keys.items())
    fb_restored = all(np.array_equal(params[k].value, v) for k, v in fbkeys.items())
    ro = lambda a, e: float(S._shell_at(a, rs, Ms, fb, params, Eb=e, phase="implicit", t=t)[0].rShell)
    Nc = (ro(R2 * (1 + 1e-4), Eb) - ro(R2 * (1 - 1e-4), Eb)) / (2e-4 * R2)
    DEc = (ro(R2, Eb * (1 + 1e-4)) - ro(R2, Eb * (1 - 1e-4))) / (2e-4 * Eb)
    def ro_t(tt):                                    # the shell at time tt, with the feedback of tt
        fbt = S.get_current_sps_feedback(tt, params)
        S.updateDict(params, fbt)
        return float(S._shell_at(R2, rs, Ms, fbt, params, Eb=Eb, phase="implicit", t=tt)[0].rShell)
    DTc = (ro_t(t * (1 + 1e-4)) - ro_t(t * (1 - 1e-4))) / (2e-4 * t)
    S.updateDict(params, fb)
    for k, v in keys.items():
        params[k].value = v
    rec["partials"].append(dict(out=out, Nc=Nc, DEc=DEc, DTc=DTc, restored=restored, fb_restored=fb_restored, rs=rs, Eb=Eb, t=t))
    return out
S.tangent_partials = tp
_to = S.ImplicitSwept.to_params
def to_params(self, params, fb, Eb):
    _to(self, params, fb, Eb)
    rec["rows"].append(dict(t=params["t_now"].value, Rdot2=self.Rdot2, secant=self.zeta * self.U,
                            tan=self.tan is not None, miss=self.miss_eps))
    if self.tan is not None:      # what an independent coupled wall speed needs: the partials and beta/delta's final luminosities
        R2 = self.root["R2"]
        Pb = compute_R1_Pb(R2, Eb, params["Lmech_total"].value, params["v_mech_total"].value, params["gamma_adia"].value)[1]
        rec["coupled"].append(dict(fed=self.Rdot2, U=self.U, Mdot=self.Mdot, k=4 * S.math.pi * R2**2 * Pb,
                                   Lg=params["bubble_Lgain"].value, Ll=params["bubble_Lloss"].value,
                                   **{q: self.tan[q] for q in ("N", "DE", "DM", "DT")}))
S.ImplicitSwept.to_params = to_params
_rhs = S.ImplicitSwept.rhs
def rhs(self, params, Ed, Td):
    f = _rhs(self, params, Ed, Td)
    if self.tan is not None:      # Step 7b-2: dEb/dt at the segment start, and what it needs to be checked independently
        R2, Eb = self.root["R2"], self.Eb0
        Pb = compute_R1_Pb(R2, Eb, params["Lmech_total"].value, params["v_mech_total"].value, params["gamma_adia"].value)[1]
        t0, y0 = params["t_now"].value, self.y0(Eb, params["T0"].value)
        d0 = f(t0, y0)
        # an interior state, 1e-5 Myr into the segment (1b segments are >= 5e-5 Myr): dR2/dt by a central difference of the
        # closure along the RHS, the work rate k1 from compute_R1_Pb at the closure's R2 and the feedback at t1
        t1, y1 = t0 + 1e-5, y0 + 1e-5 * d0
        d1 = f(t1, y1)
        fb1 = S.get_current_sps_feedback(t1, params)
        def g(h):
            r = y1[0] + d1[0] * h
            return self._closure(r, y1[2] + d1[2] * h, S.swept_mass(r, d1[0], self.Ms, y1[1] <= 0.0, params)[0], t1 + h)
        R2t = g(0.0)
        k1 = 4 * S.math.pi * R2t**2 * compute_R1_Pb(R2t, y1[2], fb1.Lmech_total, fb1.v_mech_total, params["gamma_adia"].value)[1]
        rec["live"].append(dict(Edot0=float(d0[2]), Ed=Ed, fed=self.Rdot2, k0=4 * S.math.pi * R2**2 * Pb, U=self.U,
                                Mdot=self.Mdot, Edot1=float(d1[2]), k1=k1, rd1=(g(1e-7) - g(-1e-7)) / 2e-7,
                                **{k: self.tan[k] for k in ("N", "DE", "DM", "DT")}))
    return f
S.ImplicitSwept.rhs = rhs
atexit.register(lambda: json.dump(rec, open("tangent.json", "w")))
sys.argv = ["run.py", "fixture.param"]
runpy.run_path(%r, run_name="__main__")
'''


@pytest.fixture(scope="module")
def tangent(tmp_path_factory):
    return _tangent_run(tmp_path_factory.mktemp("tangent"))


def _tangent_run(tmp_path, fail_at=0):
    (tmp_path / "fixture.param").write_text(_PARAM)
    (tmp_path / "driver.py").write_text(_TANGENT_DRIVER % (fail_at, str(REPO / "run.py")))
    res = subprocess.run([sys.executable, "driver.py"], cwd=tmp_path, capture_output=True, text=True, timeout=1800,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))})
    assert res.returncode == 0, res.stderr[-3000:]
    rows = [json.loads(ln) for ln in (tmp_path / "outputs" / "sweptfix" / "dictionary.jsonl").read_text().splitlines() if ln.strip()]
    return json.loads((tmp_path / "tangent.json").read_text()), [r for r in rows if r["current_phase"] == "implicit"]


def test_tangent_partials_and_wall_speed(tangent):
    rec, im = tangent
    assert len(rec["partials"]) >= 5 and len(im) >= 5
    for p in rec["partials"]:
        out = p["out"]
        assert out is not None and p["restored"]
        assert out["N"] > 0 and out["DE"] <= 0                                   # thicker skin for more pressure
        assert out["N"] == pytest.approx(p["Nc"], rel=1e-3)
        # D_E to 1e-3, or to what the shell solve's ~1e-8 resolution in r_out allows for the 1e-4 nudge: on this thin shell
        # the nudge moves r_out by ~3e-7 pc and one- and two-sided values differ by ~2e-9 pc (5e-9 of r_out), in two
        # quantised levels -- the solve's own tolerance, not a defect (amended 2026-10-26, PLAN Step 7b-1)
        assert out["DE"] == pytest.approx(p["DEc"], rel=1e-3, abs=1e-8 * p["rs"] / (1e-4 * p["Eb"]))
        # D_t: here ~3e-8 pc/Myr, so the 1e-4 nudge moves r_out by ~1e-14 pc, ~100 ulp of r_out: one- and two-sided values agree to
        # what double precision resolves, 8 ulp of r_s over the nudge (~2e-9 pc/Myr; measured <= 1.5e-10). A sign error is 2 |D_t|.
        assert out["DT"] == pytest.approx(p["DTc"], rel=1e-3, abs=8 * sys.float_info.epsilon * p["rs"] / (1e-4 * p["t"]))
        assert p["fb_restored"]                                                   # the feedback of t, not of t (1 + h)
    by_t = {r["t"]: r for r in rec["rows"]}
    differs = 0
    for r in im:
        s = by_t[r["t_now"]]
        assert s["tan"]
        assert r["cool_alpha"] * r["R2"] / r["t_now"] == pytest.approx(s["Rdot2"], rel=1e-12)   # the speed beta/delta saw
        differs += abs(s["Rdot2"] / s["secant"] - 1) > 1e-6
    assert differs >= 1                                                           # the tangent, not the secant
    # the fed speed against the coupled formula, rebuilt here from the partials and beta/delta's final luminosities: within the
    # recouple dead-band (1 %); without the coupling it is off by 1.3-1.5 % here, measured
    fin = [c for c in rec["coupled"] if math.isfinite(c["Lg"]) and math.isfinite(c["Ll"])]
    assert len(fin) >= 5
    for c in fin:
        v = (c["U"] - c["DE"] * (c["Lg"] - c["Ll"]) - c["DM"] * c["Mdot"] - c["DT"]) / (c["N"] - c["DE"] * c["k"])
        assert abs(c["fed"] / v - 1) <= 0.01
    assert all(abs(s["miss"]) <= 1e-3 for s in rec["rows"][1:])                    # the closure predicts the next root


def test_tangent_falls_back_to_the_secant(tmp_path):
    rec, im = _tangent_run(tmp_path, fail_at=3)
    assert len(rec["fallback_calls"]) == 1
    t_fail = rec["fallback_calls"][0]
    s = [r for r in rec["rows"] if r["t"] == t_fail]
    assert s and not s[0]["tan"] and s[0]["Rdot2"] == pytest.approx(s[0]["secant"], rel=1e-12)
    assert len(im) >= 5                                                           # the run goes on


def test_live_work_at_segment_start(tangent):
    """Step 7b-2: inside a 1b segment dEb/dt = Ed - (4 pi R2^2 Pb dR2/dt - W0), dR2/dt the closure's own derivative. At the
    start, with dR2/dt rebuilt from the RHS's dEb/dt through the chain rule, dEb/dt = Ed - k0 (dR2/dt - Rdot2_fed); inside
    the segment, with dR2/dt a finite difference of the closure along the RHS and Pb recomputed there, the same balance."""
    rec, _ = tangent
    assert len(rec["live"]) >= 5
    live = 0
    for s in rec["live"]:
        rd0 = (s["U"] - s["DE"] * s["Edot0"] - s["DM"] * s["Mdot"] - s["DT"]) / s["N"]
        W0 = s["k0"] * s["fed"]
        assert s["Edot0"] == pytest.approx(s["Ed"] - s["k0"] * (rd0 - s["fed"]), rel=0, abs=1e-10 * abs(W0))
        live += abs(s["Edot0"] - s["Ed"]) > 1e-9 * abs(W0)
        assert s["Edot1"] == pytest.approx(s["Ed"] + W0 - s["k1"] * s["rd1"], rel=0, abs=1e-7 * abs(W0))
        assert abs(s["k1"] / s["k0"] - 1) > 1e-9                                  # off the root: the work rate has moved
    assert live >= 1                                                              # the work term is live, not frozen


def test_live_rates_closed_form():
    """Step 7b-2: live_rates solves the closure's derivative and the energy balance together (magnitudes of ROS, 1b k = 28)."""
    import trinity.phase2_momentum.swept as S
    tan = dict(N=1.1726200469295913, DE=-1.8533470806565276e-06, DM=6.374092083063056e-06, DT=0.000596334556614344)
    Ed, k, fed, U, Md = 1.2e6, 80970.54090017421, 24.2218612084034, 26.9, 5443.642384515056
    rd, Edot = S.live_rates(Ed, k * fed, k, U, Md, tan, 0.95)
    assert Edot == pytest.approx(Ed - k * (rd - fed), rel=1e-12)
    assert tan["N"] * rd + tan["DE"] * Edot + tan["DM"] * Md + tan["DT"] == pytest.approx(U, rel=1e-12)
    rd, Edot = S.live_rates(Ed, k * fed, k, U, Md, None, 0.95)                  # the secant: dR2/dt = zeta U
    assert rd == 0.95 * U and Edot == pytest.approx(Ed - k * (0.95 * U - fed), rel=1e-12)


def test_escape_switch_cut():
    """Step 7b-2: the escape indicator and the segment cut at its predicted zero."""
    from types import SimpleNamespace as NS
    import trinity.phase2_momentum.swept as S
    assert S.escape_indicator(NS(shell_fAbsorbedIon=1.0, R_IF=0.9, rShell=1.0)) == pytest.approx(0.1)    # trapped
    assert S.escape_indicator(NS(shell_fAbsorbedIon=0.95, R_IF=1.0, rShell=1.0)) == pytest.approx(-0.05)  # escaping
    assert S.switch_dt(None, None, 0.01, 1.0, 2.0) == 2.0                       # first segment: no trend yet
    assert S.switch_dt(0.02, 0.0, 0.01, 1.0, 2.0) == pytest.approx(1.05)        # trapped -> escaping at t_x = 2
    assert S.switch_dt(-0.02, 0.0, -0.01, 1.0, 2.0) == pytest.approx(1.05)      # escaping -> trapped at t_x = 2
    assert S.switch_dt(0.02, 0.0, 0.01, 1.0, 0.5) == 0.5                        # the switch lies beyond the segment
    assert S.switch_dt(0.01, 0.0, 0.02, 1.0, 2.0) == 2.0                        # moving away from the switch
    assert S.switch_dt(0.01, 0.0, -0.01, 1.0, 2.0) == 2.0                       # already crossed: on the new branch
    assert S.switch_dt(0.02, 1.0, 0.01, 1.0, 2.0) == 2.0                        # no time between the two roots
    sw = S.ImplicitSwept.__new__(S.ImplicitSwept)                               # the cut as the runner applies it
    sw.s_prev, sw.t_prev, sw.s_esc, sw.t_seg, sw.n_switch, sw.miss_eps, sw.n_capped = 0.02, 0.0, 0.01, 1.0, 0, 0.0, 0
    assert sw.cap_dt(2.0, 1e-4) == 2.0 and sw.n_capped == 0                      # the miss cap does not see the switch
    assert sw.switch_cut(2.0, 1e-4) == pytest.approx(1.05) and sw.n_switch == 1
    assert sw.switch_cut(2.0, 1.5) == 1.5 and sw.n_switch == 2                  # floored, still shorter than dt: counted
    assert sw.switch_cut(2.0, 2.0) == 2.0 and sw.n_switch == 2                  # floor at dt: no cut, not counted


def test_miss_cap_shortens_against_the_segment_that_missed():
    """Step 7b-1 step control, as fixed 2026-10-27: after a miss above 1e-3 the next segment is a fraction of the one that missed,
    even when the runner has already grown the proposal past it; never below the floor; counted only when it cuts."""
    import trinity.phase2_momentum.swept as S
    sw = S.ImplicitSwept.__new__(S.ImplicitSwept)
    sw.t_prev, sw.t_seg, sw.n_capped, sw.miss_eps = 0.0, 1.0, 0, 2e-3          # the segment that missed ran 1.0
    f = 0.9 * math.sqrt(1e-3 / 2e-3)
    assert sw.cap_dt(1.26, 1e-4) == pytest.approx(f) and sw.n_capped == 1        # grown proposal 1.26 -> 0.64 of the missed 1.0
    assert sw.cap_dt(1.26, 0.7) == 0.7 and sw.n_capped == 2                     # floored, still a cut
    assert sw.cap_dt(0.5, 1e-4) == 0.5 and sw.n_capped == 2                     # already shorter: unchanged, not counted
    sw.miss_eps = 1.0
    assert sw.cap_dt(2.0, 1e-4) == pytest.approx(0.3)                           # the factor's lower bound
    sw.miss_eps = 5e-4
    assert sw.cap_dt(2.0, 1e-4) == 2.0                                          # below the threshold: no cap
