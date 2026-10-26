#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phii_scheme 'swept' in the implicit phase (docs/dev/phii-identity PLAN.md 0.0.11, Step 6, registered 2026-10-25).

A full run at production defaults (the stress fixture's config, mCloud 1e4, sfe 0.1, nCore 5e2) under 'swept', to
stop_t 0.0036 Myr, past the 1a -> 1b boundary at 3e-3 Myr; in its own process, since trinity keeps module-level state.
On every implicit row: the shell identity closes with the inner face at P_drive (the bubble's own pressure),
rs_dyn sits on the shell solve's rShell, the outer edge outruns the gas (U >= u), and the beta/delta solve was fed the
wall speed: cool_alpha R2 / t = zeta U, with zeta = Delta R2 / Delta r_s between consecutive rows (1 on the first),
which ties the stored R2, rs_dyn, rs_dot and cool_alpha together. The mass never decreases; 1a is c3c's.
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
    # the wall speed the beta/delta solve saw: cool_alpha R2 / t = zeta U
    assert im[0]["cool_alpha"] * im[0]["R2"] / im[0]["t_now"] == pytest.approx(im[0]["rs_dot"], rel=1e-12)
    for a, b in zip(im[:-1], im[1:]):
        zeta = (b["R2"] - a["R2"]) / (b["rs_dyn"] - a["rs_dyn"])
        assert b["cool_alpha"] * b["R2"] / b["t_now"] == pytest.approx(zeta * b["rs_dot"], rel=1e-9)
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
