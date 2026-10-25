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
