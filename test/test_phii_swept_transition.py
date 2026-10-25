#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phii_scheme 'swept' in the transition phase (docs/dev/phii-identity PLAN.md 0.0.11, Step 5, registered 2026-10-25).

The runner itself on simple_cluster's own parameters, from a transition-entry state inside the cloud (R2 = 0.6 rCloud,
10 km/s at 0.5 Myr, a hot bubble whose thermal pressure is a multiple f of the wind's ram pressure). Every transition
snapshot must close the shell identity with the inner face at P_conf = max(P_th, P_ram), keep rs_dyn on the shell
solve's rShell, sweep at 4 pi r_s^2 rho U, never lose mass, and report the drive the integrated RHS uses (checked at
each segment start: a trapezoid budget between snapshots is the wrong instrument here, the net force being a small
difference of large terms while P_th decays on the sound-crossing time, dt/tau ~ 0.5 on the first segment); and when
the bubble is spent (f small: ram-dominated or at the energy floor within a segment or two) the momentum runner must continue the transition's
(r_s, u, M_s) exactly, without a second row at the hand-off time.
"""
import json
import math
import re

import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
import trinity.phase1c_transition.run_transition_phase as TP
import trinity.phase2_momentum.run_momentum_phase as MP
import trinity.phase2_momentum.swept as S
from trinity._input.read_param import read_param
from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp
from trinity.sps import read_sps
from trinity.sps.update_feedback import get_current_sps_feedback

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def _entry(tmp_path, f_thermal, stop_t):
    txt = re.sub(r"^(path2output\s+).*$", lambda m: m.group(1) + str(tmp_path),
                 (REPO / "param" / "simple_cluster.param").read_text(), flags=re.M)
    if "path2output" not in txt:
        txt += f"\npath2output    {tmp_path}\n"
    (tmp_path / "sc.param").write_text(txt)
    p = read_param(str(tmp_path / "sc.param"))
    get_InitCloudProp(p)
    sps = read_sps.read_sps(p["mCluster"].value / p["sps_refmass"].value, p)
    p["sps_data"].value, p["sps_f"].value = sps, read_sps.get_interpolation(sps)
    t0, R2 = 0.5, 0.6 * p["rCloud"].value
    fb = get_current_sps_feedback(t0, p)
    P_ram = get_bubbleParams.pRam(R2, fb.Lmech_total, fb.v_mech_total)
    p["t_now"].value, p["stop_t"].value, p["current_phase"].value = t0, stop_t, "transition"
    p["R2"].value, p["v2"].value, p["rShell"].value = R2, 10.0 * cvt.v_kms2au, 1.2 * R2
    p["shell_mass"].value = S._scalar(S.mass_profile.get_mass_profile(R2, p))
    # thermal pressure f P_ram over the whole bubble volume (R1 enlarges it a little)
    p["Eb"].value = f_thermal * P_ram * (4 * math.pi / 3) * R2**3 / (p["gamma_adia"].value - 1)
    p["bubble_LTotal"].value, p["bubble_Tavg"].value = 0.3 * fb.Lmech_total, 3e6
    p["cool_alpha"].value = t0 / R2 * p["v2"].value
    p["phii_scheme"].value = "swept"
    return p


def _rows(p, tmp_path):
    p.flush()
    return [json.loads(ln) for f in tmp_path.rglob("dictionary.jsonl") for ln in f.read_text().splitlines()
            if ln.strip()]


def test_swept_transition_runner_invariants(tmp_path, monkeypatch):
    p = _entry(tmp_path, f_thermal=3.0, stop_t=0.52)
    g = lambda k: p[k].value  # noqa: E731
    starts, real = [], S.scipy.integrate.solve_ivp

    def spy(*a, **kw):                       # the runner's segment solves: their RHS at the segment start
        if not a and len(kw.get("y0", ())) == 3 and "events" in kw:
            starts.append((kw["y0"].copy(), kw["fun"](kw["t_span"][0], kw["y0"])))
        return real(*a, **kw)

    monkeypatch.setattr(S.scipy.integrate, "solve_ivp", spy)
    TP.run_phase_transition(p)
    monkeypatch.undo()
    tr = [r for r in _rows(p, tmp_path) if r["current_phase"] == "transition"]
    assert len(tr) >= 4 and len(starts) >= len(tr) - 1
    # the rows describe the RHS that is integrated: M du/dt + Mdot u + F_grav is the row's drive, and on the first
    # segment (zeta = 1) dEb/dt is the c3c transition's energy equation with v2 -> U
    for r, (y0, d) in zip(tr, starts):
        assert (r["rs_dyn"], r["v2"], r["Eb"]) == tuple(y0)
        drive = r["F_ram"] + r["K_s"] + r["F_rad"] - r["F_ion_in"]
        assert r["shell_mass"] * d[1] + r["shell_massDot"] * y0[1] + r["F_grav"] == pytest.approx(drive, rel=1e-9)
    r, d = tr[0], starts[0][1]
    Ed = (r["Lmech_total"] - r["bubble_LTotal"] - 4 * math.pi * r["R2"] ** 2 * r["P_drive"] * r["rs_dot"]
          - get_bubbleParams.get_leak_luminosity(g("coverFraction"), r["R2"], r["P_drive"], r["c_sound"],
                                                 g("gamma_adia")))
    assert d[2] == pytest.approx(min(Ed, -r["Eb"] * r["c_sound"] / r["R2"]), rel=1e-12)
    for r in tr:
        rs = r["rs_dyn"]
        outer = 4 * math.pi * r["rShell"] ** 2 * r["P_outer"]
        assert abs((4 * math.pi * r["R2"] ** 2 * r["P_drive"] + r["K_s"] + r["F_rad_sh"] - outer) / outer) < 1e-6
        assert abs(rs / r["rShell"] - 1) < 1e-9
        assert r["P_drive"] == pytest.approx(max(r["Pb"], r["P_ram"]), rel=1e-12)
        assert r["F_ram"] == pytest.approx(4 * math.pi * r["R2"] ** 2 * r["P_drive"], rel=1e-12)
        assert r["Eb"] > 0
        rho = g("mu_convert") * S._scalar(S.density_profile.get_density_profile(np.array([rs]), p))
        assert r["shell_massDot"] == pytest.approx(4 * math.pi * rs ** 2 * rho * r["rs_dot"], rel=1e-9)
        assert r["F_grav"] == pytest.approx(g("G") * r["shell_mass"] / rs ** 2
                                            * (g("mCluster") + 0.5 * r["shell_mass"]), rel=1e-12)
    assert len({r["t_now"] for r in tr}) == len(tr)
    for a, b in zip(tr[:-1], tr[1:]):
        assert b["shell_mass"] >= a["shell_mass"]


def test_swept_momentum_continues_the_transition_state(tmp_path):
    p = _entry(tmp_path, f_thermal=0.05, stop_t=0.53)
    res = TP.run_phase_transition(p)
    assert res.termination_reason in ("ram_dominated", "energy_floor")      # both hand the shell to momentum
    assert not p["EndSimulationDirectly"].value
    t_x, rs_x, u_x, M_x = p["t_now"].value, p["rs_dyn"].value, p["v2"].value, p["shell_mass"].value
    p["current_phase"].value = "momentum"
    MP.run_phase_momentum(p)
    rows = _rows(p, tmp_path)
    assert len({r["t_now"] for r in rows}) == len(rows)                       # no second row at the hand-off
    tr = [r for r in rows if r["current_phase"] == "transition"]
    mom = [r for r in rows if r["current_phase"] == "momentum"]
    assert tr and mom and tr[-1]["t_now"] < t_x
    first = mom[0]
    assert first["t_now"] == t_x
    assert (first["rs_dyn"], first["v2"], first["shell_mass"]) == (rs_x, u_x, M_x)
