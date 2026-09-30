"""End-to-end runs for the 2026-09-30 phase-1a changes (subprocess, like
test_energy_collapse_snapshot: trinity leaks module state in-process).

- A young seed whose phase-1a bubble collapses is handed to momentum via 1c,
  phase 1b skipped (docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md).
- An old seed gets the stretched switch-on window, phase 1a capped at stop_t,
  and the in-loop cooling refresh (docs/dev/switchon-successor/PLAN.md §D5).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE = (
    "PISM 0\nnISM 1\ninclude_PHII True\ncooling_boost_mode multiplier\n"
    "cooling_boost_fmix 1\ndens_profile densPL\ndensPL_alpha 0\nZCloud 1\n"
    "coverFraction 1.0\nrCloud_max 1e9\nallowShellDissolution True\nstop_t_diss 1\n"
    "stop_r 500\ncoll_r 1\nlog_console False\n"
)


def _run(tmp_path, name, extra):
    param = tmp_path / f"{name}.param"
    param.write_text(f"model_name {name}\n" + BASE + extra)
    res = subprocess.run([sys.executable, str(REPO_ROOT / "run.py"), str(param)],
                         cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stderr[-3000:]
    out = tmp_path / "outputs" / name
    rows = [json.loads(line) for line in (out / "dictionary.jsonl").read_text().splitlines() if line.strip()]
    meta = json.loads((out / "metadata.json").read_text())
    log = (out / "trinity.log").read_text(errors="ignore")
    return rows, meta, log


def test_young_seed_phase1a_collapse_is_handed_to_momentum(tmp_path):
    # dt_phase0 = 0.309 kyr (window unchanged); 1a energy_collapse at ~1.4 kyr on 2bcfc345
    rows, meta, log = _run(tmp_path, "handoff",
                           "mCloud 5e6\nsfe 0.5\nnCore 1e4\nFB_thermCoeffWind 0.01\nstop_t 0.004\n")
    assert "routing to momentum via 1c, phase 1b skipped" in log
    assert "Implicit phase completed: energy_to_momentum" in log
    assert meta["termination"]["outcome"] != "energy_collapsed"
    assert meta["final_state"]["energy_handoff_1a"] is True
    assert meta["final_state"]["dt_switchon"] == 1e-3
    phases = [r["current_phase"] for r in rows]
    assert "implicit" not in phases and "momentum" in phases
    last_energy = max(i for i, p in enumerate(phases) if p == "energy")
    handoff, nxt = rows[last_energy], rows[last_energy + 1]
    assert handoff["Eb"] == 1e3                      # ENERGY_HANDOFF_FLOOR
    assert handoff["isCollapse"] is False
    assert nxt["t_now"] > handoff["t_now"] and nxt["R2"] > handoff["R2"] > 0
    assert all(r["Pb"] > 0 for r in rows if isinstance(r.get("Pb"), (int, float)))


def test_old_seed_window_stop_t_cap_and_cooling_refresh(tmp_path):
    # dt_phase0 = 0.92 kyr: window 2.76 kyr, phase 1a would run to 8.28 kyr; stop_t 7 kyr caps it
    rows, meta, log = _run(tmp_path, "oldseed",
                           "mCloud 1e7\nsfe 0.7\nnCore 1e2\nFB_thermCoeffWind 0.1\nstop_t 0.007\n")
    t0 = rows[0]["t_now"]
    assert meta["final_state"]["dt_switchon"] == pytest.approx(3 * t0, rel=1e-12)
    assert "phase 1a to t=7.0000e-03 Myr" in log
    energy = [r for r in rows if r["current_phase"] == "energy"]
    assert max(r["t_now"] for r in energy) <= 0.007 * (1 + 1e-12)
    refreshed = {r["t_previousCoolingUpdate"] for r in energy}
    assert len(refreshed) >= 2, "phase 1a longer than 5 kyr did not refresh the cooling cube"
