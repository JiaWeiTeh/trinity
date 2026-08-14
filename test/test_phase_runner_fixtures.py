"""Fast full-run regression fixtures for the phase runners (roadmap lane B, B2;
`docs/dev/roadmap/solver-audit.md` F4).

F4's finding: ``run_phase_energy`` (1b), ``run_phase_transition`` (1c) and
``run_phase_momentum`` (2) have **zero direct tests** — coverage is whatever
path ``test_run_smoke.py``'s single quickstart config happens to take (phase 1a
only, ``stop_t = 1e-4``), plus the beta-delta suite for the 1b *inner* solve.
The failure scenario F4 names: an edit to the 1c sound-crossing fallback or the
momentum force loop passes the entire suite, and the break surfaces days later
as a wrong fate in an hours-long run.

``test_all_four_phase_runners_execute`` closes that: one ~2-minute run that
walks **energy -> implicit -> transition -> momentum** and pins the phase-entry
snapshots, the termination outcome, the snapshot count and two finals. It is
deliberately in the default suite, not marked ``stress``, because F4 asks for a
gate the rest of lane B executes against and a deselected-by-default gate does
not gate.

WHY THIS CONFIG IS NOT A PRODUCTION CONFIGURATION
-------------------------------------------------
**Read this before quoting any number from this fixture as physics.** Phases 1c
and 2 are *unreachable* from TRINITY's own defaults: the implicit->momentum
hand-off fires on ``(Lgain - Lloss)/Lgain < phaseSwitch_LlossLgain`` (0.05), and
the modelled bubble retains so much energy that the ratio never gets there — the
run sits in ``implicit`` to the ``stop_t`` cap. That is a diagnosed
physics-completeness gap, not a tunable threshold
(``docs/dev/transition/cleanroom/FINDINGS.md``; the root fix is the
mixing-layer work, roadmap lane A item A4). Measured corroboration: of the 72
committed full runs in ``docs/dev/rosette-cf/data/cf_scan_PISM1e5_traj``, no arm
with ``coverFraction = 1.0`` reached transition or momentum in 3 Myr, while 45
of the deviating arms reached momentum.

So the fixture must deviate deliberately or the two runners get no coverage at
any runtime. The parameters below are one arm of that committed scan
(``docs/dev/rosette-cf/rosette_cf_survey_PISM1e5_fmix.param``) — an external ISM
pressure, a patchy shell, no P_HII and a 4x cooling multiplier — chosen only
because it reaches momentum fastest (t = 0.0087 Myr). **This fixture asserts
that the code paths still run and still produce the same numbers. It asserts
nothing about whether those numbers are physically right.**

Verified reproducible before the goldens were written: two runs in separate
processes produced a byte-identical ``dictionary.jsonl``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# One arm of docs/dev/rosette-cf/rosette_cf_survey_PISM1e5_fmix.param. Every
# line that deviates from TRINITY's defaults is load-bearing for reaching phases
# 1c/2 — see the module docstring. stop_t = 0.02 Myr sits just past the momentum
# hand-off at t = 0.0087 so the fixture stays ~2 minutes.
_FIXTURE_PARAM = """\
mCloud                1e4
sfe                   0.1
nCore                 5e2
PISM                  1e5
coverFraction         0.70
include_PHII          False
cooling_boost_mode    multiplier
cooling_boost_fmix    4
dens_profile          densPL
densPL_alpha          0
nISM                  1
ZCloud                1
allowShellDissolution True
stop_t_diss           1
stop_r                None
coll_r                1
stop_at_rCloud_nSnap  None
stop_t                0.02
model_name            phasefix
log_console           False
"""

# Snapshot index at which each phase is first seen. 'implicit' at 87 is the
# TFINAL_ENERGY_PHASE = 3e-3 Myr schedule boundary, not a physics trigger, so it
# is stable across configs; 'transition' and 'momentum' are the physics hand-offs
# this fixture exists to exercise.
_PHASE_ENTRY_SNAPSHOT = {
    "energy": 0,
    "implicit": 87,
    "transition": 89,
    "momentum": 91,
}

_SNAPSHOT_COUNT = 97

# Captured on this branch, 2026-08-06. Eb is deliberately absent: it is exactly
# 0.0 in the momentum phase (no thermal bubble), so it pins nothing.
_FINAL_GOLDENS = {
    "R2": 0.7092571238286148,
    "v2": 11.222515227888755,
}


def _run_fixture(tmp_path: Path, param_text: str, model_name: str):
    """Run ``run.py`` on a fixture param in its own CWD; return the snapshots."""
    param = tmp_path / "fixture.param"
    param.write_text(param_text)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), str(param)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, (
        f"run.py exited {result.returncode}\n"
        f"---stdout (tail)---\n{result.stdout[-4000:]}\n"
        f"---stderr (tail)---\n{result.stderr[-4000:]}"
    )
    run_dir = tmp_path / "outputs" / model_name
    rows = [
        json.loads(line)
        for line in (run_dir / "dictionary.jsonl").read_text().splitlines()
        if line.strip()
    ]
    termination = json.loads((run_dir / "metadata.json").read_text()).get("termination") or {}
    return rows, termination


def _phase_entries(rows):
    entries = {}
    for index, row in enumerate(rows):
        entries.setdefault(row["current_phase"], index)
    return entries


def test_all_four_phase_runners_execute(tmp_path):
    """The lane-B gate: all four phase runners execute and still agree.

    Fails loudly if an edit changes *which* phases a run reaches, *when* it
    reaches them, how many snapshots it writes, how it terminates, or where the
    shell ends up. Any of those moving is either an intended scheme change that
    should re-baseline this fixture on purpose, or the regression F4 warns about.
    """
    rows, termination = _run_fixture(tmp_path, _FIXTURE_PARAM, "phasefix")

    entries = _phase_entries(rows)
    assert entries == _PHASE_ENTRY_SNAPSHOT, (
        "phase reachability or ordering changed.\n"
        f"  expected {_PHASE_ENTRY_SNAPSHOT}\n"
        f"  got      {entries}\n"
        "If phases 1c/2 have gone missing, the implicit->momentum hand-off no "
        "longer fires for this arm — see the module docstring and "
        "docs/dev/transition/cleanroom/FINDINGS.md."
    )
    assert rows[-1]["current_phase"] == "momentum"
    assert len(rows) == _SNAPSHOT_COUNT
    assert termination.get("exit_code") == 1
    assert termination.get("outcome") == "stopping_time"

    final = rows[-1]
    for key, expected in _FINAL_GOLDENS.items():
        value = final.get(key)
        assert isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        assert value == pytest.approx(expected, rel=1e-6), f"final {key} moved"


@pytest.mark.stress
def test_phase1b_runner_at_production_defaults(tmp_path):
    """The fidelity complement: reach the 1b runner with **no** deviations.

    ``test_all_four_phase_runners_execute`` buys its 1c/2 coverage with a
    deviating config. This one keeps every default TRINITY actually ships and
    only asks that the run cross into ``implicit``, which it does at the
    TFINAL_ENERGY_PHASE = 3e-3 Myr schedule boundary. ``stress``-marked because
    it adds no runner coverage the default-suite fixture lacks — only fidelity —
    and the suite should not pay ~2 minutes twice.
    """
    rows, termination = _run_fixture(
        tmp_path,
        "mCloud 1e4\nsfe 0.1\nnCore 5e2\nstop_t 0.0036\n"
        "model_name defaultfix\nlog_console False\n",
        "defaultfix",
    )
    entries = _phase_entries(rows)
    assert entries.get("energy") == 0
    assert (
        "implicit" in entries
    ), f"production defaults no longer reach phase 1b; phases seen: {sorted(entries)}"
    assert rows[entries["implicit"]]["t_now"] == pytest.approx(3.5e-3, rel=1e-3)
    assert termination.get("outcome") == "stopping_time"
