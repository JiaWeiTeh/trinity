"""Regression: an energy-driven collapse must never emit a garbage negative Pb.

When the energy-driven bubble loses Eb through 0, the phase-boundary reconciliation
snapshot at the end of the implicit phase used to recompute Pb = (gamma-1)*Eb/V from
the now-NEGATIVE collapse Eb and save it as the terminal dictionary.jsonl row
(Pb ~ -1.6e18). The Pb-fix (PB_COLLAPSE_GUARD_FIX.md) skips the reconciliation recompute
on a bad-Eb exit so no negative-Pb row is ever written.

Since PR #715 (`bugfix/high-mass-cluster-transition-without-ebpeak`), a *finite* Eb<=0
collapse no longer dead-stops as ENERGY_COLLAPSED — phase 1b now ROUTES it to the
momentum phase (`classify_energy_collapse`, ENERGY_HANDOFF_FLOOR=1e3). This heavy cloud
(5e9, n=1e2) is the canonical case that used to dead-stop and now hands off. This test
locks in the COMBINED post-merge invariant: the heavy cloud (a) reaches the momentum
phase (the handoff works, no ENERGY_COLLAPSED dead-stop) and (b) still writes zero
negative-Pb rows and a finite-positive terminal Pb (the reconciliation stays clean).

Slow (~1-2 min) end-to-end run, mirroring test_run_smoke's subprocess pattern.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

ENERGY_COLLAPSED_CODE = 51


@pytest.mark.stress
def test_energy_collapse_emits_no_negative_Pb(tmp_path):
    """The heavy cloud that used to dead-stop now hands off to momentum with clean Pb."""
    param = tmp_path / "collapse.param"
    param.write_text(
        "model_name      collapse\n"
        "mCloud          5e9\n"      # heavy GMC: energy-driven phase cannot self-sustain
        "sfe             0.1\n"
        "nCore           1e2\n"
        "PISM            1e4\n"
        "nISM            0.1\n"
        "dens_profile    densPL\n"
        "densPL_alpha    0\n"
        "ZCloud          1\n"
        "rCloud_max      1e9\n"
        "stop_t          0.05\n"     # bound runtime; the collapse/handoff fires ~3e-3 Myr
        "log_console     False\n"
    )

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), str(param)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"run.py exited {result.returncode}\n"
        f"---stdout (tail)---\n{result.stdout[-3000:]}\n"
        f"---stderr (tail)---\n{result.stderr[-3000:]}"
    )

    run_dir = tmp_path / "outputs" / "collapse"
    rows = [
        json.loads(line)
        for line in (run_dir / "dictionary.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows, "dictionary.jsonl is empty — the run never wrote a snapshot"

    # PR #715: the finite Eb<=0 collapse now ROUTES to momentum instead of dead-stopping
    # as ENERGY_COLLAPSED. Assert the handoff actually happened (the run reached momentum),
    # and was NOT recorded as the old code-51 dead-stop.
    phases = {str(r.get("current_phase")) for r in rows}
    assert "momentum" in phases, (
        f"heavy cloud did not reach the momentum phase (handoff regressed); phases seen: {phases}"
    )
    assert rows[-1].get("SimulationEndCode") != ENERGY_COLLAPSED_CODE, (
        "heavy cloud dead-stopped on ENERGY_COLLAPSED — the PR #715 handoff regressed"
    )

    # The Pb-fix invariant (independent of fate): no row may carry a negative bubble
    # pressure. Pre-fix the reconciliation snapshot wrote a terminal Pb ~ -1.6e18.
    negative = [(i, r.get("Pb")) for i, r in enumerate(rows)
                if isinstance(r.get("Pb"), (int, float)) and r["Pb"] < 0]
    assert not negative, (
        f"{len(negative)} row(s) carry a negative Pb (the collapse reconciliation bug); "
        f"first: row {negative[0][0]} Pb={negative[0][1]:.3e}"
    )

    # And the last recorded row is a finite, positive pressure.
    last_Pb = rows[-1].get("Pb")
    assert isinstance(last_Pb, (int, float)) and last_Pb > 0, (
        f"terminal Pb is not finite-positive: {last_Pb!r}"
    )
