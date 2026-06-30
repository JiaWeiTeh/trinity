"""Regression: an energy-collapsed run must not emit a garbage negative Pb.

When the energy-driven bubble collapses (Eb falls through 0 -> ENERGY_COLLAPSED),
the phase-boundary reconciliation snapshot at the end of the implicit phase used to
recompute Pb = (gamma-1)*Eb/V from the now-NEGATIVE collapse Eb and save it as the
terminal dictionary.jsonl row (Pb ~ -1.6e18). The stop fate (code 51) was already
correct; only the trailing row was garbage. The fix skips the reconciliation snapshot
on the energy_collapsed exit, so the last recorded row keeps the last healthy Pb>0.

This is a slow (~1-2 min) end-to-end run of a heavy cloud that collapses very early
(t ~ 3e-3 Myr), mirroring test_run_smoke's subprocess pattern. It asserts the collapse
is recorded (code 51) AND that no row carries a negative Pb. Before the fix the final
row fails the Pb>0 assertion; after, it passes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ENERGY_COLLAPSED_CODE = 51


def test_energy_collapse_emits_no_negative_Pb(tmp_path):
    """A heavy cloud that ENERGY_COLLAPSEs writes code 51 and zero negative-Pb rows."""
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
        "stop_t          0.05\n"     # bound runtime; collapse fires first (~3e-3 Myr)
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

    # The collapse must be recorded (the stop fate still propagates).
    assert rows[-1].get("SimulationEndCode") == ENERGY_COLLAPSED_CODE, (
        f"expected ENERGY_COLLAPSED (code {ENERGY_COLLAPSED_CODE}); "
        f"got {rows[-1].get('SimulationEndCode')} — config no longer collapses, test is moot"
    )

    # The fix: no row may carry a negative bubble pressure. (Pre-fix the reconciliation
    # snapshot wrote a terminal Pb ~ -1.6e18 from the negative collapse Eb.)
    negative = [(i, r.get("Pb")) for i, r in enumerate(rows)
                if isinstance(r.get("Pb"), (int, float)) and r["Pb"] < 0]
    assert not negative, (
        f"{len(negative)} row(s) carry a negative Pb (the collapse reconciliation bug); "
        f"first: row {negative[0][0]} Pb={negative[0][1]:.3e}"
    )

    # And the last recorded row is a finite, positive pressure (the last healthy snapshot).
    last_Pb = rows[-1].get("Pb")
    assert isinstance(last_Pb, (int, float)) and last_Pb > 0, (
        f"terminal Pb is not finite-positive: {last_Pb!r}"
    )
