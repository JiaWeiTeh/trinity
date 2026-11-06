"""
End-to-end smoke test for the ``run.py`` CLI entry point.

This is an integration test that exercises the full read-param → integrate →
write-outputs pipeline in about a minute on a typical development machine.
Heavier than the unit tests but catches install-time regressions that
unit tests cannot: missing bundled defaults, broken imports, scipy /
numpy API drift, path-resolution bugs, etc.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The stored final-row values live in test/goldens/run_smoke_final.json;
# `pytest --regen-goldens` rewrites them. The history below says why each set was taken.
_FINAL_KEYS = (
    # Re-baselined 2026-08-14 for C3c (`c43a50e`): the capped-Strömgren `P_HII` was
    # `Pb` un-ramped, and `P_drive = max(press_bubble, P_HII)` selected it over the
    # dt_switchon-ramped pressure, so the ramp never reached the momentum equation.
    # C3c returns exactly 0.0 on the confined branch (every energy/implicit row), so
    # the early drive is the ramped pressure alone and this exit state moves down:
    # R2 -1.09%, v2 -0.57%, Eb -0.75%.  Before/after table with the reproduce
    # commands: docs/dev/phii-identity/data/g34_golden_rebaseline.csv (the G3.4
    # condition D4 attaches to re-baselining).  Previous values, pre-C3c, were
    # R2 0.25955975558205835, v2 49.226111525136076, Eb 662533.9662741426.
    #
    # Earlier re-baseline 2026-08-05 for the phase-1a segment schedule
    # (docs/dev/phase1a-init/, phase1a_segFrac=0.1 + the vd=-1e8 override
    # deleted).  stop_t=1e-4 Myr is below TFINAL_ENERGY_PHASE=3e-3, and phase
    # 1a integrates in whole segments, so this triple IS the phase-1a exit
    # state at t=3e-3 Myr -- exactly what that change moves, by design.  The
    # previous values (R2 0.2857315185200479, v2 44.73918438203256,
    # Eb 778236.3470566473, captured 2026-07-10 on Python 3.9.6, numpy 1.26.4,
    # scipy 1.13.1) are the pre-fix exit state; the equivalence evidence that
    # the trajectories reconverge afterwards is data/gate_results.csv there.
    #
    # Re-baselined 2026-09-06. ⚠ This file is NOT on D4's standing re-baselining list
    # (docs/dev/phii-identity/PLAN.md:638), so this carries the maintainer's explicit
    # per-change sign-off, given 2026-09-06 for the three changes below together.
    # Before/after table: docs/dev/phii-identity/shell-structure-test/golden_three_way.csv
    #
    #   W32  shell-solver termination flags read at the termination index rather than with
    #        any() over the slice. The neutral-region integration had been SKIPPED on rows
    #        that have a neutral rind; restoring it saturates the non-ionising absorption
    #        and raises tau_kappa_IR, so F_rad rises and the shell is pushed harder. This
    #        is a RADIATION-PRESSURE change: P_HII is exactly 0.0 on every row of this run.
    #        Exit state moves R2 +1.02%, v2 +1.13%, Eb +0.32%.
    #   W52  adaptive shell march (+0.002% here; large elsewhere).
    #   W63  TShell_ion 1e4 -> 8000 K, caseB_alpha auto-scaled (+0.0022% here -- this config
    #        never leaves C3c's confined branch, so the temperature does not reach P_HII).
    #
    # For scale, the C3c re-baseline of 2026-08-14 moved these same three by R2 -1.09%,
    # v2 -0.57%, Eb -0.75%. R2 is comparable and opposite; v2 and Eb do NOT mirror it, so
    # this is not a cancellation of C3c.
    # Previous values, pre-W32: R2 0.2567222331253797, v2 48.944358738549326,
    # Eb 657558.3776158141.
    # Re-baselined 2026-11-01 for the PCHIP stellar table (sps_interp pchip, the new
    # default): before the first table row (0-0.01 Myr) PCHIP is flat where the cubic
    # spline was not. R2 -1.28e-6, v2 -3.45e-6, Eb -1.73e-5 (relative); with
    # sps_interp cubic the previous values are reproduced (R2 0.25934419485301125,
    # v2 49.49653363172906, Eb 659632.2441434182).
    # docs/dev/phii-identity/weak-wind/early-1b/SPS_TRANSIENTS.md section 7.
    # Re-baselined 2026-11-02 for the two-zone neutral layer as the default (TShell_neu 'auto', TShell_mol 20;
    # docs/dev/phii-identity/weak-wind/neutral-shell/PDR_DEFAULT_PLAN.md P5, ruling D-F5): the neutral rows now carry a PDR
    # atomic zone (here at the K99 table's 5000 K top, G0 ~ 1.6e6) and a 20 K molecular zone. R2 -1.8e-6, v2 -3.6e-6,
    # Eb +1.1e-7 (relative); with TShell_neu 100 and TShell_mol 0 the previous values are reproduced exactly
    # (R2 0.25934386495844797, v2 49.49636304648124, Eb 659620.8513924862).
    # Re-baselined 2026-11-06 for Kaufman et al. 1999's numeric surface-temperature grid in place of the digitised Fig. 1
    # (docs/dev/phii-identity/weak-wind/neutral-shell/pdr-table/PLAN.md): against what d8ababfa computes, R2 -7.35e-07,
    # v2 -1.63e-06, Eb +8.2e-08 (relative; at c126ba10 the same change gave -7.35e-07, -1.63e-06, +1.03e-07). The values
    # stored before were c126ba10's (R2 0.2593433916087158, v2 49.496185853010495, Eb 659620.924962753), which d8ababfa
    # itself misses by R2 -8.56e-06, v2 -7.84e-06, Eb -2.68e-05: #767's CIE-bracket fix, the one default-path code change
    # between the two, was never re-pinned here. This re-pin takes that up too.
    "R2",
    "v2",
    "Eb",
)


def test_quickstart_completes_cleanly(tmp_path, golden):
    """``python run.py <fast param>`` exits 0 and writes the expected outputs.

    Uses ``stop_t = 1e-4`` Myr so the run stays short while writing enough
    snapshots to enter Phase 1 and exercise the bubble-structure code path
    (where past regressions have lived) without running for minutes.
    ``stop_t_diss`` is left at its default. Runs from a fresh CWD
    (``tmp_path``) — outputs land under ``tmp_path/outputs/smoke/`` and
    are cleaned by pytest.
    """
    param = tmp_path / "smoke.param"
    # TCloud 0: the goldens below predate the cloud gas's own pressure (TCloud defaults to 10 K since 2026-10-29)
    param.write_text(
        "mCloud      1e5\n" "sfe         0.3\n" "stop_t      1e-4\n" "model_name  smoke\n" "TCloud      0\n"
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
        f"---stdout (tail)---\n{result.stdout[-4000:]}\n"
        f"---stderr (tail)---\n{result.stderr[-4000:]}"
    )

    run_dir = tmp_path / "outputs" / "smoke"
    # Only assert on artifacts written unconditionally during setup/integration.
    # summary.txt and simulationEnd.txt are emitted on "normal termination"
    # paths a very short stop_t does not reach.
    for fname in ("metadata.json", "dictionary.jsonl"):
        assert (run_dir / fname).exists(), (
            f"expected output {fname} missing from {run_dir}; "
            f"got: {sorted(p.name for p in run_dir.iterdir())}"
        )
    # dictionary.jsonl should be non-trivial — proves the integrator ran,
    # not just that setup wrote a placeholder.
    rows = [
        json.loads(line)
        for line in (run_dir / "dictionary.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) >= 2, (
        f"dictionary.jsonl has only {len(rows)} snapshot lines; " f"integrator likely never ran"
    )

    final = rows[-1]
    for key in _FINAL_KEYS:  # before golden(), so --regen-goldens never stores a bad value
        value = final.get(key)
        assert isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    expected = golden("run_smoke_final", {key: final[key] for key in _FINAL_KEYS})
    for key, stored in expected.items():
        assert final[key] == pytest.approx(stored, rel=1e-6)
