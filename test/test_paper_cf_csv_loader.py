"""Unit test for ``paper_Cf.load_cf_runs_from_csv`` — the CSV-input path that
lets the covering-fraction figure run off the reduced SSOT
(``trajectory_points.csv`` + ``summary.csv``) instead of per-run
``dictionary.jsonl`` (no Helix round-trip).

Pure-parse test: hand-built tiny CSVs in ``tmp_path``, no TRINITY run.  Locks in
the three things that path must get right — km/s unit conversion, cf-token
grouping, and duplicate-timestamp hygiene — plus the ``select`` / PHII filters.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: paper_Cf imports pyplot at module load

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from paper.rosette.figures.paper_Cf import (  # noqa: E402
    load_cf_runs_from_csv,
    PC_MYR_TO_KM_S,
)

# One noPHII IC cell swept over cf (RUN_A carries a duplicate seed timestamp),
# plus a yesPHII run in its own cell for the PHII-filter test.
_CELL = "1e5_sfe001_n1e3_BE14_noPHII_PISM100000p0_coolingBoostFmix4p0_nISM1p0"
RUN_A = "1e5_sfe001_n1e3_BE14_noPHII_PISM100000p0_coolingBoostFmix4p0_coverFraction0p89_nISM1p0"
RUN_B = "1e5_sfe001_n1e3_BE14_noPHII_PISM100000p0_coolingBoostFmix4p0_coverFraction0p95_nISM1p0"
RUN_C = "1e5_sfe001_n1e3_BE14_yesPHII_PISM100000p0_coolingBoostFmix4p0_coverFraction0p95_nISM1p0"


def _write(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "run_name,coverFraction,phii\n"
        f"{RUN_A},0.89,False\n"
        f"{RUN_B},0.95,False\n"
        f"{RUN_C},0.95,True\n"
    )
    traj = tmp_path / "trajectory_points.csv"
    traj.write_text(
        "run_name,t,R2,v2,rShell\n"
        # RUN_A: a repeated t=0.0 seed row (must be de-duplicated), then a ramp.
        f"{RUN_A},0.0,0.1,5.0,0.2\n"
        f"{RUN_A},0.0,0.1,5.0,0.2\n"
        f"{RUN_A},1.0,3.0,2.0,10.0\n"
        f"{RUN_A},2.0,4.0,1.0,20.0\n"
        # RUN_B, same cell, unsorted rows on input.
        f"{RUN_B},2.0,16.0,7.0,18.0\n"
        f"{RUN_B},1.0,8.0,4.0,30.0\n"
        # RUN_C, yesPHII (different cell).
        f"{RUN_C},1.0,8.0,4.0,30.0\n"
        f"{RUN_C},2.0,16.0,7.0,18.0\n"
    )
    return traj, summary


def test_grouping_units_and_dedup(tmp_path):
    traj, summary = _write(tmp_path)
    groups = load_cf_runs_from_csv(traj, summary, select=_CELL)

    # RUN_A and RUN_B strip to the same cf-token-free base condition -> one panel.
    assert list(groups) == [_CELL]
    runs = groups[_CELL]
    assert [r["cf"] for r in runs] == [0.89, 0.95]  # sorted by cf, from summary

    a = runs[0]
    # Duplicate t=0.0 collapsed -> strictly increasing, 3 snapshots not 4.
    assert a["n_snaps"] == 3
    assert np.all(np.diff(a["t"]) > 0)
    # v2 stored in pc/Myr -> km/s in `v`.
    assert np.allclose(a["v"], np.array([5.0, 2.0, 1.0]) * PC_MYR_TO_KM_S)
    # rShell carried through; vShell finite (numeric d rShell/dt).
    assert np.allclose(a["rShell"], [0.2, 10.0, 20.0])
    assert np.all(np.isfinite(a["vShell"]))


def test_select_and_phii_filters(tmp_path):
    traj, summary = _write(tmp_path)

    # select overrides PHII: both noPHII cf runs of the cell are kept.
    assert len(load_cf_runs_from_csv(traj, summary, select=_CELL)[_CELL]) == 2

    # Without select, PHII filter applies against the summary boolean column.
    g_no = load_cf_runs_from_csv(traj, summary, phii_mode="no")
    assert sorted(r["cf"] for runs in g_no.values() for r in runs) == [0.89, 0.95]
    g_yes = load_cf_runs_from_csv(traj, summary, phii_mode="yes")
    assert [r["cf"] for runs in g_yes.values() for r in runs] == [0.95]  # only yesPHII
