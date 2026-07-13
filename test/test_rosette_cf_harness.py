"""Checks for the rosette-cf scan harness (docs/dev/rosette-cf/harness/).

The harness scripts are stdlib-only and not a package; import them by path. Values are
physically plausible for the Rosette scan regime: radii in pc (R2 ~ 4-9, rShell ~ 13-22),
ages in Myr within the 0-3 Myr stop_t window.
"""

import csv
import importlib.util
import math
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "docs" / "dev" / "rosette-cf" / "harness"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


match_mod = _load("match_cf_scan")
run_mod = _load("run_cf_scan_local")


def _traj(t_final, dt=0.01):
    """Synthetic trajectory: R2 = 4 + 2t pc, rShell = 13 + 3t pc (t in Myr).

    Under the policy (R2<->7+/-1, rShell<->19+/-2) chi2(t) = (2t-3)^2 + (1.5t-3)^2,
    minimised at t* = 21/12.5 = 1.68 Myr with chi2* = 0.36.
    """
    n = int(round(t_final / dt))
    return [(i * dt, 4 + 2 * i * dt, 13 + 3 * i * dt) for i in range(n + 1)]


def test_match_run_finds_best_age():
    m = match_mod.match_run(_traj(3.0))
    assert m["t_best_7"] == pytest.approx(1.68, abs=0.02)
    assert m["chi2_min_7"] == pytest.approx(0.36, abs=0.01)
    # matched-age collation columns: R2(2.0) = 8, rShell(2.0) = 19
    assert m["R2_at_2p0"] == pytest.approx(8.0, abs=1e-6)
    assert m["rShell_at_2p0"] == pytest.approx(19.0, abs=1e-6)
    assert m["age_censored"] is False
    # the 6.2 pc base (F-12) is reported alongside and prefers a smaller cavity -> earlier age
    assert m["t_best_62"] < m["t_best_7"]


def test_match_run_age_censored_clips_window():
    m = match_mod.match_run(_traj(2.0))  # truncates inside the 1.5-2.5 Myr prior
    assert m["age_censored"] is True
    assert m["R2_at_2p5"] == ""  # beyond t_final: empty, never extrapolated
    assert m["t_best_7"] <= 2.0
    # t* = 1.68 < 2.0 still inside the clipped window
    assert m["chi2_min_7"] == pytest.approx(0.36, abs=0.01)


def test_match_run_below_prior_is_unmatchable():
    m = match_mod.match_run(_traj(1.0))  # ends before AGE_MIN
    assert m["chi2_min_7"] == "" and m["t_best_7"] == ""


def test_parabola_vertex_interpolates_between_grid_points():
    xs, ys, flag = match_mod.parabola_vertex([(0.70, 5.0), (0.85, 2.0), (1.0, 3.0)])
    assert flag == "ok"
    assert xs == pytest.approx(0.8875)  # between 0.85 and 1.0, like the pilot's ~0.89
    assert ys < 2.0


def test_parabola_vertex_flags_edge_and_nonconvex():
    # monotonically falling toward Cf=1 -> convex fit but vertex beyond the grid
    xs, _, flag = match_mod.parabola_vertex([(0.70, 9.0), (0.85, 4.0), (1.0, 1.0)])
    assert flag == "outside-grid" and xs > 1.0
    # concave (non-convex) -> fall back to the best grid point
    xs, ys, flag = match_mod.parabola_vertex([(0.70, 2.0), (0.85, 5.0), (1.0, 1.0)])
    assert flag == "non-convex" and (xs, ys) == (1.0, 1.0)


def test_interp_at_stays_inside_range():
    rows = _traj(2.0)
    assert match_mod.interp_at(rows, 2.5) is None
    r2, rshell = match_mod.interp_at(rows, 1.005)  # between snapshots
    assert r2 == pytest.approx(4 + 2 * 1.005)
    assert rshell == pytest.approx(13 + 3 * 1.005)


def test_runner_order_is_decision_first():
    def arm(name, ncore, cf, fmix):
        return {
            "name": name,
            "params": {"nCore": ncore, "coverFraction": cf, "cooling_boost_fmix": fmix},
        }

    arms = [
        arm("diffuse_sealed", 5e3, 1.0, 1),
        arm("dense_open", 1e5, 0.70, 1),
        arm("dense_sealed_f4", 1e5, 1.0, 4),
        arm("dense_sealed_f1", 1e5, 1.0, 1),
    ]
    ordered = [a["name"] for a in sorted(arms, key=run_mod.order_key)]
    # dense nCore first; within it Cf=1.0 (the sealed §0.3 baseline) before open; fmix 1 before 4
    assert ordered == ["dense_sealed_f1", "dense_sealed_f4", "dense_open", "diffuse_sealed"]


def test_runner_skips_only_quotable_summary_rows(tmp_path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "# stamp line\n"
        "run_name,exit_code\n"
        "arm_ok,0\n"
        "arm_wallkilled,124\n"
        "arm_crashed,1\n"
    )
    done = run_mod.done_in_summary(str(summary))
    assert done == {"arm_ok"}  # 124/1 stay in the todo list (📏: re-run, never quote)


def test_chi2_policy_constants_match_the_brief():
    # transcribed policy: R2<->7+/-1 pc (alt 6.2), rShell<->19+/-2 pc, prior 1.5-2.5 Myr
    assert (match_mod.R2_TARGET, match_mod.R2_ERR) == (7.0, 1.0)
    assert match_mod.R2_TARGET_ALT == 6.2
    assert (match_mod.RSHELL_TARGET, match_mod.RSHELL_ERR) == (19.0, 2.0)
    assert (match_mod.AGE_MIN, match_mod.AGE_MAX) == (1.5, 2.5)
    assert match_mod.chi2(8.0, 21.0, 7.0) == pytest.approx(1.0 + 1.0)


def test_matcher_end_to_end_writes_cells(tmp_path):
    """Full pipeline on synthetic files: 1 cell x 3 Cf, one arm wall-killed -> flagged cell."""
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    summary = tmp_path / "summary.csv"
    axes = dict(mCloud=1e5, sfe=0.01, nCore=1e4, cooling_boost_fmix=1, include_PHII=True)
    names = {0.70: "arm_cf0p7", 0.85: "arm_cf0p85", 1.0: "arm_cf1p0"}
    with summary.open("w", newline="") as fh:
        fh.write("# stamp\n")
        cols = ["run_name", "coverFraction", "exit_code", *axes]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for cf, name in names.items():
            w.writerow(
                {
                    "run_name": name,
                    "coverFraction": cf,
                    "exit_code": "124" if cf == 0.85 else "0",
                    **axes,
                }
            )
            with (traj_dir / f"{name}.csv").open("w", newline="") as tf:
                tw = csv.writer(tf)
                tw.writerow(["t_now", "R2", "v2", "rShell", "current_phase"])
                # smaller Cf -> weaker bubble -> smaller radii at fixed t (plausible ordering)
                for t, r2, rs in _traj(3.0, dt=0.05):
                    tw.writerow([t, r2 * cf, 15.0, rs * cf, "energy"])
    out, cells_out = tmp_path / "match.csv", tmp_path / "cells.csv"
    match_mod.main(
        [
            "--summary",
            str(summary),
            "--traj-dir",
            str(traj_dir),
            "--out",
            str(out),
            "--cells-out",
            str(cells_out),
        ]
    )
    with out.open() as fh:
        rows = {
            r["run_name"]: r
            for r in csv.DictReader(x for x in fh if not x.lstrip().startswith("#"))
        }
    assert rows["arm_cf0p85"]["matchable"] == "False"  # wall-killed: excluded from minima
    assert rows["arm_cf1p0"]["matchable"] == "True"
    with cells_out.open() as fh:
        cell_rows = list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))
    (cell,) = cell_rows
    assert cell["n_quotable"] == "2"
    assert "incomplete Cf grid" in cell["note"]  # 2/3 points: no parabola, flagged loudly
    assert cell["fit_flag_7"] == "only-2-points"
    # matched-t overshoot columns align with cf_grid and carry both bases (F-12)
    assert len(cell["over7_at_tmatch"].split(";")) == len(cell["cf_grid"].split(";"))
    over7 = [float(v) for v in cell["over7_at_tmatch"].split(";")]
    over62 = [float(v) for v in cell["over62_at_tmatch"].split(";")]
    assert all(math.isclose(a - b, 6.2 - 7.0, abs_tol=1e-9) for a, b in zip(over7, over62))
