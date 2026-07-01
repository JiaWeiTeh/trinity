"""The 📏 standard-protocol theta5 tooling stays honest.

Covers docs/dev/transition/pdv-trigger/runs/{harvest_theta_max,make_theta5_calibration}.py:
harvest_theta_max is THE sanctioned theta measurement (theta_max from dictionary.jsonl accepted
implicit rows — PLAN 📏 rule 2 / retraction R6), so a regression here silently poisons every
future calibration table.
"""

import importlib.util
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "docs/dev/transition/pdv-trigger/runs"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, RUNS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_run(tmp_path, thetas, fired=True):
    """Synthetic run dir: implicit rows with theta(t) = thetas, then a momentum row."""
    run = tmp_path / "cfg__mult2"
    run.mkdir()
    lines = []
    for i, th in enumerate(thetas):
        lines.append(
            json.dumps(
                {
                    "current_phase": "implicit",
                    "t_now": 0.1 * (i + 1),
                    "bubble_Lloss": th * 100.0,
                    "Lmech_total": 100.0,
                }
            )
        )
    lines.append(json.dumps({"current_phase": "momentum", "t_now": 5.0}))
    (run / "dictionary.jsonl").write_text("\n".join(lines) + "\n")
    term = {"outcome": "stopping_time", "detail": "reason: cooling_balance" if fired else "t"}
    (run / "metadata.json").write_text(json.dumps({"termination": term}))
    return run


def test_harvest_theta_max_reads_accepted_rows(tmp_path):
    harvest = _load("harvest_theta_max")
    run = _write_run(tmp_path, thetas=[0.2, 0.8, 0.96, 0.5])
    row = harvest.harvest(run)
    assert abs(row["theta_max"] - 0.96) < 1e-12  # the peak, NOT the last/blowout value
    assert abs(row["t_at_theta_max"] - 0.3) < 1e-12
    assert abs(row["theta_first"] - 0.2) < 1e-12
    assert row["n_impl"] == 4
    assert row["phase_final"] == "momentum"
    assert row["fired_cooling_balance"] is True


def test_harvest_ignores_non_implicit_and_bad_rows(tmp_path):
    harvest = _load("harvest_theta_max")
    run = _write_run(tmp_path, thetas=[0.3], fired=False)
    # momentum-phase and Lmech=0 rows must not contribute a theta
    extra = [
        json.dumps({"current_phase": "momentum", "t_now": 9.0, "bubble_Lloss": 1e9, "Lmech_total": 1.0}),
        json.dumps({"current_phase": "implicit", "t_now": 9.1, "bubble_Lloss": 1.0, "Lmech_total": 0.0}),
    ]
    with (run / "dictionary.jsonl").open("a") as fh:
        fh.write("\n".join(extra) + "\n")
    row = harvest.harvest(run)
    assert abs(row["theta_max"] - 0.3) < 1e-12
    assert row["fired_cooling_balance"] is False


def test_theta5_calibration_selftest():
    _load("make_theta5_calibration").selftest()
