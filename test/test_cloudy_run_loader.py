"""
Tests for ``src._output.cloudy.run_loader``.

Covers:
- End-to-end load against ``outputs/mockOutput/mockFullrun/``: model_name,
  ZCloud (from summary), dens_profile, Status, jsonl-backed TrinityOutput.
- ``_coerce_scalar`` — int, float, bool, None, nan, inf, list, string.
- ``_parse_summary_txt`` — comments, blank lines, mixed types.
- ``_parse_simulation_end`` — flat fields, numeric fields with units stripped.
- Error paths: missing run dir, missing metadata.json, missing summary,
  missing simulationEnd.txt, bad dens_profile, missing model_name.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src._output.cloudy.run_loader import (
    RunBundle,
    RunLoadError,
    VALID_DENS_PROFILES,
    load_run,
)
from src._output.cloudy.run_loader import (
    _coerce_scalar,
    _parse_simulation_end,
    _parse_summary_txt,
)


MOCK_FULLRUN = Path("outputs/mockOutput/mockFullrun")


# --------------------------------------------------------------------------- #
# End-to-end against real mockFullrun
# --------------------------------------------------------------------------- #

def test_load_run_against_mockFullrun():
    bundle = load_run(MOCK_FULLRUN)
    assert isinstance(bundle, RunBundle)
    # metadata.json
    assert bundle.model_name == "4e3_sfe001_n5e2_PL0"
    assert bundle.metadata["dens_profile"] == "densPL"
    assert bundle.metadata["tSF"] == 0
    # summary.txt — has ZCloud, mCloud, etc.
    assert bundle.summary["ZCloud"] == 1.0
    assert bundle.summary["dens_profile"] == "densPL"
    assert bundle.summary["allowShellDissolution"] is True
    # simulationEnd.txt
    assert bundle.end_state["outcome"] == "stopping_time"
    assert bundle.end_state["exit_code"] == 1
    assert bundle.end_state["model_name"] == "4e3_sfe001_n5e2_PL0"
    assert bundle.end_state["t_now_myr"] == pytest.approx(0.300, abs=1e-3)
    # TrinityOutput is open and has the expected snapshot count
    assert len(bundle.output) == 178


def test_run_dir_must_exist():
    with pytest.raises(FileNotFoundError, match="Run directory"):
        load_run(Path("/nonexistent/path"))


# --------------------------------------------------------------------------- #
# _coerce_scalar
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw, expected", [
    ("", ""),
    ("True", True),
    ("False", False),
    ("None", None),
    ("0", 0),
    ("-1", -1),
    ("+42", 42),
    ("1.5", 1.5),
    ("-1.5e-10", -1.5e-10),
    ("INFO", "INFO"),
    ("/path/to/somewhere", "/path/to/somewhere"),
    ("[]", []),
    ("(1, 2)", (1, 2)),
])
def test_coerce_scalar_table(raw, expected):
    assert _coerce_scalar(raw) == expected


def test_coerce_scalar_nan():
    v = _coerce_scalar("nan")
    assert isinstance(v, float)
    assert math.isnan(v)


@pytest.mark.parametrize("raw", ["inf", "+inf", "-inf"])
def test_coerce_scalar_inf(raw):
    v = _coerce_scalar(raw)
    assert isinstance(v, float)
    assert math.isinf(v)


def test_coerce_scalar_does_not_accept_bare_expressions():
    """Arbitrary Python expressions (not list/tuple/dict literals) stay as strings."""
    # `1+1` is not a literal opener; should fall through to plain str.
    assert _coerce_scalar("1+1") == "1+1"


# --------------------------------------------------------------------------- #
# _parse_summary_txt
# --------------------------------------------------------------------------- #

def test_parse_summary_txt_skips_comments_and_blanks():
    text = (
        "# header line\n"
        "# another\n"
        "\n"
        "key1   42\n"
        "key2   3.14\n"
        "\n"
        "key3   True\n"
        "# trailing comment\n"
    )
    parsed = _parse_summary_txt(text)
    assert parsed == {"key1": 42, "key2": 3.14, "key3": True}


def test_parse_summary_txt_handles_nan_and_lists():
    text = "bubble_dMdt   nan\nshell_r_arr   []\n"
    parsed = _parse_summary_txt(text)
    assert math.isnan(parsed["bubble_dMdt"])
    assert parsed["shell_r_arr"] == []


def test_parse_summary_txt_keeps_paths_as_strings():
    text = "path_sps   /Users/jwt/lib/sps/starburst99/\n"
    parsed = _parse_summary_txt(text)
    assert parsed["path_sps"] == "/Users/jwt/lib/sps/starburst99/"


def test_parse_summary_txt_empty_value():
    text = "current_phase     \n"
    parsed = _parse_summary_txt(text)
    assert parsed["current_phase"] == ""


# --------------------------------------------------------------------------- #
# _parse_simulation_end
# --------------------------------------------------------------------------- #

SIM_END_SAMPLE = """\
==================================================
TRINITY Simulation End Report
==================================================
Timestamp: 2026-04-30 00:30:37
Model: 4e3_sfe001_n5e2_PL0

--------------------------------------------------
TERMINATION
--------------------------------------------------
Outcome: stopping_time
Detail: Stopping time reached
Exit Code: 1

--------------------------------------------------
FINAL STATE
--------------------------------------------------
  Time:           0.300 Myr
  Radius (R2):    2.51 pc
  Shell nMax:     1.13e+00 cm^-3
  Shell Velocity: 2.54 km/s

--------------------------------------------------
INITIAL CLOUD PARAMETERS
--------------------------------------------------
  mCloud:  3.97e+03 Msun
  nCore:   5.00e+02 cm^-3
  rCloud:  3.80 pc
  rCore:   0.01 pc
  alpha:   0.0
  nISM:    1.00e-01 cm^-3

==================================================
"""


def test_parse_simulation_end_full_sample():
    end = _parse_simulation_end(SIM_END_SAMPLE)
    assert end["model_name"] == "4e3_sfe001_n5e2_PL0"
    assert end["outcome"] == "stopping_time"
    assert end["detail"] == "Stopping time reached"
    assert end["exit_code"] == 1
    # final state — units stripped, value as float
    assert end["t_now_myr"] == 0.300
    assert end["R2_pc"] == 2.51
    assert end["shell_nMax_cm3"] == pytest.approx(1.13)
    assert end["shell_v_kms"] == pytest.approx(2.54)
    # initial cloud
    assert end["mCloud_msun"] == pytest.approx(3.97e3)
    assert end["nCore_cm3"] == pytest.approx(500.0)
    assert end["rCloud_pc"] == 3.80
    assert end["alpha"] == 0.0
    assert end["nISM_cm3"] == pytest.approx(0.1)


def test_parse_simulation_end_collapse_outcome():
    text = SIM_END_SAMPLE.replace(
        "Outcome: stopping_time", "Outcome: shell_collapsed"
    ).replace("Exit Code: 1", "Exit Code: 4")
    out = _parse_simulation_end(text)
    assert out["outcome"] == "shell_collapsed"
    assert out["exit_code"] == 4


def test_parse_simulation_end_tolerates_missing_sections():
    # Just an outcome line, nothing else
    text = "Outcome: stopping_time\nExit Code: 1\n"
    out = _parse_simulation_end(text)
    assert out["outcome"] == "stopping_time"
    assert out["exit_code"] == 1
    assert "t_now_myr" not in out  # absent → not added


def test_parse_simulation_end_legacy_back_compat():
    """Pre-fix runs wrote Status / End Reason / Raw Reason; we still parse them."""
    text = (
        "Status: SUCCESS\n"
        "End Reason: Maximum simulation time reached\n"
        "Raw Reason: Stopping time reached\n"
        "Exit Code: 1\n"
    )
    out = _parse_simulation_end(text)
    assert out["exit_code"] == 1
    assert out["detail"] == "Stopping time reached"
    assert out["outcome"] == "legacy_success"


# --------------------------------------------------------------------------- #
# Error paths (synthesise broken run dirs in tmp_path)
# --------------------------------------------------------------------------- #

def _make_minimal_run_dir(tmp_path: Path, *, write_metadata=True,
                          write_summary=True, write_end=True,
                          write_jsonl=True, dens_profile="densPL",
                          model_name="m"):
    """Build a minimal run dir; toggle which files are present."""
    rd = tmp_path / "run"
    rd.mkdir()
    if write_metadata:
        (rd / "metadata.json").write_text(json.dumps({
            "model_name": model_name,
            "dens_profile": dens_profile,
            "tSF": 0,
        }))
    if write_summary:
        (rd / f"{model_name}_summary.txt").write_text(
            "ZCloud  1.0\n"
            f"dens_profile  {dens_profile}\n"
        )
    if write_end:
        (rd / "simulationEnd.txt").write_text(
            "Outcome: stopping_time\nDetail: ok\nExit Code: 1\n"
        )
    if write_jsonl:
        (rd / "dictionary.jsonl").write_text(
            json.dumps({
                "t_now": 0.1, "tSF": 0, "current_phase": "transition",
                "R2": 1.0, "rShell": 1.5, "Qi": 1e60,
                "shell_r_arr": [1.0, 1.5], "log_shell_n_arr": [58.0, 57.0],
            }) + "\n"
        )
    return rd


def test_load_run_missing_metadata(tmp_path):
    rd = _make_minimal_run_dir(tmp_path, write_metadata=False)
    with pytest.raises(RunLoadError, match="metadata.json missing"):
        load_run(rd)


def test_load_run_missing_summary(tmp_path):
    rd = _make_minimal_run_dir(tmp_path, write_summary=False)
    with pytest.raises(RunLoadError, match="summary file missing"):
        load_run(rd)


def test_load_run_missing_simulation_end(tmp_path):
    # Phase 2: load_run prefers metadata.json[termination]; falls back to
    # simulationEnd.txt for legacy runs.  With both absent the error
    # message names both sources.
    rd = _make_minimal_run_dir(tmp_path, write_end=False)
    with pytest.raises(
        RunLoadError,
        match="neither metadata.json.termination. nor simulationEnd.txt",
    ):
        load_run(rd)


def test_load_run_missing_jsonl(tmp_path):
    rd = _make_minimal_run_dir(tmp_path, write_jsonl=False)
    with pytest.raises(RunLoadError, match="snapshot stream not found"):
        load_run(rd)


def test_load_run_unknown_dens_profile(tmp_path):
    rd = _make_minimal_run_dir(tmp_path, dens_profile="densXYZ")
    with pytest.raises(RunLoadError, match="unknown dens_profile"):
        load_run(rd)


def test_load_run_metadata_missing_model_name(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "metadata.json").write_text(json.dumps({"dens_profile": "densPL"}))
    with pytest.raises(RunLoadError, match="lacks a 'model_name'"):
        load_run(rd)


def test_load_run_metadata_malformed_json(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "metadata.json").write_text("{not valid json")
    with pytest.raises(RunLoadError, match="metadata.json malformed"):
        load_run(rd)


def test_load_run_returns_frozen_bundle(tmp_path):
    rd = _make_minimal_run_dir(tmp_path)
    bundle = load_run(rd)
    # frozen=True → assignment should raise
    with pytest.raises(Exception):  # FrozenInstanceError
        bundle.model_name = "other"
    # Sanity: VALID_DENS_PROFILES is what we expect
    assert VALID_DENS_PROFILES == frozenset({"densBE", "densPL"})
