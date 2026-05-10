"""
Tests for ``src._output.cloudy.snapshot_to_deck.snapshot_to_values``.

Covers:
- End-to-end against a real ``mockFullrun`` snapshot at age ~0.15 Myr:
  every substitution key present, well-formed, and within physically
  expected ranges.
- AGE_YR formula (regression guard against the v4-plan bug).
- LOG_QH sign (regression guard against the v4-plan sign-flip).
- Validation table: missing keys, NaN, Inf, t_now <= tSF, R2 <= 0,
  rShell <= R2, Qi <= 0, length mismatch, short shell, endpoint drift.
- SB99 age band: warns by default, errors with hard_age_bounds.
- Z handling: summary.ZCloud default, z_override precedence, bad override.
- Outer-radius / ambient extension paths.
- Title / formatting / DLAW_ROWS-vs-DLAW_BLOCK split.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from src._functions.unit_conversions import INV_CONV
from src._output.cloudy.dlaw import (
    DEFAULT_DLAW_CLOSE,
    DEFAULT_DLAW_OPEN,
)
from src._output.cloudy.run_loader import load_run
from src._output.cloudy.snapshot_to_deck import (
    SnapshotInvalid,
    snapshot_to_values,
)


MOCK_FULLRUN = Path("outputs/mockOutput/mockFullrun")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_synth_bundle(
    tmp_path: Path,
    *,
    tSF: float = 0.0,
    ZCloud: float = 1.0,
    dens_profile: str = "densPL",
    initial_cloud_r: list[float] | None = None,
    initial_cloud_n: list[float] | None = None,
    model_name: str = "synth",
):
    """Build a minimal RunBundle with synthesised metadata/summary/end."""
    rd = tmp_path / "run"
    rd.mkdir()
    metadata = {
        "model_name": model_name,
        "dens_profile": dens_profile,
        "tSF": tSF,
    }
    if initial_cloud_r is not None:
        metadata["initial_cloud_r_arr"] = list(initial_cloud_r)
        metadata["initial_cloud_n_arr"] = list(initial_cloud_n)
    (rd / "metadata.json").write_text(json.dumps(metadata))
    (rd / f"{model_name}_summary.txt").write_text(f"ZCloud  {ZCloud}\n")
    (rd / "simulationEnd.txt").write_text(
        "Outcome: stopping_time\nDetail: Stopping time reached\nExit Code: 1\n"
    )
    # Need at least one real snapshot for find_data_path to succeed
    (rd / "dictionary.jsonl").write_text(json.dumps({
        "t_now": 0.15, "current_phase": "transition",
        "R2": 1.0, "rShell": 1.5, "Qi": 1e60,
        "shell_r_arr": [1.0, 1.5], "log_shell_n_arr": [58.0, 57.0],
    }) + "\n")
    return load_run(rd)


def _good_snap(t_now: float = 0.15) -> dict:
    """A self-consistent snapshot with R2/rShell endpoints matching shell_r_arr."""
    return {
        "t_now": t_now,
        "R2": 1.0,
        "rShell": 1.5,
        "Qi": 5.835e61,
        "shell_r_arr": [1.0, 1.2, 1.5],
        "log_shell_n_arr": [58.0, 57.5, 57.0],
        "current_phase": "transition",
    }


# --------------------------------------------------------------------------- #
# End-to-end against real mockFullrun
# --------------------------------------------------------------------------- #

def test_e2e_against_mockFullrun_transition_snapshot():
    bundle = load_run(MOCK_FULLRUN)
    # Snapshot closest to t_now = 0.15 Myr (well inside default age band)
    snap = bundle.output.get_at_time(0.15, mode="closest", quiet=True)

    vals = snapshot_to_values(snap, bundle)

    # Substitution keys present and well-typed
    for k in ["TITLE", "AGE_YR", "LOG_QH", "LOG_RIN", "LOG_ROUT", "ZREL",
              "DLAW_BLOCK", "DLAW_ROWS"]:
        assert k in vals, f"missing key {k}"
        assert isinstance(vals[k], str)

    # DLAW_BLOCK has the dlaw header / footer; DLAW_ROWS does NOT
    assert vals["DLAW_BLOCK"].startswith(DEFAULT_DLAW_OPEN + "\n")
    assert vals["DLAW_BLOCK"].rstrip().endswith(DEFAULT_DLAW_CLOSE)
    assert not vals["DLAW_ROWS"].startswith(DEFAULT_DLAW_OPEN)
    assert not vals["DLAW_ROWS"].rstrip().endswith(DEFAULT_DLAW_CLOSE)

    # Z from summary (mockFullrun has ZCloud=1.0)
    assert vals["ZREL"] == "1.0000"

    # log Q(H) ≈ 48.27 for these clusters (cross-checked against mockFullrun)
    log_qh = float(vals["LOG_QH"])
    assert 47.5 < log_qh < 49.0

    # Diagnostics
    diag = vals["_diagnostics"]
    assert diag["phase"] == snap["current_phase"]
    assert diag["age_yr"] == pytest.approx(diag["age_myr"] * 1e6, rel=1e-12)
    assert diag["n_dlaw_rows"] >= 10
    assert diag["ambient_extended"] is False  # no --radius-out


# --------------------------------------------------------------------------- #
# Regression guards: the two arithmetic bugs from the v4 plan
# --------------------------------------------------------------------------- #

def test_age_yr_formula_is_just_times_1e6(tmp_path):
    """AGE_YR = (t_now - tSF) * 1e6 — NOT * Myr2s * s_per_yr (v4 bug)."""
    bundle = _make_synth_bundle(tmp_path, tSF=0.0)
    snap = _good_snap(t_now=0.0971)  # 9.71e4 yr — below default band, so warn
    with pytest.warns(UserWarning, match="outside SB99 band"):
        vals = snapshot_to_values(snap, bundle)
    # 0.0971 Myr × 1e6 = 9.71e4 yr exactly
    assert float(vals["AGE_YR"]) == pytest.approx(9.71e4, rel=1e-9)


def test_age_yr_subtracts_tSF(tmp_path):
    """Cluster age = t_now - tSF, not t_now alone."""
    # Pick values that subtract exactly in IEEE-754 (powers of 2) so the
    # band check is unambiguous: 0.75 - 0.25 = 0.5 Myr → 5e5 yr (in band).
    bundle = _make_synth_bundle(tmp_path, tSF=0.25)
    snap = _good_snap(t_now=0.75)
    vals = snapshot_to_values(snap, bundle)
    assert float(vals["AGE_YR"]) == pytest.approx(5.0e5, rel=1e-9)


def test_log_qh_sign_is_subtraction(tmp_path):
    """LOG_QH = log10(Qi) - log10(Myr2s), NOT + (v4 bug)."""
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    snap["Qi"] = 5.835e61  # mockFullrun representative value
    vals = snapshot_to_values(snap, bundle)
    expected = math.log10(5.835e61) - math.log10(INV_CONV.Myr2s)
    assert float(vals["LOG_QH"]) == pytest.approx(expected, abs=1e-4)
    # Crude sanity: ~48.27, not ~75.27 (the wrong-sign result)
    assert 47 < float(vals["LOG_QH"]) < 49


# --------------------------------------------------------------------------- #
# Validation table
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("missing_key", [
    "t_now", "R2", "rShell", "Qi", "shell_r_arr", "log_shell_n_arr",
    "current_phase",
])
def test_rejects_missing_required_key(tmp_path, missing_key):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    del snap[missing_key]
    with pytest.raises(SnapshotInvalid, match="missing required keys"):
        snapshot_to_values(snap, bundle)


@pytest.mark.parametrize("field, value", [
    ("t_now", float("nan")), ("R2", float("nan")), ("rShell", float("nan")),
    ("Qi", float("nan")),
    ("t_now", float("inf")), ("R2", float("inf")), ("rShell", float("inf")),
    ("Qi", float("inf")),
])
def test_rejects_nonfinite_scalar(tmp_path, field, value):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    snap[field] = value
    with pytest.raises(SnapshotInvalid, match="not finite"):
        snapshot_to_values(snap, bundle)


def test_rejects_nonfinite_in_shell_arrays(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    snap["log_shell_n_arr"] = [58.0, float("nan"), 57.0]
    with pytest.raises(SnapshotInvalid, match="non-finite"):
        snapshot_to_values(snap, bundle)


def test_rejects_t_now_le_tSF(tmp_path):
    bundle = _make_synth_bundle(tmp_path, tSF=0.5)
    snap = _good_snap(t_now=0.5)  # age = 0
    with pytest.raises(SnapshotInvalid, match="non-positive"):
        snapshot_to_values(snap, bundle)


def test_rejects_nonpositive_R2(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    snap["R2"] = 0.0
    snap["shell_r_arr"][0] = 0.0  # keep endpoint consistency to reach the R2 check
    with pytest.raises(SnapshotInvalid, match="R2 must be positive"):
        snapshot_to_values(snap, bundle)


def test_rejects_rShell_le_R2(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    snap["rShell"] = 1.0  # equal to R2
    snap["shell_r_arr"][-1] = 1.0
    with pytest.raises(SnapshotInvalid, match="must exceed R2"):
        snapshot_to_values(snap, bundle)


def test_rejects_nonpositive_Qi(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    snap["Qi"] = 0.0
    with pytest.raises(SnapshotInvalid, match="Qi must be positive"):
        snapshot_to_values(snap, bundle)


def test_rejects_shell_length_mismatch(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    snap["log_shell_n_arr"] = [58.0, 57.5]  # one short
    with pytest.raises(SnapshotInvalid, match="length mismatch"):
        snapshot_to_values(snap, bundle)


def test_rejects_short_shell(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    snap["shell_r_arr"] = [1.0]
    snap["log_shell_n_arr"] = [58.0]
    with pytest.raises(SnapshotInvalid, match=">= 2 points"):
        snapshot_to_values(snap, bundle)


def test_rejects_endpoint_drift_from_R2(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    snap["shell_r_arr"][0] = 1.0 + 1e-9  # drift beyond rel_tol=1e-12
    with pytest.raises(SnapshotInvalid, match="does not match R2"):
        snapshot_to_values(snap, bundle)


def test_rejects_endpoint_drift_from_rShell(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    snap["shell_r_arr"][-1] = 1.5 + 1e-9
    with pytest.raises(SnapshotInvalid, match="does not match rShell"):
        snapshot_to_values(snap, bundle)


# --------------------------------------------------------------------------- #
# SB99 age band
# --------------------------------------------------------------------------- #

def test_age_below_band_warns(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap(t_now=0.05)  # 5e4 yr — below default 1e5
    with pytest.warns(UserWarning, match="outside SB99 band"):
        snapshot_to_values(snap, bundle)


def test_age_above_band_warns(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap(t_now=200.0)  # 2e8 yr — above default 1e8
    with pytest.warns(UserWarning, match="outside SB99 band"):
        snapshot_to_values(snap, bundle)


def test_hard_age_bounds_promotes_warning_to_error(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap(t_now=0.05)
    with pytest.raises(SnapshotInvalid, match="outside SB99 band"):
        snapshot_to_values(snap, bundle, hard_age_bounds=True)


def test_age_inside_band_no_warning(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap(t_now=0.15)  # 1.5e5 yr — well inside band
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        snapshot_to_values(snap, bundle)  # must not warn


# --------------------------------------------------------------------------- #
# Z handling
# --------------------------------------------------------------------------- #

def test_z_default_from_summary(tmp_path):
    bundle = _make_synth_bundle(tmp_path, ZCloud=0.5)
    snap = _good_snap()
    vals = snapshot_to_values(snap, bundle)
    assert vals["ZREL"] == "0.5000"
    assert vals["_diagnostics"]["z_used"] == 0.5


def test_z_override_takes_precedence(tmp_path):
    bundle = _make_synth_bundle(tmp_path, ZCloud=1.0)
    snap = _good_snap()
    vals = snapshot_to_values(snap, bundle, z_override=0.2)
    assert vals["ZREL"] == "0.2000"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_z_override_rejects_bad_values(tmp_path, bad):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    with pytest.raises(SnapshotInvalid, match="z_override"):
        snapshot_to_values(snap, bundle, z_override=bad)


# --------------------------------------------------------------------------- #
# Outer radius / ambient extension
# --------------------------------------------------------------------------- #

def test_radius_out_default_is_rShell(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    vals = snapshot_to_values(snap, bundle)
    assert vals["_diagnostics"]["r_out_pc"] == snap["rShell"]
    assert vals["_diagnostics"]["ambient_extended"] is False


def test_radius_out_extends_with_ambient(tmp_path):
    bundle = _make_synth_bundle(
        tmp_path,
        # Ambient profile in pc / pc^-3 (linear). Covers from far inside the
        # shell out to 5 pc; only points past rShell=1.5 are spliced.
        initial_cloud_r=[0.001, 1.0, 2.0, 3.0, 5.0],
        initial_cloud_n=[1e58, 1e58, 1e54, 1e54, 1e54],
    )
    snap = _good_snap()
    vals = snapshot_to_values(snap, bundle, radius_out_pc=5.0)
    assert vals["_diagnostics"]["r_out_pc"] == 5.0
    assert vals["_diagnostics"]["ambient_extended"] is True
    # The dlaw block now extends past rShell — last row's r should be near 5 pc
    last_row = vals["DLAW_ROWS"].split("\n")[-1]
    log_r_cm_last = float(last_row.split()[1])
    expected_log_r = math.log10(5.0) + math.log10(INV_CONV.pc2cm)
    assert log_r_cm_last == pytest.approx(expected_log_r, abs=1e-4)


def test_radius_out_below_rShell_errors(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()  # rShell = 1.5
    with pytest.raises(SnapshotInvalid, match="must be >= rShell"):
        snapshot_to_values(snap, bundle, radius_out_pc=1.0)


def test_radius_out_nonfinite_errors(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    with pytest.raises(SnapshotInvalid, match="must be finite"):
        snapshot_to_values(snap, bundle, radius_out_pc=float("inf"))


def test_extension_requested_without_metadata_errors(tmp_path):
    # Bundle without initial_cloud_*_arr in metadata
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    with pytest.raises(SnapshotInvalid, match="metadata lacks"):
        snapshot_to_values(snap, bundle, radius_out_pc=5.0)


def test_extend_with_ambient_false_uses_no_ambient(tmp_path):
    """Without extension, r_out=rShell so dlaw bracket check is satisfied
    by the shell alone — extend_with_ambient=False is fine here."""
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    vals = snapshot_to_values(snap, bundle, extend_with_ambient=False)
    assert vals["_diagnostics"]["ambient_extended"] is False


def test_extend_disabled_but_radius_out_past_rShell_errors(tmp_path):
    """If the user asks for r_out > rShell but disables ambient, fail in
    snapshot_to_deck (clear attribution) rather than deeper in dlaw."""
    bundle = _make_synth_bundle(
        tmp_path,
        initial_cloud_r=[1.0, 5.0], initial_cloud_n=[1e58, 1e54],
    )
    snap = _good_snap()  # rShell = 1.5
    with pytest.raises(SnapshotInvalid, match="extend_with_ambient=True"):
        snapshot_to_values(
            snap, bundle, radius_out_pc=5.0, extend_with_ambient=False,
        )


# --------------------------------------------------------------------------- #
# Title and formatting
# --------------------------------------------------------------------------- #

def test_title_format(tmp_path):
    bundle = _make_synth_bundle(tmp_path, model_name="my_model")
    snap = _good_snap(t_now=0.15)
    snap["current_phase"] = "transition"
    vals = snapshot_to_values(snap, bundle)
    assert vals["TITLE"] == "TRINITY my_model phase=transition age=0.1500Myr"


def test_log_rin_log_rout_use_pc2cm(tmp_path):
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()  # R2=1.0, rShell=1.5
    vals = snapshot_to_values(snap, bundle)
    log_pc_cm = math.log10(INV_CONV.pc2cm)
    assert float(vals["LOG_RIN"]) == pytest.approx(log_pc_cm, abs=1e-4)
    assert float(vals["LOG_ROUT"]) == pytest.approx(
        math.log10(1.5) + log_pc_cm, abs=1e-4
    )


def test_dlaw_rows_is_block_minus_open_close(tmp_path):
    """DLAW_ROWS should equal DLAW_BLOCK with the first and last lines stripped."""
    bundle = _make_synth_bundle(tmp_path)
    snap = _good_snap()
    vals = snapshot_to_values(snap, bundle)
    block_lines = vals["DLAW_BLOCK"].split("\n")
    rows_lines = vals["DLAW_ROWS"].split("\n")
    assert len(block_lines) == len(rows_lines) + 2
    assert block_lines[1:-1] == rows_lines
    assert vals["_diagnostics"]["n_dlaw_rows"] == len(rows_lines)
