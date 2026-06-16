"""
Tests for ``trinity._output.cloudy.trinity_to_cloudy`` (the CLI driver).

Covers:
- ``render_template``: substitutes {{KEY}}, leaves <<<EDIT_ME>>> alone, raises
  ``UnsubstitutedPlaceholder`` on leftovers.
- Argument parsing: exactly-one-picker mutex, --all + --dry-run mutex.
- Snapshot picking: --age, --t-now, --index (incl. negative), --phase.
- Status gate: SUCCESS proceeds, FAILED requires --force.
- End-to-end against ``mockFullrun``: single --age writes deck + sidecar +
  linelist; --all writes 178 decks + manifest.json with the documented
  schema.
- Prefix derivation (dots replaced by 'p', filename-safe).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from trinity._output.cloudy.trinity_to_cloudy import (
    LINELIST_FILENAME,
    UnsubstitutedPlaceholder,
    _build_prefix,
    _parse_args,
    _pick_snapshots,
    main,
    render_template,
)


MOCK_FULLRUN = Path(__file__).resolve().parents[1] / "outputs" / "mockOutput" / "mockFullrun"


# --------------------------------------------------------------------------- #
# render_template
# --------------------------------------------------------------------------- #

def test_render_template_substitutes_keys():
    out = render_template("hello {{NAME}}", {"NAME": "world"})
    assert out == "hello world"


def test_render_template_preserves_edit_me_sentinel():
    text = 'table star "<<<EDIT_ME>>>" age = {{AGE_YR}} years'
    out = render_template(text, {"AGE_YR": "1.5e5"})
    assert "<<<EDIT_ME>>>" in out
    assert "1.5e5" in out


def test_render_template_raises_on_unsubstituted_placeholder():
    with pytest.raises(UnsubstitutedPlaceholder, match=r"\['MISSING'\]"):
        render_template("{{MISSING}}", {})


def test_render_template_substitutes_repeated_key():
    out = render_template("{{X}}-{{X}}-{{X}}", {"X": "a"})
    assert out == "a-a-a"


def test_render_template_handles_multiline_value():
    """DLAW_BLOCK is multi-line; the renderer must pass embedded newlines."""
    out = render_template("dlaw:\n{{BLOCK}}\nend", {"BLOCK": "row1\nrow2\nrow3"})
    assert out == "dlaw:\nrow1\nrow2\nrow3\nend"


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def test_parse_args_requires_picker():
    with pytest.raises(SystemExit):
        _parse_args(["-F", str(MOCK_FULLRUN)])


def test_parse_args_rejects_all_with_dry_run():
    with pytest.raises(SystemExit):
        _parse_args(["-F", str(MOCK_FULLRUN), "--all", "--dry-run"])


def test_parse_args_rejects_two_pickers():
    # argparse mutex catches this
    with pytest.raises(SystemExit):
        _parse_args(["-F", str(MOCK_FULLRUN), "--age", "0.1", "--index", "0"])


def test_parse_args_age_picker():
    args = _parse_args(["-F", str(MOCK_FULLRUN), "--age", "0.15"])
    assert args.age == 0.15
    assert args.all is False
    assert args.dry_run is False


# --------------------------------------------------------------------------- #
# Prefix derivation
# --------------------------------------------------------------------------- #

def test_build_prefix_replaces_dots_with_p(tmp_path):
    """Auto-built prefix must avoid '.' (problematic in filenames + CLOUDY)."""
    from trinity._output.cloudy.trinity_to_cloudy import PickedSnapshot
    from trinity._output.cloudy.run_loader import load_run
    bundle = load_run(MOCK_FULLRUN)
    snap = bundle.output[170]
    pick = PickedSnapshot(index=170, snap=snap)
    args = _parse_args(["-F", str(MOCK_FULLRUN), "--age", "0.15"])
    prefix = _build_prefix(args, bundle, pick, age_myr=0.1482, phase="momentum")
    assert "." not in prefix
    assert "0p1482" in prefix
    assert prefix == "4e3_sfe001_n5e2_PL0_170_momentum_t0p1482myr"


def test_build_prefix_sanitizes_user_supplied(tmp_path):
    from trinity._output.cloudy.trinity_to_cloudy import PickedSnapshot
    from trinity._output.cloudy.run_loader import load_run
    bundle = load_run(MOCK_FULLRUN)
    pick = PickedSnapshot(index=0, snap=bundle.output[0])
    args = _parse_args(
        ["-F", str(MOCK_FULLRUN), "--age", "0.15", "--prefix", "my run/with bad chars!"]
    )
    prefix = _build_prefix(args, bundle, pick, age_myr=0.1, phase="energy")
    # spaces, slashes, and '!' all get replaced
    assert "/" not in prefix
    assert " " not in prefix
    assert "!" not in prefix


# --------------------------------------------------------------------------- #
# Snapshot picking
# --------------------------------------------------------------------------- #

def test_pick_age(tmp_path):
    from trinity._output.cloudy.run_loader import load_run
    bundle = load_run(MOCK_FULLRUN)
    args = _parse_args(["-F", str(MOCK_FULLRUN), "--age", "0.15"])
    picks = _pick_snapshots(bundle, args)
    assert len(picks) == 1
    # mockFullrun has tSF=0; closest snapshot to 0.15 Myr is the momentum-phase
    # one at t_now ≈ 0.148 Myr
    assert picks[0].snap.t_now == pytest.approx(0.1482, abs=1e-3)


def test_pick_index_negative(tmp_path):
    from trinity._output.cloudy.run_loader import load_run
    bundle = load_run(MOCK_FULLRUN)
    args = _parse_args(["-F", str(MOCK_FULLRUN), "--index", "-1"])
    picks = _pick_snapshots(bundle, args)
    assert len(picks) == 1
    assert picks[0].index == 177  # last of 178 snapshots


def test_pick_index_out_of_range_errors():
    from trinity._output.cloudy.run_loader import load_run
    bundle = load_run(MOCK_FULLRUN)
    args = _parse_args(["-F", str(MOCK_FULLRUN), "--index", "999"])
    with pytest.raises(SystemExit, match="out of range"):
        _pick_snapshots(bundle, args)


def test_pick_phase_last():
    from trinity._output.cloudy.run_loader import load_run
    bundle = load_run(MOCK_FULLRUN)
    args = _parse_args(["-F", str(MOCK_FULLRUN), "--phase", "transition", "--pick", "last"])
    picks = _pick_snapshots(bundle, args)
    # transition spans indices 147..163 in mockFullrun
    assert picks[0].snap.phase == "transition"
    assert picks[0].index == 163


def test_pick_phase_first():
    from trinity._output.cloudy.run_loader import load_run
    bundle = load_run(MOCK_FULLRUN)
    args = _parse_args(["-F", str(MOCK_FULLRUN), "--phase", "transition", "--pick", "first"])
    picks = _pick_snapshots(bundle, args)
    assert picks[0].snap.phase == "transition"
    assert picks[0].index == 147


def test_pick_phase_unknown_errors():
    from trinity._output.cloudy.run_loader import load_run
    bundle = load_run(MOCK_FULLRUN)
    args = _parse_args(["-F", str(MOCK_FULLRUN), "--phase", "nonexistent"])
    with pytest.raises(SystemExit, match="no snapshots in phase"):
        _pick_snapshots(bundle, args)


def test_pick_all():
    from trinity._output.cloudy.run_loader import load_run
    bundle = load_run(MOCK_FULLRUN)
    args = _parse_args(["-F", str(MOCK_FULLRUN), "--all"])
    picks = _pick_snapshots(bundle, args)
    assert len(picks) == 178


# --------------------------------------------------------------------------- #
# End-to-end runs against mockFullrun
# --------------------------------------------------------------------------- #

def test_e2e_age_writes_deck_sidecar_and_linelist(tmp_path):
    out_dir = tmp_path / "out"
    rc = main([
        "-F", str(MOCK_FULLRUN),
        "--age", "0.15",
        "--out", str(out_dir),
    ])
    assert rc == 0
    # Deck + sidecar + linelist
    decks = list(out_dir.glob("*.in"))
    sidecars = list(out_dir.glob("*.dlaw.txt"))
    assert len(decks) == 1
    assert len(sidecars) == 1
    assert (out_dir / LINELIST_FILENAME).is_file()

    deck_text = decks[0].read_text()
    # Default SB99 sentinel preserved in the substituted deck; no leftover {{KEY}}
    assert '"<<<EDIT_ME>>>"' in deck_text
    assert "{{" not in deck_text
    assert "}}" not in deck_text
    # Bundled template hardcodes the linelist filename verbatim
    assert f'"{LINELIST_FILENAME}"' in deck_text


def test_e2e_age_dry_run_does_not_write(tmp_path, capsys):
    out_dir = tmp_path / "out"
    rc = main([
        "-F", str(MOCK_FULLRUN),
        "--age", "0.15",
        "--out", str(out_dir),
        "--dry-run",
    ])
    assert rc == 0
    assert not out_dir.exists()
    captured = capsys.readouterr()
    assert "title TRINITY" in captured.out
    assert "<<<EDIT_ME>>>" in captured.out
    assert "dlaw table radius" in captured.out


def test_e2e_all_writes_manifest_and_all_decks(tmp_path):
    out_dir = tmp_path / "out"
    rc = main([
        "-F", str(MOCK_FULLRUN),
        "--all",
        "--out", str(out_dir),
    ])
    assert rc == 0
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.is_file()
    records = json.loads(manifest_path.read_text())
    assert len(records) == 178
    # Schema check on every record
    expected_keys = {
        "index", "t_now_myr", "phase", "age_myr", "age_yr",
        "n_dlaw_rows", "ambient_extended", "z_used",
        "prefix", "deck", "dlaw_sidecar", "status", "reason",
    }
    for r in records:
        assert set(r.keys()) == expected_keys
    # All-ok in mockFullrun
    assert all(r["status"] == "ok" for r in records)
    # One .in and one .dlaw.txt per snapshot, plus linelist + manifest
    assert len(list(out_dir.glob("*.in"))) == 178
    assert len(list(out_dir.glob("*.dlaw.txt"))) == 178


def test_e2e_failed_status_requires_force(tmp_path):
    """If the simulationEnd exit code is outside the clean range, refuse without --force."""
    # Synthesise a copy of mockFullrun with a non-clean exit code
    work = tmp_path / "failed_run"
    work.mkdir()
    for src in MOCK_FULLRUN.iterdir():
        if src.is_file():
            (work / src.name).write_bytes(src.read_bytes())
    # Patch the outcome lines to a numerical-error code (out of 0-9 range)
    end_path = work / "simulationEnd.txt"
    end_text = (
        end_path.read_text()
        .replace("Outcome: stopping_time", "Outcome: error_solver")
        .replace("Exit Code: 1", "Exit Code: 22")
    )
    end_path.write_text(end_text)

    out_dir = tmp_path / "out"
    # Without --force: hard exit
    with pytest.raises(SystemExit, match="refusing to convert"):
        main(["-F", str(work), "--age", "0.15", "--out", str(out_dir)])
    # With --force: proceeds
    rc = main(["-F", str(work), "--age", "0.15", "--out", str(out_dir), "--force"])
    assert rc == 0
    assert any(out_dir.glob("*.in"))


def test_e2e_age_records_request_vs_actual_delta(tmp_path, capsys):
    out_dir = tmp_path / "out"
    main([
        "-F", str(MOCK_FULLRUN),
        "--age", "0.15",
        "--out", str(out_dir),
    ])
    captured = capsys.readouterr()
    assert "requested 1.500e+05 yr" in captured.out
    # Closest snapshot is at age ≈ 1.482e5 yr → Δ ≈ 1.83 kyr
    assert re.search(r"Δ=\d+\.\d+ kyr", captured.out)
    assert "TODO" in captured.out


def test_e2e_all_with_dry_run_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main([
            "-F", str(MOCK_FULLRUN),
            "--all", "--dry-run",
            "--out", str(tmp_path / "out"),
        ])


def test_e2e_no_picker_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(["-F", str(MOCK_FULLRUN), "--out", str(tmp_path / "out")])


def test_e2e_z_override_changes_zrel_in_deck(tmp_path):
    out_dir = tmp_path / "out"
    rc = main([
        "-F", str(MOCK_FULLRUN),
        "--age", "0.15",
        "--z-override", "0.2",
        "--out", str(out_dir),
    ])
    assert rc == 0
    deck_text = next(out_dir.glob("*.in")).read_text()
    assert "metals and grains 0.2000" in deck_text


def test_e2e_sb99_default_is_sentinel(tmp_path):
    """Default --sb99 is the sentinel, present verbatim in the deck."""
    out_dir = tmp_path / "out"
    rc = main([
        "-F", str(MOCK_FULLRUN), "--age", "0.15", "--out", str(out_dir),
    ])
    assert rc == 0
    deck_text = next(out_dir.glob("*.in")).read_text()
    assert 'table star "<<<EDIT_ME>>>"' in deck_text


def test_e2e_sb99_flag_overrides(tmp_path, capsys):
    """--sb99 NAME drops NAME into the deck and suppresses the TODO line."""
    out_dir = tmp_path / "out"
    rc = main([
        "-F", str(MOCK_FULLRUN), "--age", "0.15",
        "--sb99", "starburst99_z020.mod",
        "--out", str(out_dir),
    ])
    assert rc == 0
    deck_text = next(out_dir.glob("*.in")).read_text()
    assert 'table star "starburst99_z020.mod"' in deck_text
    assert "<<<EDIT_ME>>>" not in deck_text
    captured = capsys.readouterr()
    assert "TODO" not in captured.out
