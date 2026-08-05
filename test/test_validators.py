"""Phase-6 validator contract tests.

Pins the three validators wired into the registry against the
verbatim error messages the pre-Phase-6 Step-5 block produced.  These
catch any drift if a validator is rewritten or a spec loses its
``validator=`` attachment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trinity._input.errors import ParameterFileError
from trinity._input.read_param import read_param
from trinity._input.registry import (
    COMPANION_RULES,
    REGISTRY,
    _validate_dens_profile,
    _validate_phase1a_segFrac,
    _validate_stop_at_rCloud_nSnap,
    _validate_ZCloud,
    validate_all,
    validate_companions,
)


def _write_param(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The three specs carry the right validator
# ---------------------------------------------------------------------------
def test_ZCloud_spec_has_validator() -> None:
    assert REGISTRY["ZCloud"].validator is _validate_ZCloud


def test_dens_profile_spec_has_validator() -> None:
    assert REGISTRY["dens_profile"].validator is _validate_dens_profile


def test_stop_at_rCloud_nSnap_spec_has_validator() -> None:
    assert REGISTRY["stop_at_rCloud_nSnap"].validator is _validate_stop_at_rCloud_nSnap


# ---------------------------------------------------------------------------
# read_param malformed .param trust boundary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("not_a_param 1\n", r"Invalid parameter\(s\) in bad\.param: not_a_param"),
        ("mCloud\n", r"bad\.param, line 1: Expected format 'key value', got: 'mCloud'"),
    ],
)
def test_read_param_rejects_malformed_user_param(tmp_path: Path, body: str, message: str) -> None:
    path = _write_param(tmp_path, "bad.param", body)

    with pytest.raises(ParameterFileError, match=message):
        read_param(path)


def test_read_param_duplicate_user_key_uses_later_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    duplicate = _write_param(tmp_path, "duplicate.param", "mCloud 1e5\nmCloud 2e5\n")
    first_only = _write_param(tmp_path, "first.param", "mCloud 1e5\n")
    later_only = _write_param(tmp_path, "later.param", "mCloud 2e5\n")

    # Current behavior, not an endorsement; PLAN P5-T6 flags this as a fix candidate.
    duplicate_value = read_param(duplicate)["mCloud"].value
    assert duplicate_value == read_param(later_only)["mCloud"].value
    assert duplicate_value != read_param(first_only)["mCloud"].value


# ---------------------------------------------------------------------------
# ZCloud
# ---------------------------------------------------------------------------
def test_ZCloud_accepts_1() -> None:
    _validate_ZCloud(1, {})  # no raise


def test_ZCloud_rejects_nonsolar() -> None:
    with pytest.raises(ParameterFileError, match="Metallicity Z=0.5 not implemented"):
        _validate_ZCloud(0.5, {})


# ---------------------------------------------------------------------------
# dens_profile
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["densBE", "densPL"])
def test_dens_profile_accepts_valid(value: str) -> None:
    _validate_dens_profile(value, {})  # no raise


def test_dens_profile_rejects_other() -> None:
    with pytest.raises(ParameterFileError, match="Invalid dens_profile 'foo'"):
        _validate_dens_profile("foo", {})


# ---------------------------------------------------------------------------
# stop_at_rCloud_nSnap (validate + coerce)
# ---------------------------------------------------------------------------
class _Item:
    """Minimal stand-in for DescribedItem so the validator can mutate
    ``params['stop_at_rCloud_nSnap'].value`` without a real run."""

    def __init__(self, value):
        self.value = value


def _params_with_nSnap(value):
    return {"stop_at_rCloud_nSnap": _Item(value)}


def test_nSnap_None_accepted() -> None:
    _validate_stop_at_rCloud_nSnap(None, _params_with_nSnap(None))


def test_nSnap_int_accepted_unchanged() -> None:
    p = _params_with_nSnap(3)
    _validate_stop_at_rCloud_nSnap(3, p)
    assert p["stop_at_rCloud_nSnap"].value == 3
    assert isinstance(p["stop_at_rCloud_nSnap"].value, int)


def test_nSnap_whole_number_float_coerced_to_int() -> None:
    """parse_value returns floats for '5'; the validator coerces in place."""
    p = _params_with_nSnap(5.0)
    _validate_stop_at_rCloud_nSnap(5.0, p)
    assert p["stop_at_rCloud_nSnap"].value == 5
    assert isinstance(p["stop_at_rCloud_nSnap"].value, int)


def test_nSnap_fractional_float_rejected() -> None:
    with pytest.raises(ParameterFileError, match="whole-number integer"):
        _validate_stop_at_rCloud_nSnap(2.5, _params_with_nSnap(2.5))


def test_nSnap_negative_rejected() -> None:
    with pytest.raises(ParameterFileError, match="non-negative integer"):
        _validate_stop_at_rCloud_nSnap(-1.0, _params_with_nSnap(-1.0))


def test_nSnap_bool_rejected() -> None:
    """bool is technically int — verify it's explicitly rejected."""
    with pytest.raises(ParameterFileError, match="non-negative integer"):
        _validate_stop_at_rCloud_nSnap(True, _params_with_nSnap(True))


def test_nSnap_string_rejected() -> None:
    with pytest.raises(ParameterFileError, match="non-negative integer"):
        _validate_stop_at_rCloud_nSnap("foo", _params_with_nSnap("foo"))


# ---------------------------------------------------------------------------
# phase1a_segFrac — the phase-1a segment schedule (docs/dev/phase1a-init/)
#
# dt_segment = phase1a_segFrac*(t_now - tSF), driving terms frozen across the
# segment. run_energy_phase.py falls back to the fixed 30-yr SEGMENT_DURATION
# whenever that product is <= 0, which is the documented meaning of 0 — but it
# also swallows every negative value, so without a validator a typo'd -0.1
# silently runs stock segments and the run looks fine. That silent revert is
# what these pin.
# ---------------------------------------------------------------------------
def test_phase1a_segFrac_spec_has_validator() -> None:
    assert REGISTRY["phase1a_segFrac"].validator is _validate_phase1a_segFrac


@pytest.mark.parametrize("value", [0.0, 0.03, 0.1, 0.3, 1.0])
def test_phase1a_segFrac_accepts_in_range(value: float) -> None:
    """0 is the documented fixed-segment fallback; 0.03-0.3 is the measured
    convergence range (docs/dev/phase1a-init/data/gate_results.csv); 1.0 is the
    boundary where a segment spans the bubble's whole present age."""
    _validate_phase1a_segFrac(value, {})


def test_phase1a_segFrac_rejects_negative() -> None:
    """The trap: negative -> dt_segment < 0 -> the <= 0 fallback -> stock
    30-yr segments, with no error and no warning."""
    with pytest.raises(ParameterFileError, match=r"0 <= phase1a_segFrac <= 1"):
        _validate_phase1a_segFrac(-0.1, {})


def test_phase1a_segFrac_rejects_above_one() -> None:
    """Above 1 a single frozen segment covers more than the bubble's entire
    age — the defect class this parameter exists to prevent."""
    with pytest.raises(ParameterFileError, match=r"0 <= phase1a_segFrac <= 1"):
        _validate_phase1a_segFrac(1.5, {})


@pytest.mark.parametrize("value", [True, "0.1", None])
def test_phase1a_segFrac_rejects_non_numeric(value) -> None:
    with pytest.raises(ParameterFileError, match="Must be a number"):
        _validate_phase1a_segFrac(value, {})


def test_read_param_rejects_negative_phase1a_segFrac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end at the .param trust boundary, not just the validator."""
    monkeypatch.chdir(tmp_path)
    path = _write_param(
        tmp_path, "seg.param", "mCloud 1e5\nsfe 0.3\nphase1a_segFrac -0.1\n"
    )

    with pytest.raises(ParameterFileError, match=r"0 <= phase1a_segFrac <= 1"):
        read_param(path)


# ---------------------------------------------------------------------------
# validate_all skips missing keys (so densBE/densPL keys don't cross over)
# ---------------------------------------------------------------------------
def test_validate_all_skips_missing_keys() -> None:
    """A spec with a validator whose key isn't in params is a no-op,
    not a KeyError."""
    validate_all({})  # empty params, must not raise


# ---------------------------------------------------------------------------
# validate_companions: trigger/companion bundle enforcement
# ---------------------------------------------------------------------------
def test_validate_companions_empty_user_dict_passes() -> None:
    """All-defaults case: user touches nothing, no trigger fires."""
    validate_companions({})  # no raise


def test_validate_companions_trigger_without_companion_raises_PL() -> None:
    with pytest.raises(ParameterFileError, match="densPL_alpha"):
        validate_companions({"dens_profile": "densPL"})


def test_validate_companions_trigger_without_companion_raises_BE() -> None:
    with pytest.raises(ParameterFileError, match="densBE_Omega"):
        validate_companions({"dens_profile": "densBE"})


def test_validate_companions_trigger_with_companion_passes_PL() -> None:
    validate_companions({"dens_profile": "densPL", "densPL_alpha": 0})


def test_validate_companions_trigger_with_companion_passes_BE() -> None:
    validate_companions({"dens_profile": "densBE", "densBE_Omega": 14.1})


def test_validate_companions_unrelated_user_keys_ignored() -> None:
    """User-set keys that aren't a CompanionRule trigger don't gate anything."""
    validate_companions({"mCloud": 1e7, "sfe": 0.01})


def test_companion_rules_targets_are_real_spec_names() -> None:
    """Every trigger and companion in COMPANION_RULES is an actual spec."""
    bad = []
    for rule in COMPANION_RULES:
        if rule.trigger not in REGISTRY:
            bad.append(("trigger", rule.trigger))
        for companions in rule.requires.values():
            for companion in companions:
                if companion not in REGISTRY:
                    bad.append(("companion", companion))
    assert not bad, f"CompanionRule references unknown specs: {bad}"
