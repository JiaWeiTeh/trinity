"""Phase-6 validator contract tests.

Pins the three validators wired into the registry against the
verbatim error messages the pre-Phase-6 Step-5 block produced.  These
catch any drift if a validator is rewritten or a spec loses its
``validator=`` attachment.
"""
from __future__ import annotations

import pytest

from trinity._input.errors import ParameterFileError
from trinity._input.registry import (
    COMPANION_RULES,
    REGISTRY,
    _validate_dens_profile,
    _validate_stop_at_rCloud_nSnap,
    _validate_ZCloud,
    validate_all,
    validate_companions,
)


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
