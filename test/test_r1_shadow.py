"""Unit tests for the R1 transition SHADOW criterion (docs/dev/transition/pt4).

`evaluate_r1_shadow` is pure and must never depend on global state. It is the
criterion a future *flip* would act on; here it only feeds shadow logging, so the
truth table is the contract.
"""
import numpy as np

import pytest

from trinity.phase1b_energy_implicit.run_energy_implicit_phase import (
    evaluate_r1_shadow,
    parse_transition_triggers,
    r1_transition_decision,
)


def test_blowout_fires_when_R2_exceeds_rCloud():
    assert evaluate_r1_shadow(R2=11.0, rCloud=10.0, edot_balance=1.0) == (True, False)


def test_blowout_not_fired_inside_cloud():
    assert evaluate_r1_shadow(R2=9.0, rCloud=10.0, edot_balance=1.0) == (False, False)


def test_ebpeak_fires_when_net_energy_nonpositive():
    assert evaluate_r1_shadow(R2=1.0, rCloud=10.0, edot_balance=0.0) == (False, True)
    assert evaluate_r1_shadow(R2=1.0, rCloud=10.0, edot_balance=-5.0) == (False, True)


def test_ebpeak_not_fired_while_growing():
    assert evaluate_r1_shadow(R2=1.0, rCloud=10.0, edot_balance=5.0) == (False, False)


def test_nonfinite_and_none_are_safe():
    assert evaluate_r1_shadow(1.0, 10.0, None) == (False, False)
    assert evaluate_r1_shadow(1.0, 10.0, float("nan")) == (False, False)
    # rCloud None/0 => no blowout, but ebpeak still evaluates
    assert evaluate_r1_shadow(1.0, None, -1.0) == (False, True)
    assert evaluate_r1_shadow(1.0, 0.0, 1.0) == (False, False)


def test_k_blowout_scaling():
    assert evaluate_r1_shadow(R2=15.0, rCloud=10.0, edot_balance=1.0, k_blowout=2.0) == (False, False)
    assert evaluate_r1_shadow(R2=21.0, rCloud=10.0, edot_balance=1.0, k_blowout=2.0) == (True, False)


def test_both_can_fire_together():
    assert evaluate_r1_shadow(R2=11.0, rCloud=10.0, edot_balance=-1.0) == (True, True)


# --- parse_transition_triggers: comma-separated SET, 'r1' alias, validation ---

def test_parse_default_is_cooling_balance_only():
    assert parse_transition_triggers("cooling_balance") == frozenset({"cooling_balance"})


def test_parse_multiple_comma_separated():
    assert parse_transition_triggers("cooling_balance,blowout") == frozenset(
        {"cooling_balance", "blowout"})
    assert parse_transition_triggers(" blowout , ebpeak ") == frozenset({"blowout", "ebpeak"})


def test_parse_r1_alias_expands():
    assert parse_transition_triggers("r1") == frozenset({"blowout", "ebpeak"})
    assert parse_transition_triggers("cooling_balance,r1") == frozenset(
        {"cooling_balance", "blowout", "ebpeak"})


def test_parse_unknown_token_raises():
    with pytest.raises(ValueError):
        parse_transition_triggers("blowowt")
    with pytest.raises(ValueError):
        parse_transition_triggers("cooling_balance,nonsense")


# --- r1_transition_decision: which R1 criterion DRIVES, given the active SET ---

def test_decision_default_set_never_drives_r1():
    # cooling_balance-only set => R1 (blowout/ebpeak) never drives (cooling is inline)
    s = parse_transition_triggers("cooling_balance")
    assert r1_transition_decision(s, True, True) is None


def test_decision_blowout_only_on_blowout():
    s = parse_transition_triggers("blowout")
    assert r1_transition_decision(s, True, False) == "blowout"
    assert r1_transition_decision(s, False, True) is None


def test_decision_combined_set_blowout_precedence():
    s = parse_transition_triggers("cooling_balance,blowout,ebpeak")
    assert r1_transition_decision(s, True, True) == "blowout"
    assert r1_transition_decision(s, False, True) == "ebpeak"
    assert r1_transition_decision(s, False, False) is None
