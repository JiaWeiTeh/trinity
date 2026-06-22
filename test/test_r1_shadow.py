"""Unit tests for the R1 transition SHADOW criterion (docs/dev/transition/pt4).

`evaluate_r1_shadow` is pure and must never depend on global state. It is the
criterion a future *flip* would act on; here it only feeds shadow logging, so the
truth table is the contract.
"""
import numpy as np

from trinity.phase1b_energy_implicit.run_energy_implicit_phase import evaluate_r1_shadow


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
