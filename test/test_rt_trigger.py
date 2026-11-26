"""blowout_rule 'rt', the Rayleigh-Taylor rule of the blowout trigger (Paper II Sect. 2.6;
docs/dev/turnaround-stall/PLAN_RT.md).

Past the cloud edge the shell vents once a Rayleigh-Taylor ripple as long as the shell is thick has grown by the
factor G = rt_growth_factor while the bubble is over-pressured: N_Delta = sum over segments of
sqrt(max(a, 0) / Delta) dt > ln G and Pb > PISM, with a the shell's mean acceleration over the segment. These tests pin
the count, the gate, the rule's place beside the blowout token (a rule of that route, not a route of its own) and the
validators. Values are those of a shell of 10 pc leaving its cloud at 20 pc/Myr with Delta / R = 1.4e-3 (the median at
the edge in the BR pilot).
"""
from types import SimpleNamespace

import numpy as np
import pytest

import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
from trinity._input.errors import ParameterFileError
from trinity._input.registry import SPECS, _validate_blowout_rule, _validate_rt_growth_factor, _validate_vent_model
from trinity.phase1b_energy_implicit.run_energy_implicit_phase import parse_transition_triggers, rt_count_step, rt_gate

DELTA = 0.014        # pc: 1.4e-3 of a 10 pc shell
U0 = 20.0            # pc/Myr
A = 80.0             # pc/Myr^2: Q = a R / u^2 = 2 at the edge


def _ns(**kw):
    """Hand-built params: {key: object with .value}, as the validators and helpers read them."""
    return {k: SimpleNamespace(value=v) for k, v in kw.items()}


def test_constant_push_counts_sqrt_a_over_delta_times_t():
    N, t, u = 0.0, 1.0, U0
    for _ in range(10):
        N = rt_count_step(N, t, u, t + 0.005, u + A * 0.005, DELTA)
        t, u = t + 0.005, u + A * 0.005
    assert N == pytest.approx(np.sqrt(A / DELTA) * 0.05, rel=1e-12)


def test_deceleration_pauses_the_count_without_resetting_it():
    assert rt_count_step(2.0, 1.0, U0, 1.005, U0 - A * 0.005, DELTA) == 2.0
    assert rt_count_step(2.0, 1.0, U0, 1.005, U0, DELTA) == 2.0


def test_zero_length_segment_or_thickness_leaves_the_count():
    assert rt_count_step(1.5, 1.0, U0, 1.0, U0 + 1.0, DELTA) == 1.5
    assert rt_count_step(1.5, 1.0, U0, 1.005, U0 + 1.0, 0.0) == 1.5


def test_gate_needs_growth_overpressure_and_the_edge_outside():
    ln100, rs, rc = np.log(100.0), 10.2, 10.0
    assert rt_gate(ln100 + 0.01, 100.0, 3.0e6, 1.0e6, rs, rc)          # grown, over-pressured, outside the cloud
    assert not rt_gate(ln100 - 0.01, 100.0, 3.0e6, 1.0e6, rs, rc)      # not yet grown a hundredfold
    assert not rt_gate(ln100 + 0.01, 100.0, 1.0e6, 1.0e6, rs, rc)      # Pb = PISM: no hot gas can leave
    assert not rt_gate(ln100 + 0.01, 100.0, 0.5e6, 1.0e6, rs, rc)      # under-pressured
    assert not rt_gate(ln100 + 0.01, 100.0, 3.0e6, 1.0e6, 9.8, rc)     # fell back inside: a hole opens into the cloud
    assert rt_gate(np.log(10.0) + 0.01, 10.0, 3.0e6, 1.0e6, rs, rc)    # G = 10
    assert rt_gate(ln100 + 0.01, 100.0, 3.0e6, 0.0, rs, rc)            # PISM = 0


def test_rt_is_a_rule_of_blowout_not_a_token():
    with pytest.raises(ValueError):
        parse_transition_triggers("cooling_balance,rt")
    assert parse_transition_triggers("r1") == frozenset({"blowout", "ebpeak"})


def test_blowout_rule_spec_and_accepted_settings():
    spec = next(s for s in SPECS if s.name == "blowout_rule")
    assert spec.default == "geometric" and spec.validator is _validate_blowout_rule and spec.run_const
    _validate_blowout_rule("geometric", {})                                       # the default needs nothing
    _validate_blowout_rule("geometric", _ns(phii_scheme="c3c", transition_trigger="cooling_balance"))
    _validate_blowout_rule("rt", _ns(phii_scheme="swept", transition_trigger="cooling_balance,blowout"))
    _validate_blowout_rule("rt", _ns(transition_trigger="r1"))                     # phii_scheme defaults to swept
    _validate_blowout_rule("rt", _ns(transition_trigger="blowout", start_scheme="auto"))   # warns, accepted


def test_blowout_rule_refusals():
    for bad in ("RT", "Geometric", "none", "", None):
        with pytest.raises(ParameterFileError):
            _validate_blowout_rule(bad, {})
    with pytest.raises(ParameterFileError):                                       # rt reads the swept thickness
        _validate_blowout_rule("rt", _ns(phii_scheme="c3c", transition_trigger="cooling_balance,blowout"))
    with pytest.raises(ParameterFileError):                                       # no energy phase, so no blowout
        _validate_blowout_rule("rt", _ns(start_scheme="stromgren", transition_trigger="cooling_balance,blowout"))
    for tt in ("cooling_balance", "cooling_balance,ebpeak"):                       # rt decides only the blowout route
        with pytest.raises(ParameterFileError):
            _validate_blowout_rule("rt", _ns(phii_scheme="swept", transition_trigger=tt))


def test_an_rt_vent_is_a_blowout_vent():
    assert get_bubbleParams.VENT_CHANNELS == ("blowout",)
    assert get_bubbleParams.vented(_ns(vent_model="ambient", transition_channel="blowout")) is True
    _validate_vent_model("ambient", _ns(phii_scheme="swept", transition_trigger="cooling_balance,blowout",
                                        blowout_rule="rt"))


def test_growth_factor_spec_and_validator():
    spec = next(s for s in SPECS if s.name == "rt_growth_factor")
    assert float(spec.default) == 100.0 and spec.validator is _validate_rt_growth_factor and spec.run_const
    for ok in (100, 100.0, 10, 1000.0, 1.5):
        _validate_rt_growth_factor(ok, {})
    for bad in (1, 1.0, 0.5, 0, -3.0, True, "100", None, float("inf"), float("nan")):
        with pytest.raises(ParameterFileError):
            _validate_rt_growth_factor(bad, {})
