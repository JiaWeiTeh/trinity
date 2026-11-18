"""Truth table for the opt-in unresolved-interface-cooling closure (Paper-II note).

`effective_Lloss` is the single helper that feeds the beta-delta residual, the energy ODE
(`Edot_from_balance`), and the energy->momentum trigger. The load-bearing property is that the
DEFAULT mode ('none') returns the resolved loss UNCHANGED, so a default run is byte-identical; the
two boost modes add only the missing part and never double-count.

See docs/dev/transition/pdv-trigger/PLAN.md and the methods note
"Adding unresolved interface cooling to TRINITY without double-counting".
"""
import pytest

from trinity.phase1b_energy_implicit.get_betadelta import effective_Lloss


def test_none_is_resolved_loss_unchanged():
    # the byte-identity contract: 'none' == Lcool + Lleak, exactly
    assert effective_Lloss("none", 99.0, 0.99, Lcool=3.0, Lleak=0.5, Lmech=10.0) == 3.5
    # leak-free (the regime of every screened config) -> just Lcool
    assert effective_Lloss("none", 2.0, 0.9, Lcool=3.0, Lleak=0.0, Lmech=10.0) == 3.0


def test_unknown_mode_falls_back_to_resolved():
    # defensive: an unrecognised token must not perturb a run
    assert effective_Lloss("typo", 5.0, 0.9, Lcool=3.0, Lleak=0.5, Lmech=10.0) == 3.5


def test_multiplier_boosts_only_cool_not_leak():
    # Lloss_eff = Lleak + f_mix * Lcool   (note Eq. for the f_mix closure)
    assert effective_Lloss("multiplier", 2.0, 0.0, Lcool=3.0, Lleak=0.5, Lmech=10.0) == 0.5 + 6.0
    # f_mix = 1 is a no-op equal to resolved
    assert effective_Lloss("multiplier", 1.0, 0.0, Lcool=3.0, Lleak=0.5, Lmech=10.0) == 3.5


def test_theta_target_tops_up_to_target():
    # resolved (3+0.5=3.5) below target (0.9*10=9.0) -> top up to the target
    assert effective_Lloss("theta_target", 1.0, 0.9, Lcool=3.0, Lleak=0.5, Lmech=10.0) == 9.0


def test_theta_target_switches_off_when_resolved_exceeds_target():
    # resolved (8+0.5=8.5) already above target (0.3*10=3.0) -> correction OFF, stays resolved
    assert effective_Lloss("theta_target", 1.0, 0.3, Lcool=8.0, Lleak=0.5, Lmech=10.0) == 8.5


def test_theta_target_is_double_count_free():
    # the counted loss is max(resolved, theta*Lmech), NEVER resolved + theta*Lmech
    Lcool, Lleak, Lmech, theta = 3.0, 0.5, 10.0, 0.9
    eff = effective_Lloss("theta_target", 1.0, theta, Lcool=Lcool, Lleak=Lleak, Lmech=Lmech)
    assert eff == max(Lcool + Lleak, theta * Lmech)
    assert eff < (Lcool + Lleak) + theta * Lmech  # strictly below the double-count sink


# --- F2: the phase-1b cooling_balance trigger (run_energy_implicit_phase.trigger_Lloss) ---------
class _P:
    """Minimal stand-in for a params entry (only .value is read)."""

    def __init__(self, value):
        self.value = value


class _Props:
    """Minimal stand-in for BubbleProperties (only bubble_LTotal is read)."""

    def __init__(self, LTotal):
        self.bubble_LTotal = LTotal


def _params(mode, fmix, Lloss_stored, Leak, theta=0.0):
    return {'cooling_boost_mode': _P(mode), 'cooling_boost_fmix': _P(fmix),
            'cooling_boost_theta': _P(theta), 'bubble_Lloss': _P(Lloss_stored),
            'bubble_Leak': _P(Leak)}


def test_trigger_no_root_segment_holds_stored_loss_without_reboost():
    # F2: bubble_Lloss is stored already boosted (f*Lcool + Lleak = 4*3 + 0.5 = 12.5). A no-root
    # segment (bubble_props None) must use it as is, not 4*12.5 = f^2*Lcool + f*Lleak = 50.
    from trinity.phase1b_energy_implicit.run_energy_implicit_phase import trigger_Lloss
    stored = effective_Lloss("multiplier", 4.0, 0.0, Lcool=3.0, Lleak=0.5, Lmech=10.0)
    assert stored == 12.5
    assert trigger_Lloss(_params("multiplier", 4.0, stored, 0.5), None, 10.0) == 12.5


def test_trigger_with_props_boosts_once():
    from trinity.phase1b_energy_implicit.run_energy_implicit_phase import trigger_Lloss
    p = _params("multiplier", 4.0, float("nan"), 0.5)
    assert trigger_Lloss(p, _Props(3.0), 10.0) == 0.5 + 4.0 * 3.0


def test_trigger_default_mode_unchanged():
    # 'none': with props -> Lcool + Lleak; without -> the stored value, exactly as before the fix
    from trinity.phase1b_energy_implicit.run_energy_implicit_phase import trigger_Lloss
    assert trigger_Lloss(_params("none", 1.0, 7.25, 0.5), _Props(3.0), 10.0) == 3.5
    assert trigger_Lloss(_params("none", 1.0, 7.25, 0.5), None, 10.0) == 7.25


def test_trigger_no_root_theta_target_follows_current_lmech():
    # theta_target's max is idempotent, so the no-root segment keeps re-applying it: the target
    # follows the current Lgain (stored = max(3 + 0.5, 0.5 * 10) = 5.0)
    from trinity.phase1b_energy_implicit.run_energy_implicit_phase import trigger_Lloss
    stored = effective_Lloss("theta_target", 1.0, 0.5, Lcool=3.0, Lleak=0.5, Lmech=10.0)
    assert stored == 5.0
    assert trigger_Lloss(_params("theta_target", 1.0, stored, 0.5, theta=0.5), None, 20.0) == 10.0
    assert trigger_Lloss(_params("theta_target", 1.0, stored, 0.5, theta=0.5), None, 4.0) == 5.0



# --- F3 (docs/dev/cooling-enhance-suppress/PLAN.md): the cooling-boost knobs are validated ---
from trinity._input.errors import ParameterFileError  # noqa: E402
from trinity._input.registry import (  # noqa: E402
    REGISTRY,
    _validate_C_thermal,
    _validate_cooling_boost_fmix,
    _validate_cooling_boost_kappa,
    _validate_cooling_boost_mode,
    _validate_cooling_boost_theta,
)

_F3 = {
    "cooling_boost_mode": (_validate_cooling_boost_mode, [None, "none", "multiplier", "theta_target"],
                           ["multiplyer", "Multiplier", "", 4.0]),
    "cooling_boost_fmix": (_validate_cooling_boost_fmix, [1.0, 1, 2.5, 4.0], [0.0, 0.5, -2.0, True, "4", float("nan")]),
    "cooling_boost_theta": (_validate_cooling_boost_theta, [0.0, 0.5, 0.96], [1.0, -0.1, 1.5, False, float("inf")]),
    "cooling_boost_kappa": (_validate_cooling_boost_kappa, [1.0, 0.05, 0.2, 16, "auto", "AUTO"], [0.0, -1.0, "x", True]),
    "C_thermal": (_validate_C_thermal, [6e-7, 1.2e-6], [0.0, -6e-7, None, float("nan")]),
}


@pytest.mark.parametrize("name", sorted(_F3))
def test_f3_validator_is_attached(name):
    assert REGISTRY[name].validator is _F3[name][0]


@pytest.mark.parametrize("name,value", [(n, v) for n, (_, ok, _) in sorted(_F3.items()) for v in ok])
def test_f3_validator_accepts(name, value):
    _F3[name][0](value, {})  # must not raise


@pytest.mark.parametrize("name,value", [(n, v) for n, (_, _, bad) in sorted(_F3.items()) for v in bad])
def test_f3_validator_refuses(name, value):
    with pytest.raises(ParameterFileError):
        _F3[name][0](value, {})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
