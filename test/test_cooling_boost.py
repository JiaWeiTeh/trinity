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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
