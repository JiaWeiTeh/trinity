"""Phase-3: the hybr (betadelta_solver='hybr') root-finder.

Pins arm D's production port (plan §2.3): the pole-free g residual, the
scipy hybr root-finder, and the physical dMdt>0 / valid-structure
acceptance gate that — when it rejects everything the search reaches —
flags ``no_physical_root`` for the runner to hand off on.

The bubble-structure solve is replaced by a synthetic g landscape, so
these tests exercise the solver control flow, not the physics.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import trinity.phase1b_energy_implicit.get_betadelta as GBD
from trinity.bubble_structure.bubble_luminosity import BubbleProperties

# =============================================================================
# Helpers
# =============================================================================


def make_props(dMdt: float = 1.0) -> BubbleProperties:
    arr = np.zeros(3)
    return BubbleProperties(
        bubble_LTotal=1.0,
        bubble_T_r_Tb=1e6,
        bubble_Tavg=1e6,
        bubble_mass=1.0,
        bubble_L1Bubble=0.5,
        bubble_L2Conduction=0.3,
        bubble_L3Intermediate=0.2,
        bubble_v_arr=arr,
        bubble_T_arr=arr,
        bubble_dTdr_arr=arr,
        bubble_r_arr=arr,
        bubble_n_arr=arr,
        bubble_dMdt=dMdt,
        R1=0.1,
        Pb=1.0,
        bubble_r_Tb=0.5,
    )


def make_params(Lmech_total: float = 1.0) -> dict:
    return {"Lmech_total": SimpleNamespace(value=Lmech_total)}


def install_landscape(monkeypatch, gE, gT, dmdt=lambda b, d: 1.0, edot_beta=None):
    """Patch get_residual_pure + get_residual_detailed to a synthetic landscape.

    The g residual the hybr solver sees is (gE(b,d), gT(b,d)); dmdt(b,d) drives
    the acceptance gate (return None -> structure failure / props=None). The
    optional edot_beta lets a test drive Edot_from_beta -> 0 (the legacy f
    pole) while keeping g finite, by construction of the balance term.
    """

    def pure(beta, delta, params, return_bubble_props=False, dMdt_guess=None, **kw):
        dm = dmdt(beta, delta)
        if dm is None:
            return 100.0, 100.0, None  # structure-failure contract
        props = make_props(dMdt=dm) if return_bubble_props else None
        return 0.0, gT(beta, delta), props

    def detailed(beta, delta, params, bubble_props=None):
        Lm = float(params["Lmech_total"].value)
        eb = edot_beta(beta, delta) if edot_beta else gE(beta, delta) * Lm
        # ebal chosen so (eb - ebal)/Lm == gE exactly, for any eb.
        ebal = eb - gE(beta, delta) * Lm
        Edot_residual = (eb - ebal) / eb if abs(eb) > 1e-300 else 0.0
        props = bubble_props if bubble_props is not None else make_props()
        return GBD.ResidualDetails(
            Edot_residual=Edot_residual,
            T_residual=gT(beta, delta),
            Edot_from_beta=eb,
            Edot_from_balance=ebal,
            T_bubble=1e6,
            T0=1e6,
            bubble_props=props,
            L_gain=Lm,
            L_loss=0.0,
        )

    monkeypatch.setattr(GBD, "get_residual_pure", pure)
    monkeypatch.setattr(GBD, "get_residual_detailed", detailed)


def solve(params=None):
    return GBD._solve_betadelta_hybr(0.5, -0.5, params or make_params())


# =============================================================================
# Convergence
# =============================================================================


def test_hybr_converges_to_root(monkeypatch):
    # Root at (0.7, -0.3); linear residual is trivial for hybr.
    install_landscape(monkeypatch, gE=lambda b, d: b - 0.7, gT=lambda b, d: d + 0.3)
    res = solve()
    assert res.converged
    assert not res.no_physical_root
    assert res.beta == pytest.approx(0.7, abs=1e-6)
    assert res.delta == pytest.approx(-0.3, abs=1e-6)
    assert res.total_residual < GBD.RESIDUAL_THRESHOLD


def test_hybr_short_circuits_when_guess_is_root(monkeypatch):
    install_landscape(monkeypatch, gE=lambda b, d: b - 0.5, gT=lambda b, d: d + 0.5)
    res = solve()  # guess (0.5, -0.5) is exactly the root
    assert res.converged
    assert res.iterations == 0  # no root-finder iterations needed


def test_hybr_root_outside_legacy_box_is_accepted(monkeypatch):
    # The whole point of arm D: roots at beta>1 / delta<-1 are reachable.
    install_landscape(monkeypatch, gE=lambda b, d: b - 2.6, gT=lambda b, d: d + 1.5)
    res = solve()
    assert res.converged
    assert res.beta == pytest.approx(2.6, abs=1e-6)
    assert res.delta == pytest.approx(-1.5, abs=1e-6)


# =============================================================================
# Physical acceptance gate -> no_physical_root
# =============================================================================


def test_hybr_no_root_on_structure_failure(monkeypatch):
    install_landscape(
        monkeypatch, gE=lambda b, d: b - 0.7, gT=lambda b, d: d + 0.3, dmdt=lambda b, d: None
    )
    res = solve()
    assert res.no_physical_root
    assert not res.converged
    assert "structure" in res.no_root_reason
    assert res.bubble_properties is None


def test_hybr_no_root_on_negative_dmdt(monkeypatch):
    install_landscape(
        monkeypatch, gE=lambda b, d: b - 0.7, gT=lambda b, d: d + 0.3, dmdt=lambda b, d: -5.0
    )
    res = solve()
    assert res.no_physical_root
    assert "dMdt" in res.no_root_reason


def test_hybr_gate_trips_mid_search(monkeypatch):
    # Guess region is physical; the root sits where dMdt goes negative, so the
    # search walks into the gate and aborts -> no physical root.
    install_landscape(
        monkeypatch,
        gE=lambda b, d: b - 2.0,
        gT=lambda b, d: d + 0.3,
        dmdt=lambda b, d: 1.0 if b < 1.0 else -1.0,
    )
    res = solve()
    assert res.no_physical_root
    assert "dMdt" in res.no_root_reason


# =============================================================================
# g is pole-free where the legacy f metric diverges
# =============================================================================


def test_hybr_converges_through_the_f_pole(monkeypatch):
    # Edot_from_beta -> 0 at the root: the legacy f residual (which divides by
    # it) would blow up, but g (Lmech denominator) stays finite and converges.
    install_landscape(
        monkeypatch,
        gE=lambda b, d: b - 0.7,
        gT=lambda b, d: d + 0.3,
        edot_beta=lambda b, d: 1e-20,
    )
    res = solve()
    assert res.converged
    assert res.beta == pytest.approx(0.7, abs=1e-6)
    assert abs(res.Edot_from_beta) < 1e-12  # the pole the f metric chokes on


# =============================================================================
# Structure-failure rescue ladder (solve_betadelta_pure hybr path)
# =============================================================================


def make_hybr_params(Lmech_total: float = 1.0) -> dict:
    p = make_params(Lmech_total)
    p["betadelta_solver"] = SimpleNamespace(value="hybr")
    return p


def test_rescue_reseeds_hybr_after_structure_failure(monkeypatch):
    # Structure integrable only for beta > 1.5; warm seed (0.5) wanders and
    # aborts. The (patched) legacy grid proposes a seed inside the good
    # region; hybr from there must find the gated root.
    install_landscape(
        monkeypatch,
        gE=lambda b, d: b - 2.0,
        gT=lambda b, d: d + 0.3,
        dmdt=lambda b, d: 1.0 if b > 1.5 else None,
    )
    monkeypatch.setattr(
        GBD, "_solve_betadelta_legacy",
        lambda bg, dg, params, method="grid": GBD.BetaDeltaResult(
            beta=1.8, delta=-0.3, Edot_residual=0.0, T_residual=0.0,
            total_residual=0.0, converged=False, iterations=1,
            bubble_properties=None),
    )
    res = GBD.solve_betadelta_pure(0.5, -0.5, make_hybr_params())
    assert not res.no_physical_root
    assert res.converged
    assert res.beta == pytest.approx(2.0, abs=1e-6)


def test_rescue_skipped_for_condensation_root(monkeypatch):
    # A found dMdt<0 root is real physics (the condensation branch): the
    # ladder must NOT engage -- the legacy solver must never be consulted.
    install_landscape(
        monkeypatch, gE=lambda b, d: b - 0.7, gT=lambda b, d: d + 0.3,
        dmdt=lambda b, d: -5.0,
    )
    def _boom(*a, **kw):
        raise AssertionError("legacy rescue must not run for a condensation root")
    monkeypatch.setattr(GBD, "_solve_betadelta_legacy", _boom)
    res = GBD.solve_betadelta_pure(0.5, -0.5, make_hybr_params())
    assert res.no_physical_root
    assert "dMdt" in res.no_root_reason


def test_rescue_keeps_original_diagnosis_when_retry_fails(monkeypatch):
    # Structure fails everywhere: grid proposes a different seed, hybr fails
    # again -> the ORIGINAL structure-failure no-root is returned.
    install_landscape(
        monkeypatch, gE=lambda b, d: b - 0.7, gT=lambda b, d: d + 0.3,
        dmdt=lambda b, d: None,
    )
    monkeypatch.setattr(
        GBD, "_solve_betadelta_legacy",
        lambda bg, dg, params, method="grid": GBD.BetaDeltaResult(
            beta=1.8, delta=-0.3, Edot_residual=0.0, T_residual=0.0,
            total_residual=0.0, converged=False, iterations=1,
            bubble_properties=None),
    )
    res = GBD.solve_betadelta_pure(0.5, -0.5, make_hybr_params())
    assert res.no_physical_root
    assert "structure" in res.no_root_reason


# =============================================================================
# Energy-balance retry after a structure failure (betadelta_balance_retry; G1_PLAN.md)
# =============================================================================


def make_retry_params(on: bool = True) -> dict:
    p = make_hybr_params()
    p["betadelta_balance_retry"] = SimpleNamespace(value=on)
    return p


def _rescue_finds_nothing(monkeypatch):
    # the legacy grid returns the failing seed itself -> the rescue keeps the original no-root
    monkeypatch.setattr(
        GBD, "_solve_betadelta_legacy",
        lambda bg, dg, params, method="grid": GBD.BetaDeltaResult(
            beta=bg, delta=dg, Edot_residual=0.0, T_residual=0.0, total_residual=0.0,
            converged=False, iterations=1, bubble_properties=None),
    )


def _ramp_landscape(monkeypatch):
    # a steep ramp: structure exists only for beta < -5, the root is at beta = -10; the warm seed (0.5) has none
    install_landscape(monkeypatch, gE=lambda b, d: b + 10.0, gT=lambda b, d: d + 0.3,
                      dmdt=lambda b, d: 1.0 if b < -5.0 else None)


def _plausible_bubble():
    # Code units (Msun, pc, Myr) for a wind bubble at 3 Myr: R2 = 30 pc expanding at 8 km/s, E_b = 1e51 erg,
    # L_mech = 1e38 erg/s with a 2000 km/s mechanical speed, losses 20 % of the gain; R1, Pb consistent via compute_R1_Pb.
    import trinity._functions.unit_conversions as cvt
    L, vm = 1.0e38 * cvt.L_cgs2au, 2.0e3 * cvt.v_kms2au
    p = dict(t_now=3.0, R2=30.0, v2=8.0 * cvt.v_kms2au, Eb=1.0e51 * cvt.E_cgs2au, gamma_adia=5.0 / 3.0,
             Lmech_total=L, v_mech_total=vm, pdot_total=2.0 * L / vm, pdotdot_total=0.1 * (2.0 * L / vm) / 3.0,
             bubble_LTotal=0.2 * L, bubble_Leak=0.0)
    R1, Pb = GBD.compute_R1_Pb(p["R2"], p["Eb"], L, vm, p["gamma_adia"])
    return p, R1, Pb


def test_balance_retry_reaches_a_far_root_after_a_structure_failure(monkeypatch):
    _ramp_landscape(monkeypatch)
    _rescue_finds_nothing(monkeypatch)
    monkeypatch.setattr(GBD, "_beta_from_balance", lambda params: -9.0)
    res = GBD.solve_betadelta_pure(0.5, -0.5, make_retry_params())
    assert not res.no_physical_root and res.converged
    assert res.beta == pytest.approx(-10.0, abs=1e-6)
    assert res.delta == pytest.approx(-0.3, abs=1e-6)


def test_balance_retry_off_keeps_the_structure_failure(monkeypatch):
    _ramp_landscape(monkeypatch)
    _rescue_finds_nothing(monkeypatch)
    monkeypatch.setattr(GBD, "_beta_from_balance", lambda params: -9.0)
    res = GBD.solve_betadelta_pure(0.5, -0.5, make_retry_params(on=False))
    assert res.no_physical_root and "structure" in res.no_root_reason


def test_balance_retry_never_retries_a_condensation_root(monkeypatch):
    install_landscape(monkeypatch, gE=lambda b, d: b - 0.7, gT=lambda b, d: d + 0.3, dmdt=lambda b, d: -5.0)

    def _boom(params):
        raise AssertionError("a found condensation root must not be retried")
    monkeypatch.setattr(GBD, "_beta_from_balance", _boom)
    res = GBD.solve_betadelta_pure(0.5, -0.5, make_retry_params())
    assert res.no_physical_root and "dMdt" in res.no_root_reason


def test_balance_retry_respects_the_rescue_cost_cap(monkeypatch):
    _ramp_landscape(monkeypatch)

    def _boom(params):
        raise AssertionError("no retry when rescue is not allowed")
    monkeypatch.setattr(GBD, "_beta_from_balance", _boom)
    res = GBD.solve_betadelta_pure(0.5, -0.5, make_retry_params(), rescue=False)
    assert res.no_physical_root


def test_balance_retry_rejects_an_unconverged_retry(monkeypatch):
    _ramp_landscape(monkeypatch)
    _rescue_finds_nothing(monkeypatch)
    monkeypatch.setattr(GBD, "_beta_from_balance", lambda params: -9.0)
    unconverged = GBD.BetaDeltaResult(beta=-9.5, delta=-0.3, Edot_residual=0.1, T_residual=0.1,
                                      total_residual=0.2, converged=False, iterations=30, bubble_properties=None)
    first = GBD._solve_hybr_rescued
    monkeypatch.setattr(GBD, "_solve_hybr_rescued",
                        lambda bg, dg, params, method, rescue, wide=True: unconverged if bg == -9.0
                        else first(bg, dg, params, method, rescue, wide))
    res = GBD.solve_betadelta_pure(0.5, -0.5, make_retry_params())
    assert res.no_physical_root and "structure" in res.no_root_reason


def test_balance_retry_survives_a_singular_balance(monkeypatch):
    _ramp_landscape(monkeypatch)
    _rescue_finds_nothing(monkeypatch)

    def _bad(params):
        raise ValueError("solve_R1: no bracket")
    monkeypatch.setattr(GBD, "_beta_from_balance", _bad)
    res = GBD.solve_betadelta_pure(0.5, -0.5, make_retry_params())
    assert res.no_physical_root and "structure" in res.no_root_reason


def test_beta_from_balance_zeroes_the_energy_residual():
    # the real _beta_from_balance (real key names): A12's rate at beta1 equals L_gain - L_loss - 4 pi R2^2 v2 Pb
    p, R1, Pb = _plausible_bubble()
    params = {k: SimpleNamespace(value=v) for k, v in p.items()}
    beta1 = GBD._beta_from_balance(params)
    balance = p["Lmech_total"] - p["bubble_LTotal"] - 4 * np.pi * p["R2"] ** 2 * p["v2"] * Pb
    rate = GBD.cool_beta_to_Ebdot_pure(beta1, Pb, p["t_now"], R1, p["R2"], p["v2"], p["Eb"],
                                       p["pdot_total"], p["pdotdot_total"])
    assert np.isfinite(beta1)
    assert rate == pytest.approx(balance, rel=1e-9)


def test_energy_balance_beta_inverts_a12():
    # Ebdot_to_cool_beta is the exact inverse of A12 (cool_beta_to_Ebdot_pure) across the ramp roots and the usual range
    import trinity.bubble_structure.get_bubbleParams as GBP
    p, R1, Pb = _plausible_bubble()
    for beta in (-48.0, -10.0, 0.8, 2.0):
        Edot = GBD.cool_beta_to_Ebdot_pure(beta, Pb, p["t_now"], R1, p["R2"], p["v2"], p["Eb"],
                                           p["pdot_total"], p["pdotdot_total"])
        back = GBP.Ebdot_to_cool_beta(Pb, R1, Edot, {k: p[k] for k in ("t_now", "pdot_total", "pdotdot_total", "R2", "v2", "Eb")})
        assert back == pytest.approx(beta, rel=1e-9, abs=1e-9)
