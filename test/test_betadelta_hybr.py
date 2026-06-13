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
