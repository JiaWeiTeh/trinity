"""
Tests for the Phase-1b beta-delta grid solver (``get_betadelta``).

Pins the grid-search call-elimination behavior:
 * the grid center (= the input guess, already evaluated by the caller) is
   never re-evaluated inside ``_solve_grid``;
 * the winning point's residual and BubbleProperties captured during the scan
   are reused, so ``solve_betadelta_pure`` performs no redundant
   bubble-structure solves (the former best-point re-evaluation and the
   detailed re-evaluation);
 * fallback semantics are unchanged: when nothing beats the input guess the
   guess is returned, and the best-of-grid selection matches a brute-force
   minimum over the evaluated points.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import trinity.phase1b_energy_implicit.get_betadelta as GBD
from trinity.bubble_structure.bubble_luminosity import BubbleProperties


# =============================================================================
# Helpers
# =============================================================================

def make_props(dMdt: float = 1.0, T_r_Tb: float = 1e6) -> BubbleProperties:
    """Minimal BubbleProperties stand-in with recognizable values."""
    arr = np.zeros(3)
    return BubbleProperties(
        bubble_LTotal=1.0,
        bubble_T_r_Tb=T_r_Tb,
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


def make_params() -> dict:
    """Fake params dict: plain mapping of name -> object with .value.

    Values only need to keep the cheap post-bubble scalar algebra in
    ``get_residual_detailed`` finite; solve_R1's full [0, R2] bracket
    finds the (tiny) R1 root for these values, so nothing here needs
    to be physically self-consistent.
    """
    vals = {
        'R2': 10.0,
        'v2': 5.0,
        'Eb': 1e5,
        'T0': 1e6,
        't_now': 1.0,
        'gamma_adia': 5.0 / 3.0,
        'Lmech_total': 1e4,
        'v_mech_total': 2e3,
        'pdot_total': 10.0,
        'pdotdot_total': 0.1,
    }
    params = {k: SimpleNamespace(value=v) for k, v in vals.items()}
    return params


class ResidualRecorder:
    """Patchable stand-in for get_residual_pure with a synthetic landscape.

    landscape(beta, delta) -> total residual; the recorder splits it as
    Edot_res = sqrt(residual), T_res = 0 so Edot**2 + T**2 reproduces it.

    Each successful call returns props with dMdt = 100 + <call number>, and
    the dMdt_guess seen by every call is recorded, so warm-start threading
    is observable. ``fail_at`` marks (beta, delta) points (1e-9 tolerance)
    that return the failure contract (100, 100, None).
    """

    def __init__(self, landscape, fail_at=()):
        self.landscape = landscape
        self.fail_at = fail_at
        self.calls = []        # (beta, delta) in evaluation order
        self.dmdt_seeds = []   # dMdt_guess kwarg per call, parallel to calls

    def _fails(self, beta, delta):
        return any(abs(beta - b) < 1e-9 and abs(delta - d) < 1e-9
                   for b, d in self.fail_at)

    def __call__(self, beta, delta, params, return_bubble_props=False,
                 dMdt_guess=None, **kwargs):
        self.calls.append((beta, delta))
        self.dmdt_seeds.append(dMdt_guess)
        if self._fails(beta, delta):
            return 100.0, 100.0, None
        total = self.landscape(beta, delta)
        props = make_props(dMdt=100.0 + len(self.calls))
        if return_bubble_props:
            return np.sqrt(total), 0.0, props
        return np.sqrt(total), 0.0, None


def forbid_bubble_solve(monkeypatch):
    """Make any real bubble-structure solve fail the test loudly."""
    def bomb(params):
        raise AssertionError("get_bubbleproperties_pure must not be called")
    monkeypatch.setattr(GBD, "get_bubbleproperties_pure", bomb)


# =============================================================================
# _solve_grid
# =============================================================================

def test_grid_skips_center_point(monkeypatch):
    """The input guess (grid center) is never re-evaluated by the scan."""
    rec = ResidualRecorder(lambda b, d: 1.0 + (b - 0.4) ** 2 + (d + 0.4) ** 2)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    beta_g, delta_g = 0.5, -0.5
    *_, n_evals = GBD._solve_grid(beta_g, delta_g, params=None,
                                  input_residual=2.0, input_props=None)

    assert n_evals == GBD.GRID_SIZE ** 2 - 1 == len(rec.calls)
    for b, d in rec.calls:
        assert not (abs(b - beta_g) < 1e-12 and abs(d - delta_g) < 1e-12)


def test_grid_skip_at_exact_bound(monkeypatch):
    """A guess exactly at a clamped bound coincides with the linspace
    endpoint and is still skipped (it was evaluated by the caller)."""
    rec = ResidualRecorder(lambda b, d: 1.0)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    *_, n_evals = GBD._solve_grid(1.0, 0.0, params=None,
                                  input_residual=2.0, input_props=None)

    assert n_evals == GBD.GRID_SIZE ** 2 - 1
    for b, d in rec.calls:
        assert not (abs(b - 1.0) < 1e-12 and abs(d - 0.0) < 1e-12)


def test_grid_full_scan_when_guess_near_bound(monkeypatch):
    """A guess near (not at) a bound shifts off the clamped grid: no point
    matches it, so the full grid is evaluated."""
    rec = ResidualRecorder(lambda b, d: 1.0)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    *_, n_evals = GBD._solve_grid(0.995, -0.5, params=None,
                                  input_residual=2.0, input_props=None)

    assert n_evals == GBD.GRID_SIZE ** 2


def test_grid_best_matches_bruteforce(monkeypatch):
    """Selected point equals the brute-force minimum over evaluated points."""
    target_b, target_d = 0.487, -0.513  # slightly off-grid
    landscape = lambda b, d: (b - target_b) ** 2 + (d - target_d) ** 2 + 0.01
    rec = ResidualRecorder(landscape)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    best_beta, best_delta, best_props, best_residual, _ = GBD._solve_grid(
        0.5, -0.5, params=None, input_residual=landscape(0.5, -0.5),
        input_props=None)

    expected = min(rec.calls, key=lambda bd: landscape(*bd))
    assert (best_beta, best_delta) == expected
    assert best_residual == pytest.approx(landscape(*expected))
    assert best_props is not None


def test_grid_returns_input_when_no_improvement(monkeypatch):
    """If no grid point beats the seeded input residual, the guess (and its
    props) is returned — the caller's candidate list semantics rely on it."""
    rec = ResidualRecorder(lambda b, d: 10.0)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    sentinel = make_props(dMdt=42.0)
    best_beta, best_delta, best_props, best_residual, _ = GBD._solve_grid(
        0.5, -0.5, params=None, input_residual=0.5, input_props=sentinel)

    assert (best_beta, best_delta) == (0.5, -0.5)
    assert best_props is sentinel
    assert best_residual == 0.5


def test_grid_point_failure_is_skipped(monkeypatch):
    """A raising grid point is skipped; the scan continues (current contract)."""
    def flaky(beta, delta, params, return_bubble_props=False, **kwargs):
        if abs(beta - 0.49) < 1e-9:
            raise RuntimeError("synthetic failure")
        return 1.0, 0.0, make_props()
    monkeypatch.setattr(GBD, "get_residual_pure", flaky)

    best_beta, best_delta, *_ = GBD._solve_grid(
        0.5, -0.5, params=None, input_residual=None, input_props=None)
    assert np.isfinite(best_beta) and np.isfinite(best_delta)


# =============================================================================
# solve_betadelta_pure: no redundant bubble solves
# =============================================================================

def test_solver_performs_no_redundant_evaluations(monkeypatch):
    """One eval for the input + one per non-center grid point, and the final
    detailed result reuses the winner's props instead of re-solving."""
    rec = ResidualRecorder(lambda b, d: 1.0 + (b - 0.49) ** 2)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)
    forbid_bubble_solve(monkeypatch)

    result = GBD.solve_betadelta_pure(0.5, -0.5, make_params())

    assert len(rec.calls) == 1 + GBD.GRID_SIZE ** 2 - 1
    assert result.bubble_properties is not None


def test_solver_converged_input_short_circuits(monkeypatch):
    """A converged input guess returns immediately after a single evaluation."""
    rec = ResidualRecorder(lambda b, d: 0.5 * GBD.RESIDUAL_THRESHOLD)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)
    forbid_bubble_solve(monkeypatch)

    result = GBD.solve_betadelta_pure(0.5, -0.5, make_params())

    assert len(rec.calls) == 1
    assert result.converged is True
    assert (result.beta, result.delta) == (0.5, -0.5)


def test_detailed_reuses_supplied_props(monkeypatch):
    """get_residual_detailed with bubble_props supplied performs no solve and
    computes its diagnostics from the supplied object."""
    forbid_bubble_solve(monkeypatch)
    props = make_props(T_r_Tb=2e6)

    details = GBD.get_residual_detailed(0.5, -0.5, make_params(),
                                        bubble_props=props)

    assert details.T_bubble == 2e6
    assert details.bubble_props is props


# =============================================================================
# dMdt warm-start threading
# =============================================================================

def test_view_dmdt_override():
    """BubbleParamsView returns the dMdt override when given, and falls
    through to params when not."""
    params = {'bubble_dMdt': SimpleNamespace(value=7.0)}

    view = GBD.BubbleParamsView(params, 0.5, -0.5, dMdt_guess=3.0)
    assert view['bubble_dMdt'].value == 3.0

    view_cold = GBD.BubbleParamsView(params, 0.5, -0.5)
    assert view_cold['bubble_dMdt'].value == 7.0


def test_warmstart_threads_dmdt_through_scan(monkeypatch):
    """Each grid point's fsolve is seeded with the previous successful
    point's solved dMdt; the first point is seeded from the input props."""
    rec = ResidualRecorder(lambda b, d: 1.0)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    GBD._solve_grid(0.5, -0.5, params=None,
                    input_residual=2.0, input_props=make_props(dMdt=42.0))

    # First scanned point seeded from input props; call k seeded with the
    # dMdt solved at call k-1 (recorder returns dMdt = 100 + call number).
    assert rec.dmdt_seeds[0] == 42.0
    for k in range(1, len(rec.dmdt_seeds)):
        assert rec.dmdt_seeds[k] == 100.0 + k


def test_warmstart_failed_point_keeps_previous_seed(monkeypatch):
    """A failed point (props=None) must not advance the warm-start seed."""
    # Make the second scanned point fail: the center-out scan visits ring-1
    # neighbours first — (0.49, -0.5) then (0.5, -0.51).
    rec = ResidualRecorder(lambda b, d: 1.0, fail_at=[(0.5, -0.51)])
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    GBD._solve_grid(0.5, -0.5, params=None,
                    input_residual=2.0, input_props=make_props(dMdt=42.0))

    # Call 0 succeeds (dMdt -> 101), call 1 fails, call 2 must still be
    # seeded with 101, after which the chain resumes.
    assert rec.dmdt_seeds[0] == 42.0
    assert rec.dmdt_seeds[1] == 101.0
    assert rec.dmdt_seeds[2] == 101.0
    assert rec.dmdt_seeds[3] == 103.0


def test_warmstart_no_input_props_starts_cold(monkeypatch):
    """Without usable input props the first point runs cold (None seed) and
    threading starts from the first success."""
    rec = ResidualRecorder(lambda b, d: 1.0)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    GBD._solve_grid(0.5, -0.5, params=None,
                    input_residual=None, input_props=None)

    assert rec.dmdt_seeds[0] is None
    assert rec.dmdt_seeds[1] == 101.0


def test_usable_dmdt_guards():
    """Only finite, positive dMdt from a successful solve is usable."""
    assert GBD._usable_dMdt(None) is None
    assert GBD._usable_dMdt(make_props(dMdt=np.nan)) is None
    assert GBD._usable_dMdt(make_props(dMdt=-5.0)) is None
    assert GBD._usable_dMdt(make_props(dMdt=0.0)) is None
    assert GBD._usable_dMdt(make_props(dMdt=13.0)) == 13.0


# =============================================================================
# Center-out scan order and excellent-point early exit
# =============================================================================

def test_scan_order_is_center_out(monkeypatch):
    """The first scanned points are the ring-1 index-space neighbours of the
    (skipped) center, in the documented deterministic order."""
    rec = ResidualRecorder(lambda b, d: 1.0)  # nothing excellent
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    GBD._solve_grid(0.5, -0.5, params=None,
                    input_residual=2.0, input_props=None)

    ring1 = [(0.49, -0.5), (0.5, -0.51), (0.5, -0.49), (0.51, -0.5)]
    assert [(round(b, 6), round(d, 6)) for b, d in rec.calls[:4]] == ring1
    assert len(rec.calls) == GBD.GRID_SIZE ** 2 - 1


def test_no_exit_on_merely_converged_point(monkeypatch):
    """A point below RESIDUAL_THRESHOLD but above the early-exit margin must
    NOT stop the scan: merely-converged picks decay above the acceptance
    threshold within ~a segment, costing a fresh grid search downstream."""
    merely = (0.5, -0.51)  # scanned second; converged but not excellent

    def landscape(b, d):
        if abs(b - merely[0]) < 1e-9 and abs(d - merely[1]) < 1e-9:
            return 0.5 * GBD.RESIDUAL_THRESHOLD
        return 1.0

    rec = ResidualRecorder(landscape)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    best_beta, best_delta, _, _, n_evals = GBD._solve_grid(
        0.5, -0.5, params=None, input_residual=2.0, input_props=None)

    assert n_evals == GBD.GRID_SIZE ** 2 - 1  # full scan
    assert (round(best_beta, 6), round(best_delta, 6)) == merely  # still best


def test_early_exit_on_excellent_point(monkeypatch):
    """The scan stops at the first point below GRID_EARLY_EXIT_RESIDUAL."""
    excellent = (0.5, -0.51)  # second point in the center-out scan

    def landscape(b, d):
        if abs(b - excellent[0]) < 1e-9 and abs(d - excellent[1]) < 1e-9:
            return 0.5 * GBD.GRID_EARLY_EXIT_RESIDUAL
        return 1.0

    rec = ResidualRecorder(landscape)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    best_beta, best_delta, best_props, best_residual, n_evals = GBD._solve_grid(
        0.5, -0.5, params=None, input_residual=2.0, input_props=None)

    assert n_evals == 2
    assert (round(best_beta, 6), round(best_delta, 6)) == excellent
    assert best_residual < GBD.GRID_EARLY_EXIT_RESIDUAL
    assert best_props is not None


def test_no_excellence_scans_all_and_returns_best(monkeypatch):
    """Without any sub-margin point the full grid is evaluated and the global
    best returned — original best-of-grid semantics preserved."""
    target_b, target_d = 0.513, -0.487
    landscape = lambda b, d: 1.0 + (b - target_b) ** 2 + (d - target_d) ** 2
    rec = ResidualRecorder(landscape)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)

    best_beta, best_delta, _, _, n_evals = GBD._solve_grid(
        0.5, -0.5, params=None, input_residual=landscape(0.5, -0.5),
        input_props=None)

    assert n_evals == GBD.GRID_SIZE ** 2 - 1
    expected = min(rec.calls, key=lambda bd: landscape(*bd))
    assert (best_beta, best_delta) == expected


def test_solver_early_exit_end_to_end(monkeypatch):
    """End to end: an excellent ring-1 point means solve_betadelta_pure does
    1 (input) + 2 (scan) evaluations instead of 25, converges, and reuses
    the winning props without re-solving."""
    excellent = (0.5, -0.51)

    def landscape(b, d):
        if abs(b - excellent[0]) < 1e-9 and abs(d - excellent[1]) < 1e-9:
            return 0.5 * GBD.GRID_EARLY_EXIT_RESIDUAL
        return 1.0

    rec = ResidualRecorder(landscape)
    monkeypatch.setattr(GBD, "get_residual_pure", rec)
    forbid_bubble_solve(monkeypatch)

    result = GBD.solve_betadelta_pure(0.5, -0.5, make_params())

    assert len(rec.calls) == 3
    # plain truthiness: on the grid path `converged` is a numpy bool
    # (residuals are numpy scalars), so `is True` would be wrong
    assert result.converged
    assert (round(result.beta, 6), round(result.delta, 6)) == excellent
    assert result.bubble_properties is not None
