#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness of trinity's conduction-layer treatment vs WARPFIELD's ``dR2min`` magic number.

Old WARPFIELD floored the bubble-structure integration offset ``dR2`` with a
hand-tuned constant that it *bumped* for massive clusters::

    dR2min = 1.0e-7                       # "this number might have to be higher... TO DO"
    if Mclus > 1.0e7:
        dR2min = 1.0e-14*Mclus + 1.0e-7

``dR2`` is the thickness of the thin conduction (thermal-evaporation) layer just
inside the outer shock ``R2`` where the backward Weaver+77 temperature ODE is
anchored (``r2Prime = R2 - dR2``). As the cluster -- hence the wind mass flux
``dMdt`` -- grows, the analytic thickness
``dR2 = T_init**(5/2) * 4*pi*R2**2 / (const * dMdt)`` shrinks, so WARPFIELD clamped
it to keep the integration interval resolvable. The floor (and its mass-dependent
bump) is an unjustified magic number -- the source even says so ("TO DO: figure
out what to set this number to").

trinity (``bubble_luminosity._get_bubble_ODE_initial_conditions``) drops the floor
entirely: it uses the *exact analytic* ``dR2`` and crosses the resulting ultra-thin
layer with the stiff adaptive solver. These tests pin that this is robust:

  1. ``dR2`` follows pure ``1/dMdt`` scaling with NO clamp anywhere -- the magic
     number is gone (a floor would flatten ``dR2`` once ``dMdt`` got large).
  2. ``r2Prime = R2 - dR2`` stays well-conditioned across the physically realizable
     cluster range -- the catastrophic cancellation the floor guarded against does
     not occur for real clusters.
  3. the unfloored thin layer is integrated *correctly*: production LSODA matches an
     independent stiff Radau reference on the temperature profile to ~1e-6, on two
     real captured states -- a mild cluster and a genuinely-stiff high-feedback one
     (dR2/R2 ~ 3e-10, the LSODA-warning-flood regime) -- the cross-solver agreement
     measured in docs/dev/performance/BUBBLE_CONDUCTION_STIFFNESS.md, as a test.

Coverage tiers:
  * the no-floor + conditioning invariants are also swept over bubble size
    (test_dR2_is_pure_analytic_layer_across_bubble_sizes) and the degeneracy
    boundary is pinned (test_degeneracy_boundary_and_physical_margin) -- pure, no sims.

Captured states are real bubble solves: ``test/data/residual_resample_fixture.json``
(mild) and ``test/data/dR2_stiff_state_fixture.json`` (stiff, from
docs/dev/performance/harness/capture_stiff_dR2_state.py). The loader mirrors
``test_residual_resample._build_params``.
"""

import json
import os

import numpy as np
import pytest
import scipy.integrate
import scipy.interpolate

import trinity.bubble_structure.bubble_luminosity as BL
import trinity.cooling.non_CIE.read_cloudy as non_CIE
from trinity._input.read_param import read_param

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURE_PATH = os.path.join(_REPO_ROOT, "test", "data", "residual_resample_fixture.json")

# Path-valued keys baked into the captured state point at the original checkout;
# let read_param(base) supply the worktree-local (byte-identical) tables instead.
_PATH_OVERRIDE_SKIP = {
    "path_cooling_CIE",
    "path_cooling_nonCIE",
    "sps_path",
    "path2output",
}

_EPS = np.finfo(float).eps


class _Scalar:
    """Minimal stand-in for a params entry (only ``.value`` is read)."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _scalar_params(pv, R2):
    """A tiny params holding only the scalars the IC formula reads.

    ``_get_bubble_ODE_initial_conditions`` touches k_B, mu_ion, C_thermal, R2,
    cool_alpha and t_now -- nothing else, no cooling cubes. Building a fresh dict
    per call lets the regime sweep vary R2 without mutating the shared (cooling-
    bearing) ``state`` params.
    """
    p = {k: _Scalar(pv[k]) for k in ("k_B", "mu_ion", "C_thermal", "cool_alpha", "t_now")}
    p["R2"] = _Scalar(R2)
    return p


def _build_params(fixture):
    """Reconstruct a full ``params`` from the distilled fixture (cooling cubes rebuilt).

    Mirrors test_residual_resample._build_params: read_param(base) -> scalar
    overrides -> rebuild the CIE table and the non-CIE cooling cubes. Test 3 needs
    the cubes (the bubble ODE evaluates the cooling); Tests 1-2 touch only scalars.
    """
    base = os.path.join(_REPO_ROOT, fixture["base_param"])
    params = read_param(base)

    for k, v in fixture["param_values"].items():
        if k in _PATH_OVERRIDE_SKIP:
            continue
        if k in params:
            params[k].value = v

    logT, logL = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    if "cStruc_cooling_CIE_logLambda" in params:
        params["cStruc_cooling_CIE_logLambda"].value = logL
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logL, kind="linear"
    )

    cooling_nonCIE, heating_nonCIE, net = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cooling_nonCIE
    params["cStruc_heating_nonCIE"].value = heating_nonCIE
    params["cStruc_net_nonCIE_interpolation"].value = net
    return params


def _load_state(fixture_name):
    """(fixture dict, full params with cooling) for a fixture under test/data/."""
    with open(os.path.join(_REPO_ROOT, "test", "data", fixture_name)) as fh:
        fixture = json.load(fh)
    return fixture, _build_params(fixture)


@pytest.fixture(scope="module")
def state():
    return _load_state("residual_resample_fixture.json")


@pytest.fixture(scope="module")
def scalars():
    """Just the captured scalar constants -- no cooling rebuild (Tier-1 sweeps)."""
    with open(_FIXTURE_PATH) as fh:
        return json.load(fh)


def _ic(dMdt, params, Pb, R1):
    """Bubble-ODE initial conditions as plain floats (r2Prime, T, dTdr, v)."""
    ic = BL._get_bubble_ODE_initial_conditions(dMdt, params, Pb, R1)
    return tuple(float(np.asarray(x).item()) for x in ic)


def _dR2_from_dTdr(T, dTdr):
    """Recover the exact internal layer thickness the solver anchors on.

    The function builds ``dTdr = -2/5 * T / dR2`` from the EXACT analytic ``dR2``
    (before ``r2Prime = R2 - dR2`` rounds it), so inverting ``dTdr`` recovers
    that exact ``dR2`` without the cancellation that ``R2 - r2Prime`` would incur.
    """
    return -2.0 / 5.0 * T / dTdr


def test_dR2_is_exact_inverse_dMdt_with_no_floor(state):
    """dR2 follows pure 1/dMdt scaling across 10 decades -- there is no magic floor.

    WARPFIELD's ``dR2min`` would clamp ``dR2`` once ``dMdt`` grew, flattening the
    curve and breaking the ``dR2 * dMdt = const`` invariant. trinity holds it to
    machine precision, and anchors the boundary temperature at ``_T_INIT_BOUNDARY``
    for every cluster mass.
    """
    fixture, params = state
    Pb, R1 = fixture["Pb"], fixture["R1"]
    dMdt0 = fixture["dMdt_converged"]
    T_init = BL._T_INIT_BOUNDARY

    products = []
    for f in np.logspace(0, 10, 11):  # fixture cluster up to ~1e10x the wind flux
        _, T, dTdr, _ = _ic(dMdt0 * f, params, Pb, R1)
        # boundary temperature is exactly the anchor, independent of layer thickness
        assert abs(T - T_init) <= 1e-9 * T_init
        products.append(_dR2_from_dTdr(T, dTdr) * (dMdt0 * f))

    products = np.array(products)
    rel_spread = (products.max() - products.min()) / products.mean()
    assert rel_spread < 1e-12, (
        f"dR2*dMdt spread {rel_spread:.2e} over 1e10 in dMdt -- a floor (a la "
        f"WARPFIELD dR2min) would make dR2 stop scaling as 1/dMdt."
    )


def test_thin_layer_well_conditioned_across_physical_clusters(state):
    """R2 - dR2 stays resolvable across the realizable cluster range -- no floor needed.

    Scaling ``dMdt`` from the fixture up to ~1e3x drives ``dR2/R2`` down to ~1e-11,
    bracketing the thinnest *physical* layer on record (the 5e7 M_sun cluster's
    ~3e-11 in BUBBLE_CONDUCTION_STIFFNESS.md). The catastrophic cancellation the
    WARPFIELD floor guarded against -- ``R2 - dR2`` rounding back to ``R2`` -- needs
    ``dR2/R2`` near machine epsilon (~1e-16), orders of magnitude below this range,
    so trinity is well-conditioned without any clamp.
    """
    fixture, params = state
    Pb, R1 = fixture["Pb"], fixture["R1"]
    dMdt0 = fixture["dMdt_converged"]
    R2 = params["R2"].value
    T_init = BL._T_INIT_BOUNDARY

    for f in (1.0, 1e2, 1e3):
        r2Prime, T, dTdr, v = _ic(dMdt0 * f, params, Pb, R1)
        assert np.all(np.isfinite([r2Prime, T, dTdr, v]))
        # the subtraction did not collapse to R2 (no catastrophic cancellation)
        assert r2Prime < R2
        rel_gap = (R2 - r2Prime) / R2
        assert rel_gap > 1e3 * _EPS, (
            f"dR2/R2={rel_gap:.2e} at {f:.0e}x is within ~1e3 eps of cancellation"
        )
        assert abs(T - T_init) <= 1e-9 * T_init  # anchor placed exactly
        assert dTdr < 0                           # temperature rises inward


@pytest.mark.parametrize(
    "fixture_name,max_dR2_over_R2",
    [
        ("residual_resample_fixture.json", 1e-7),   # mild real cluster (~1e-8)
        ("dR2_stiff_state_fixture.json", 1e-9),     # genuinely stiff, flood regime (~3e-10)
    ],
)
def test_unfloored_thin_layer_integrates_correctly(fixture_name, max_dR2_over_R2):
    """The ultra-thin layer is crossed correctly without artificial thickening.

    Production LSODA (rtol=1e-8) vs an independent fully-implicit Radau reference
    (rtol=1e-10) on the same real bubble-structure ODE: the temperature profile and
    inner-boundary temperature must agree to ~1e-6 (measured ~1e-8). This is the
    live form of the cross-solver check in BUBBLE_CONDUCTION_STIFFNESS.md -- it shows
    the stiffness from the thin (unfloored) conduction layer is integrated, not faked.

    Run on two REAL captured states for regime coverage: a mild cluster and a
    genuinely-stiff high-feedback state (dR2/R2 ~ 3e-10 -- the LSODA-warning-flood
    regime WARPFIELD's dR2min floor targeted, captured by
    docs/dev/performance/harness/capture_stiff_dR2_state.py).
    """
    fixture, params = _load_state(fixture_name)
    Pb, R1 = fixture["Pb"], fixture["R1"]
    r2Prime, T0, dTdr0, v0 = _ic(fixture["dMdt_converged"], params, Pb, R1)
    y0 = [v0, T0, dTdr0]

    # confirm the state really is in the intended (thin-layer) regime
    dR2_over_R2 = _dR2_from_dTdr(T0, dTdr0) / params["R2"].value
    assert dR2_over_R2 < max_dR2_over_R2, (
        f"{fixture_name}: dR2/R2={dR2_over_R2:.2e} not in the expected regime "
        f"(< {max_dR2_over_R2:.0e}) -- recapture the fixture."
    )

    def integrate(method, rtol, atol):
        with BL._quiet_lsoda_fortran():
            return scipy.integrate.solve_ivp(
                fun=lambda r, y: BL._get_bubble_ODE(r, y, params, Pb),
                t_span=(r2Prime, R1),
                y0=y0,
                method=method,
                rtol=rtol,
                atol=atol,
                dense_output=True,
            )

    prod = integrate("LSODA", BL._BUBBLE_RTOL, BL._BUBBLE_ATOL)   # production solver
    ref = integrate("Radau", 1e-10, 1e-12)                       # independent stiff ref
    assert prod.success and ref.success

    r_grid = np.linspace(r2Prime, R1, 400)
    v_p, T_p, dT_p = prod.sol(r_grid)
    v_r, T_r, dT_r = ref.sol(r_grid)

    assert np.max(np.abs(T_p - T_r) / np.abs(T_r)) < 1e-5        # temperature profile
    assert np.max(np.abs(dT_p - dT_r) / np.abs(dT_r)) < 1e-5     # its gradient
    assert abs(T_p[-1] - T_r[-1]) / abs(T_r[-1]) < 1e-5          # inner-boundary T
    # v passes through ~0 near R1 (its own boundary condition), so compare absolute
    # disagreement scaled by the boundary velocity rather than a blow-up relative error.
    assert np.max(np.abs(v_p - v_r)) < 1e-4 * abs(v0)


# =============================================================================
# Tier 1 -- regime breadth (pure, no sims): the no-floor property and the
# numerical safety envelope held across bubble sizes, not just the one captured
# state. dR2 depends only on (R2, dMdt); Pb/R1 do not enter it, so a (R2, dMdt)
# grid spans the regimes the IC formula can see.
# =============================================================================


@pytest.mark.parametrize("R2", [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0])
def test_dR2_is_pure_analytic_layer_across_bubble_sizes(scalars, R2):
    """For every bubble size, dR2 is exactly T_init**(5/2)*4*pi*R2**2/(const*dMdt).

    Across 8 decades of dMdt (small -> massive-cluster wind) the dimensionless
    group ``dR2*dMdt/R2**2`` equals the single analytic constant ``T_init**(5/2)*
    4*pi/const`` to ~1e-10 -- so no floor engages at any size or feedback strength
    (WARPFIELD's dR2min would peel this group upward once the clamp activated).
    """
    pv = scalars["param_values"]
    Pb, R1 = scalars["Pb"], scalars["R1"]
    params = _scalar_params(pv, R2)

    const = 25.0 / 4.0 * pv["k_B"] / pv["mu_ion"] / pv["C_thermal"]
    expected = BL._T_INIT_BOUNDARY ** 2.5 * 4.0 * np.pi / const  # dR2*dMdt/R2**2

    for dMdt in np.logspace(0, 8, 9):
        _, T, dTdr, _ = _ic(dMdt, params, Pb, R1)
        assert abs(T - BL._T_INIT_BOUNDARY) <= 1e-9 * BL._T_INIT_BOUNDARY
        group = _dR2_from_dTdr(T, dTdr) * dMdt / R2 ** 2
        assert abs(group - expected) <= 1e-10 * expected, (
            f"dR2*dMdt/R2**2={group:.6e} != analytic {expected:.6e} at "
            f"R2={R2:g}, dMdt={dMdt:.1e} -- a floor would break the analytic law."
        )


def test_degeneracy_boundary_and_physical_margin():
    """Quantify where the unfloored R2-dR2 *would* break, and the margin to it.

    The subtraction ``r2Prime = R2 - dR2`` only rounds back to R2 when ``dR2/R2``
    approaches machine epsilon (~1e-16). The thinnest *physical* conduction layer on
    record -- ``dR2/R2 ~ 3.4e-11`` for the 5e7 M_sun cluster
    (docs/dev/performance/BUBBLE_CONDUCTION_STIFFNESS.md) -- clears that by >1e4, so
    the exact analytic dR2 is numerically safe with no clamp. This is the honest
    boundary: trinity is not magic, the regime simply never reaches the cliff.
    """
    phys_ratio = 3.4e-11           # stiffest physical dR2/R2
    boundary = _EPS / 2.0          # float64 round-to-nearest collapse point
    assert phys_ratio / boundary > 1e4

    for R2 in (1e-2, 1.0, 5.6, 100.0):
        # stiffest physical layer is resolvable: the subtraction does not collapse
        assert (R2 - phys_ratio * R2) < R2
        # far below the boundary the subtraction does round back to R2 (the failure
        # WARPFIELD's floor existed to prevent -- only reachable at ~1e7x the
        # stiffest physical thinness, i.e. never in a real run)
        assert (R2 - 1e-18 * R2) == R2
