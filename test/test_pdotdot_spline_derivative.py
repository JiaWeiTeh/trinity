"""Finding #3 of the magic-number audit (docs/dev/magic-numbers/SWEEP2_PLAN.md §3):
``pdotdot_total`` must be the exact derivative of the ``pdot_total`` cubic
interpolant, and feedback evaluation within 1e-9 Myr of the SPS table edges
must not raise.

Written failing-first against the ``dt = 1e-9`` Myr central difference in
``get_current_sps_feedback``: that step sits ~3 decades into the float-roundoff
regime (measured: docs/dev/magic-numbers/data/pdotdot_percall.csv — median
relative error 6e-6, worst 3e-2 where |pdotdot| is small), and its ``t ± dt``
evaluations fall outside the table when ``t`` is within 1e-9 of either edge,
raising through interp1d despite passing the function's own range check.
"""
import numpy as np
from scipy.interpolate import make_interp_spline

import trinity._functions.unit_conversions as cvt
from trinity._input.registry import _resolve_sps_bundle
from trinity.sps import read_sps
from trinity.sps.update_feedback import get_current_sps_feedback


class _Item:
    def __init__(self, value):
        self.value = value


def _bundled_sps_feedback_params():
    """params carrying real interpolators over the bundled SB99 table."""
    params = {
        "ZCloud": _Item(1.0),
        "SB99_rotation": _Item(1),
        "sps_refmass": _Item("def_value"),
        "FB_mColdWindFrac": _Item(0.0),
        "FB_mColdSNFrac": _Item(0.0),
        "FB_thermCoeffWind": _Item(1.0),
        "FB_thermCoeffSN": _Item(1.0),
        "FB_vSN": _Item(1.0e4 * cvt.v_kms2au),
    }
    params["sps_path"] = _Item(_resolve_sps_bundle("def_path", params))
    sps_data = read_sps.read_sps(1.0, params)
    params["sps_f"] = _Item(read_sps.get_interpolation(sps_data))
    return params, sps_data


def test_pdotdot_matches_exact_spline_derivative():
    params, sps_data = _bundled_sps_feedback_params()
    t_knots, pdot_total = np.asarray(sps_data[0]), np.asarray(sps_data[10])
    # interp1d(kind='cubic') and make_interp_spline(k=3) are the same
    # not-a-knot cubic (verified bit-identical in the finding-#3 study), so
    # this is THE derivative of the interpolant production evaluates.
    exact = make_interp_spline(t_knots, pdot_total, k=3).derivative()

    # Times spanning every phase window a run visits (Myr): early 1a through
    # the SN era; includes 4.3257e-3, the worst-noise point measured.
    ts = np.geomspace(2e-8, 30.0, 200)
    ts = np.append(ts, 4.325750e-3)

    dscale = np.max(np.abs(exact(t_knots)))
    worst = 0.0
    for t in ts:
        got = get_current_sps_feedback(float(t), params).pdotdot_total
        ref = float(exact(t))
        rel = abs(got - ref) / max(abs(ref), 1e-6 * dscale)
        worst = max(worst, rel)
    assert worst <= 1e-9, (
        f"pdotdot_total deviates from the exact interpolant derivative by "
        f"{worst:.3e} (rel) — finite-difference noise, audit finding #3"
    )


def test_feedback_near_table_edges_does_not_raise():
    params, sps_data = _bundled_sps_feedback_params()
    t_knots = np.asarray(sps_data[0])
    for t in (float(t_knots[0]) + 5e-10, float(t_knots[-1]) - 5e-10):
        fb = get_current_sps_feedback(t, params)  # must not raise
        assert np.isfinite(fb.pdotdot_total)
