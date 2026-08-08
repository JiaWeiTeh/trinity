"""Weak-winds sensitivity study (docs/dev/weak-winds): scaling contract + smoke.

``FB_thermCoeffWind`` scales the wind terminal velocity by sqrt(coeff) in the
read_sps wind corrections, so with FB_mColdWindFrac = 0 the loaded feedback
tables obey an exact contract:

    Lmech_W  -> coeff       * Lmech_W(1)
    pdot_W   -> sqrt(coeff) * pdot_W(1)
    Mdot_W   (= pdot_W^2 / 2 Lmech_W)  invariant
    SN and radiative (Qi/Li/Ln/Lbol) channels untouched.

These tests pin that contract on the study's coefficient ladder and prove the
downstream feedback pipeline (update_feedback) and free-streaming initial
conditions (get_y0) stay finite and scale as derived — the failure mode a
strict winds-off switch would hit (0/0 in v_mech_total) must stay out of reach
for every ladder value. The stress test boots two real runs (control vs weak)
end-to-end and checks the weak-wind shell actually launches slower.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
from trinity._input.registry import _resolve_sps_bundle
from trinity.phase0_init import get_InitPhaseParam
from trinity.sps import read_sps
from trinity.sps.update_feedback import get_current_sps_feedback

REPO_ROOT = Path(__file__).resolve().parents[1]

# The study's ladder (docs/dev/weak-winds/harness/weak_winds_sweep.param);
# 1.0 is the control every ratio below is measured against.
LADDER = (0.3, 0.1, 0.03, 0.01)


class _Item:
    def __init__(self, value):
        self.value = value


def _sps_params(therm_coeff_wind=1.0):
    """Minimal params for read_sps against the bundled SB99 default table."""
    params = {
        "ZCloud": _Item(1.0),
        "SB99_rotation": _Item(1),
        "sps_refmass": _Item("def_value"),
        "FB_mColdWindFrac": _Item(0.0),
        "FB_mColdSNFrac": _Item(0.0),
        "FB_thermCoeffWind": _Item(therm_coeff_wind),
        "FB_thermCoeffSN": _Item(1.0),
        "FB_vSN": _Item(1.0e4 * cvt.v_kms2au),
    }
    params["sps_path"] = _Item(_resolve_sps_bundle("def_path", params))
    return params


def _sps_arrays(therm_coeff_wind=1.0):
    labels = (
        "t",
        "Qi",
        "Li",
        "Ln",
        "Lbol",
        "Lmech_W",
        "Lmech_SN",
        "Lmech_total",
        "pdot_W",
        "pdot_SN",
        "pdot_total",
    )
    data = read_sps.read_sps(1.0, _sps_params(therm_coeff_wind))
    return dict(zip(labels, data))


@pytest.fixture(scope="module")
def control():
    return _sps_arrays(1.0)


# =============================================================================
# 1. The scaling contract on the loaded tables
# =============================================================================


@pytest.mark.parametrize("coeff", LADDER)
def test_thermcoeff_scaling_contract(coeff, control):
    weak = _sps_arrays(coeff)

    # Time axis is untouched bitwise (non-log column; deterministic parse+scale).
    assert np.array_equal(weak["t"], control["t"])

    # Non-wind channels are untouched up to 1-ULP loader jitter: the SB99 log
    # columns go through 10**x (sps_columns.convert_to_canonical_au), and
    # numpy's pow can differ by 1 ULP between loads depending on buffer
    # alignment (SIMD vs scalar peel lanes). Lmech_SN inherits it through the
    # Lmech_total - Lmech_W subtraction, where a 1-ULP wobble can even flip a
    # clamped-to-zero row to +1 ULP (measured 2026-08-08: ~45/800 rows, all
    # exactly 1 ULP). A real leak of the wind knob would scale with (1-coeff)
    # — many orders above this atol — so the tolerance loses no power.
    for label in ("Qi", "Li", "Ln", "Lbol", "Lmech_SN", "pdot_SN"):
        atol = 1e-12 * float(np.max(np.abs(control[label])))
        assert np.allclose(weak[label], control[label], rtol=1e-12, atol=atol), (
            f"{label} changed under FB_thermCoeffWind={coeff} — the knob must "
            "touch only the wind channel"
        )

    # Wind channel: Lmech_W ∝ coeff, pdot_W ∝ sqrt(coeff).
    assert np.allclose(weak["Lmech_W"], coeff * control["Lmech_W"], rtol=1e-12)
    assert np.allclose(weak["pdot_W"], np.sqrt(coeff) * control["pdot_W"], rtol=1e-12)

    # Implied mass-loss rate Mdot = pdot^2 / (2 L) is invariant: the knob
    # models unthermalized colliding winds (velocity loss), not less mass.
    live = control["Lmech_W"] > 0
    mdot_control = control["pdot_W"][live] ** 2 / (2 * control["Lmech_W"][live])
    mdot_weak = weak["pdot_W"][live] ** 2 / (2 * weak["Lmech_W"][live])
    assert np.allclose(mdot_weak, mdot_control, rtol=1e-12)

    # Totals stay the sum of their parts.
    assert np.allclose(weak["Lmech_total"], weak["Lmech_W"] + weak["Lmech_SN"], rtol=1e-12)
    assert np.allclose(weak["pdot_total"], weak["pdot_W"] + weak["pdot_SN"], rtol=1e-12)


# =============================================================================
# 2. The feedback pipeline stays finite everywhere on the ladder
# =============================================================================


@pytest.mark.parametrize("coeff", LADDER)
def test_feedback_finite_and_scaled_on_ladder(coeff, control):
    sps_f = read_sps.get_interpolation(read_sps.read_sps(1.0, _sps_params(coeff)))
    params = {"sps_f": _Item(sps_f)}
    sps_f_control = read_sps.get_interpolation(read_sps.read_sps(1.0, _sps_params(1.0)))
    params_control = {"sps_f": _Item(sps_f_control)}

    # Probe both eras: wind-only (before the first SN in the table) and
    # wind+SN. Find the SN onset from the table itself so a swapped SPS
    # bundle moves the probes instead of invalidating them. Threshold on the
    # SN *fraction*, not >0: late-time rows carry 1-ULP subtraction noise in
    # the derived SN channel (see test_thermcoeff_scaling_contract).
    t_arr = control["t"]
    frac_SN = control["pdot_SN"] / control["pdot_total"]
    sn_on = t_arr[frac_SN > 1e-3]
    assert sn_on.size > 0, "default SPS table has no SN era to probe"
    t_wind_only = 0.5 * sn_on[0]
    t_with_SN = min(sn_on[0] + 1.0, float(t_arr[-1]) - 0.1)

    for t in (t_wind_only, t_with_SN):
        fb = get_current_sps_feedback(t, params)
        values = list(fb)
        assert np.all(
            np.isfinite(values)
        ), f"non-finite feedback at t={t} Myr, coeff={coeff}: {values}"
        assert fb.v_mech_total > 0
        assert fb.pdot_total > 0

    # Wind-only era: totals are pure wind, so v_mech_total = 2L/pdot must
    # scale as sqrt(coeff) relative to the control. Not exactly: the cubic
    # interpolators are global (not-a-knot), so the post-SN knots — which do
    # NOT scale with coeff — leak into the wind-only region with geometric
    # decay (measured ~2e-10 relative at coeff=0.01). rel=1e-6 still cleanly
    # separates sqrt(c)=0.1 from c=0.01 or 1.
    fb_weak = get_current_sps_feedback(t_wind_only, params)
    fb_ctrl = get_current_sps_feedback(t_wind_only, params_control)
    assert fb_weak.v_mech_total == pytest.approx(np.sqrt(coeff) * fb_ctrl.v_mech_total, rel=1e-6)
    # Radiation channel untouched.
    assert fb_weak.Qi == pytest.approx(fb_ctrl.Qi, rel=1e-12)
    assert fb_weak.Lbol == pytest.approx(fb_ctrl.Lbol, rel=1e-12)


# =============================================================================
# 3. Free-streaming initial conditions: finite, and scaling as derived
# =============================================================================


def _y0(coeff):
    sps_f = read_sps.get_interpolation(read_sps.read_sps(1.0, _sps_params(coeff)))
    params = {
        "tSF": _Item(0.0),
        "sps_f": _Item(sps_f),
        # Baseline-cloud core density (1e5 cm^-3, the simple_cluster default)
        # and mu_H = 1.4 m_H, both in code units as read_param loads them.
        "nCore": _Item(1e5 * cvt.ndens_cgs2au),
        "mu_convert": _Item(1.4 * cvt.CGS.m_H * cvt.g2Msun),
        "bubble_xi_Tb": _Item(0.98),
    }
    return get_InitPhaseParam.get_y0(params)


def test_get_y0_free_streaming_scalings():
    """v0 ∝ √c; dt_fs ∝ c^-3/4; r0 ∝ c^-1/4; E0 ∝ c^1/4 — all finite.

    Derivation from get_InitPhaseParam formulas with Lw ∝ c, pdot_w ∝ √c:
    Mdot0 invariant, v0 = 2Lw/pdot_w ∝ √c, dt = sqrt(3 Mdot0/(4π rho v0^3))
    ∝ c^-3/4, r0 = v0·dt ∝ c^-1/4, E0 = (5/11)·Lw·dt ∝ c^1/4. A weaker wind
    free-streams LONGER but from a LOWER-energy start — the ICs shift, they
    do not degenerate (that only happens at c = 0, the strict-off limit).
    """
    t0_1, r0_1, v0_1, E0_1, T0_1 = _y0(1.0)

    for coeff in LADDER:
        t0, r0, v0, E0, T0 = _y0(coeff)

        for name, val in [("t0", t0), ("r0", r0), ("v0", v0), ("E0", E0), ("T0", T0)]:
            assert np.isfinite(val) and val > 0, f"{name}={val} at coeff={coeff}"

        assert v0 == pytest.approx(np.sqrt(coeff) * v0_1, rel=1e-10)
        assert t0 == pytest.approx(coeff**-0.75 * t0_1, rel=1e-10)  # tSF = 0
        assert r0 == pytest.approx(coeff**-0.25 * r0_1, rel=1e-10)
        assert E0 == pytest.approx(coeff**0.25 * E0_1, rel=1e-10)


# =============================================================================
# 4. End-to-end: a weak-winds run boots and launches a slower shell (stress)
# =============================================================================


def _boot_run(tmp_path, tag, therm_coeff_wind):
    """Boot a tiny bounded run (M43-scale pattern from
    test_early_phase_override) and return its snapshot rows."""
    workdir = tmp_path / tag
    workdir.mkdir()
    param = workdir / f"{tag}.param"
    param.write_text(
        "mCloud                  300\n"
        "sfe                     0.01\n"
        "nCore                   8.7e3\n"
        "include_PHII            True\n"
        "dens_profile            densPL\n"
        "densPL_alpha            0\n"
        "rCore                   0.05\n"
        "nISM                    1e2\n"
        "PISM                    2.3e5\n"
        "ZCloud                  1\n"
        "coverFraction           1.0\n"
        "TShell_neu              135\n"
        f"FB_thermCoeffWind       {therm_coeff_wind}\n"
        "stop_t                  1e-4\n"  # bound runtime; phase 1a still runs
        "coll_r                  0.005\n"
        "log_console             False\n"
        "path2output             outputs/\n"
    )
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), str(param)],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"run.py ({tag}) exited {result.returncode}\n"
        f"---stdout (tail)---\n{result.stdout[-3000:]}\n"
        f"---stderr (tail)---\n{result.stderr[-3000:]}"
    )
    jsonl = next((workdir / "outputs").rglob("dictionary.jsonl"))
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) >= 2, f"{tag}: run wrote {len(rows)} snapshot(s); need at least 2"
    return rows


@pytest.mark.stress
def test_weak_winds_run_boots_and_launches_slower(tmp_path):
    rows_ctrl = _boot_run(tmp_path, "ctrl", 1.0)
    rows_weak = _boot_run(tmp_path, "weak", 0.03)

    for tag, rows in (("ctrl", rows_ctrl), ("weak", rows_weak)):
        for row in rows:
            assert np.isfinite(row["R2"]) and row["R2"] > 0, f"{tag}: R2={row['R2']}"
            assert np.isfinite(row["v2"]), f"{tag}: v2={row['v2']}"

    # Row 0 carries the free-streaming exit velocity v0 ∝ sqrt(coeff): the
    # weak-wind shell must launch measurably slower than the control.
    assert rows_weak[0]["v2"] < 0.5 * rows_ctrl[0]["v2"], (
        f"weak-wind launch v0={rows_weak[0]['v2']:.1f} pc/Myr is not below half "
        f"the control v0={rows_ctrl[0]['v2']:.1f} — FB_thermCoeffWind is not "
        "reaching the free-streaming initial conditions"
    )
