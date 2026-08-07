"""Regression: phase 1a must integrate the physical force budget, not a constant.

`energy_phase_ODEs.get_ODE_Edot_pure` used to replace the computed shell
acceleration with a hardcoded ``vd = -1e8`` (pc/Myr^2) for the whole first
energy-phase segment, gated on the `EarlyPhaseApproximation` flag. Because a
constant RHS is integrated exactly, the segment-0 exit velocity collapsed to a
closed form that is the same number for **every** run on the bundled SB99
tables:

    v_exit = v0 - 1e8 * SEGMENT_DURATION = 3739.2407 - 3000 = 739.2407 pc/Myr
           = 722.82 km/s        (v0 = 2*Lmech/pdot is mass-scale invariant)

Measured across mCloud 3e3-3e6 and two core densities, that exit velocity never
moved: docs/dev/phase1a-init/data/segment1_exit.csv. At sub-GMC scale (a 0.15 pc
compact HII region) the resulting 80%-in-30-years deceleration dominates the
whole early trajectory and the observed radius is crossed ~22x too early.

The tests below pin the *property* rather than the formula, so they survive
legitimate changes to the force budget:

1. the acceleration must depend on the integration state,
2. it must depend on the shell's inertia,
3. phases 1b/1c — which call this same RHS — must never see a constant either,
4. (end-to-end, `stress`) the segment-0 exit must not sit on the closed form.

See docs/dev/phase1a-init/{FINDINGS,PLAN}.md.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import trinity.phase1_energy.energy_phase_ODEs as ODEs
from trinity.phase1_energy.run_energy_phase import SEGMENT_DURATION

REPO_ROOT = Path(__file__).resolve().parents[1]

# The override's magnitude. Kept here (not imported) so the test still describes
# the defect once the constant is gone from production.
OVERRIDE_VD = -1e8

# Two [R2, v2, Eb] states that a *physical* RHS must respond to differently.
# They share R2 and v2 and differ only in bubble energy, on purpose:
#   - v2 alone cannot discriminate, because the shell-mass ratchet in the RHS
#     freezes mShell_dot to 0 whenever the snapshot mass is the larger one, and
#     v2 enters only through the mShell_dot*v2 ram term;
#   - scaling R2 and Eb together cannot either, because the driving force
#     4*pi*R2^2*Pb goes as Eb/R2, so doubling both leaves the acceleration
#     unchanged (verified: identical to 6 figures).
# More thermal energy in the bubble at fixed radius must mean more acceleration.
STATE_A = (4.2953e-5, 3739.2407, 3.1770e2)
STATE_B = (4.2953e-5, 3739.2407, 7.9425e2)  # 2.5x Eb


def _snapshot(**overrides):
    """A physically plausible early-phase snapshot (sub-GMC-scale compact HII region).

    Values are taken from a real segment-0 state of
    docs/dev/phase1a-init/harness/params/probe.param, not round numbers.
    `shell_fAbsorbedIon = 1.0` and `rShell < rCloud` keep P_ext at zero so the
    ODE needs no params lookups beyond the two stubbed helpers.
    """
    fields = dict(
        shell_fAbsorbedIon=1.0,
        F_rad=7.39e4,
        rShell=4.2953e-5,
        shell_mass=9.9976e-5,
        isCollapse=False,
        n_IF=8.7e3,
        include_PHII=True,
        R_IF=4.2953e-5,
        mCluster=3.0,
        bubble_LTotal=1.31e6,
        Qi=1.6e47,
        Lmech_total=6.0845e8,
        v_mech_total=3739.24,
        G=4.4986e-3,
        k_B=1.3807e-16,
        gamma_adia=5.0 / 3.0,
        caseB_alpha=2.59e-13,
        PISM=2.3e5,
        nISM=1e2,
        TShell_ion=1e4,
        tSF=0.0,
        current_phase="energy",
        rCloud=0.617,
        P_HII=1.858e9,
        coverFraction=1.0,
        c_sound=1.4e3,
    )
    fields.update(overrides)
    return ODEs.ODESnapshot(**fields)


@pytest.fixture
def pure_ode(monkeypatch):
    """`get_ODE_Edot_pure` with its two params-reading helpers stubbed out.

    Only the SPS feedback lookup and the mass profile touch `params`; freezing
    them makes the RHS pure arithmetic, so any state dependence in the result
    comes from the force budget itself.
    """
    monkeypatch.setattr(
        ODEs,
        "get_current_sps_feedback",
        lambda t, params: SimpleNamespace(Lmech_total=6.0845e8, v_mech_total=3739.24),
    )
    monkeypatch.setattr(
        ODEs.mass_profile,
        "get_mass_profile",
        # swept-up mass of a uniform n=8.7e3 cm^-3 medium; mdot = dM/dR * v2
        lambda R2, params, return_mdot=False, rdot=0.0: (
            (1.2618e6 * R2**3, 3.7854e6 * R2**2 * rdot) if return_mdot else 1.2618e6 * R2**3
        ),
    )

    def call(snapshot, t=1.1487e-6, y=STATE_A):
        rd, vd, Ed = ODEs.get_ODE_Edot_pure(t, list(y), snapshot, params_for_feedback=None)
        # A NaN would satisfy every `!=` assertion below vacuously.
        assert math.isfinite(vd), f"RHS returned a non-finite acceleration: {vd}"
        return rd, vd, Ed

    return call


def test_acceleration_depends_on_state(pure_ode):
    """Two different states must not produce the same acceleration.

    The override returned -1e8 regardless of (R2, v2, Eb), which is what made
    the segment-0 exit velocity a universal constant.
    """
    snap = _snapshot()
    _, vd_a, _ = pure_ode(snap, y=STATE_A)
    _, vd_b, _ = pure_ode(snap, y=STATE_B)

    assert vd_a != vd_b, (
        f"shell acceleration is state-independent ({vd_a:.6e} pc/Myr^2 for both states) "
        "— the early-phase override is still governing the first segment"
    )
    assert not (
        vd_a == OVERRIDE_VD or vd_b == OVERRIDE_VD
    ), f"acceleration is the hardcoded override {OVERRIDE_VD:.1e} pc/Myr^2"


@pytest.mark.parametrize("phase", ["implicit", "transition"])
def test_override_cannot_govern_later_phases(pure_ode, phase):
    """The leak path: phases 1b/1c must never integrate a constant acceleration.

    `run_energy_implicit_phase.py:618` and `run_transition_phase.py:231` call this
    same RHS, and `create_ODE_snapshot` copies `EarlyPhaseApproximation` forward,
    so a phase-1a exit that skips the flag clear hands the override to them. Four
    in-loop 1a exits precede the clear (`run_energy_phase.py:183/287/330/331`,
    which is `loop_count == 0`-guarded and sits after the event check), and a
    phase 1a whose loop never runs skips it too.

    Reproduced end-to-end on the unfixed code with a documented, validator-free
    config (`cooling_boost_mode theta_target` + `cooling_boost_theta 0.96`, which
    makes the segment-0 cooling_balance break fire): phase 1b drove v2 from
    3739 pc/Myr to -2.3e-13 and fired `velocity_sign` (collapse onset) at
    t=3.74e-5 Myr — a wind-driven shell stalling 37 years in. That run is not
    used as the test because the same defect then grinds 1b at its DT floor for
    minutes; this unit check pins the same property in milliseconds.
    """
    snap = _snapshot(current_phase=phase)
    _, vd_a, _ = pure_ode(snap, y=STATE_A)
    _, vd_b, _ = pure_ode(snap, y=STATE_B)

    assert vd_a != vd_b, (
        f"phase {phase!r} integrates a state-independent acceleration "
        f"({vd_a:.6e} pc/Myr^2) — the early-phase override leaked out of phase 1a"
    )


def test_acceleration_depends_on_shell_inertia(pure_ode):
    """A heavier shell must respond differently to the same force budget."""
    light = _snapshot(shell_mass=9.9976e-5)
    heavy = _snapshot(shell_mass=9.9976e-3)

    _, vd_light, _ = pure_ode(light)
    _, vd_heavy, _ = pure_ode(heavy)

    assert vd_light != vd_heavy, (
        f"acceleration ignores the shell mass ({vd_light:.6e} pc/Myr^2 for a 100x "
        "heavier shell) — the early-phase override is still governing the first segment"
    )


@pytest.mark.stress
def test_segment0_exit_is_not_the_frozen_closed_form(tmp_path):
    """End-to-end: the first segment must not land on ``v0 - 1e8*SEGMENT_DURATION``.

    Snapshot 0 is written before the ODE runs, so row 0 carries the free-streaming
    initial velocity v0 and row 1 the segment-0 exit. On the unfixed code the two
    are related by the closed form to machine precision (measured: 6e-16 relative).
    """
    param = tmp_path / "m43.param"
    param.write_text(
        # sub-GMC-scale compact HII region; mirrors harness/params/probe.param
        "mCloud                  300\n"
        "sfe                     0.01\n"
        "nCore                   8.7e3\n"
        "include_PHII            True\n"
        "dens_profile            densPL\n"
        "densPL_alpha            0\n"
        "rCore                   0.05\n"  # the 1 pc default exceeds rCloud=0.617 here
        "nISM                    1e2\n"
        "PISM                    2.3e5\n"
        "ZCloud                  1\n"
        "coverFraction           1.0\n"
        "TShell_neu              135\n"
        "stop_t                  1e-4\n"  # bound runtime; phase 1a still runs its segments
        "coll_r                  0.005\n"
        "log_console             False\n"
        "path2output             outputs/\n"
    )

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), str(param)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"run.py exited {result.returncode}\n"
        f"---stdout (tail)---\n{result.stdout[-3000:]}\n"
        f"---stderr (tail)---\n{result.stderr[-3000:]}"
    )

    jsonl = next((tmp_path / "outputs").rglob("dictionary.jsonl"))
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) >= 2, f"run wrote {len(rows)} snapshot(s); need at least 2"

    v0, v_exit = rows[0]["v2"], rows[1]["v2"]
    frozen = v0 + OVERRIDE_VD * SEGMENT_DURATION

    assert abs(v_exit - frozen) > 1e-6 * abs(frozen), (
        f"segment-0 exit velocity {v_exit:.6f} pc/Myr sits on the frozen-override closed "
        f"form v0 + ({OVERRIDE_VD:.1e})*{SEGMENT_DURATION:g} = {frozen:.6f} — the shell's "
        "whole first segment is set by a constant, not by the force budget"
    )
    # And the physical sanity behind it: while the wind is still ramming the shell at
    # ~v0, it cannot shed most of its velocity within one early segment.
    assert v_exit > 0.5 * v0, (
        f"shell lost {100 * (1 - v_exit / v0):.0f}% of its velocity in one segment "
        f"({v0:.1f} -> {v_exit:.1f} pc/Myr) with the wind still driving it"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
