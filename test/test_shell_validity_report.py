"""The two reporting-only shell-validity columns written by shell_structure.record_validity.

Why they exist (docs/dev/phii-identity/PLAN.md row 63, 2026-10-06): the shell solve puts its
inner edge at n0 ~ Pb (Rahner+2017 eq. 14) while, once C3c has fired, the EOM pushes with
P_HII > Pb -- so every fired row stores a layer at a pressure the dynamics has left behind, and
that layer re-equilibrates in the stored output at 6-19x the ionised sound speed through the
transition. Neither condition is visible from the dynamics; these columns name them so a reader
can gate structure diagnostics (thickness, R_IF, n_IF, profile exports) on them.

    shell_bc_mismatch   P_HII > Pb at the saved state
    shell_vt_ci         d(shell_thickness)/dt / c_i between consecutive saved states

Nothing in the dynamics reads either: the trajectory gate for the change that added them is a
separate-process full run bit-identical on every pre-existing column (homogeneous + BE configs).
"""
from pathlib import Path

import numpy as np
import pytest

from trinity._input.dictionary import DescribedDict
from trinity._input.read_param import read_param
from trinity.shell_structure.shell_structure import record_validity

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_crash_handlers(monkeypatch):
    monkeypatch.setattr(DescribedDict, "_register_crash_handlers", lambda self: None)


@pytest.fixture
def params():
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    # the homogeneous HEAD run at transition onset (t = 2.496 Myr): thermal Pb still the drive
    p["t_now"].value = 2.496
    p["Pb"].value = 730.0
    p["P_HII"].value = 0.0
    p["shell_thickness"].value = 0.332
    return p


def _c_i(p):
    return np.sqrt(p["k_B"].value * p["TShell_ion"].value / p["mu_ion_shell"].value)


def test_first_state_has_no_rate_and_sets_the_reference(params):
    assert np.isnan(params["shell_vt_t_prev"].value)
    record_validity(params)
    assert np.isnan(params["shell_vt_ci"].value)
    assert params["shell_vt_t_prev"].value == 2.496
    assert params["shell_vt_thickness_prev"].value == 0.332


def test_flag_is_strictly_P_HII_above_Pb(params):
    record_validity(params)
    assert params["shell_bc_mismatch"].value is False          # P_HII = 0 (confined)
    params["P_HII"].value = 730.0                              # the C3c branch point, P_c3a == Pb
    record_validity(params)
    assert params["shell_bc_mismatch"].value is False
    params["P_HII"].value = 730.0 * (1 + 1e-12)
    record_validity(params)
    assert params["shell_bc_mismatch"].value is True


def test_rate_is_thickness_change_over_time_in_sound_speed_units(params):
    record_validity(params)
    params["t_now"].value = 2.736
    params["shell_thickness"].value = 0.332 + 7.8 * _c_i(params) * 0.24   # thickened at 7.8 c_i
    record_validity(params)
    assert params["shell_vt_ci"].value == pytest.approx(7.8, rel=1e-12)
    assert params["shell_vt_t_prev"].value == 2.736
    # thinning is negative, not folded
    params["t_now"].value = 2.976
    params["shell_thickness"].value -= 0.5 * _c_i(params) * 0.24
    record_validity(params)
    assert params["shell_vt_ci"].value == pytest.approx(-0.5, rel=1e-12)


def test_resolve_at_the_same_time_changes_nothing(params):
    """Phase-end reconciliation re-solves at the state's own t_now, and the next phase's
    first state is at that same t_now: neither may produce a 0/0 or move the reference."""
    record_validity(params)
    params["t_now"].value = 2.903
    params["shell_thickness"].value = 33.4
    record_validity(params)
    v = params["shell_vt_ci"].value
    assert np.isfinite(v) and v > 0
    params["shell_thickness"].value = 33.6                     # a different solve, same t
    record_validity(params)
    assert params["shell_vt_ci"].value == v
    assert params["shell_vt_thickness_prev"].value == 33.4


def test_dissolved_shell_gives_nan_rate_but_still_flags(params):
    record_validity(params)
    params["t_now"].value = 2.903
    params["shell_thickness"].value = np.nan                   # shell_structure_pure's dissolved branch
    params["P_HII"].value = 1.2e3
    record_validity(params)
    assert np.isnan(params["shell_vt_ci"].value)
    assert params["shell_bc_mismatch"].value is True


def test_reference_keys_stay_out_of_snapshots(params):
    assert params["shell_vt_t_prev"].exclude_from_snapshot
    assert params["shell_vt_thickness_prev"].exclude_from_snapshot
    assert not params["shell_vt_ci"].exclude_from_snapshot
    assert not params["shell_bc_mismatch"].exclude_from_snapshot
