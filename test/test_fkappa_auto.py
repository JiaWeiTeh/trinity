"""cooling_boost_kappa='auto' — load-time f_kappa calibration lookup.

Contract under test (trinity/_input/fkappa_auto.py):
* numeric values pass through the resolver UNTOUCHED (default 1.0 path stays
  byte-identical);
* 'auto' resolves to the MEASURED f_kappa_fire of the 819-run sweep at grid
  points, log-space trilinear interpolation between them, clamped to the hull;
* the censored diffuse/high-SFE corner resolves to the sweep ceiling (64) —
  firing was NOT demonstrated there;
* end-to-end: a .param carrying `cooling_boost_kappa auto` comes out of
  read_param as the resolved float (mCloud axis = PRE-SFE input mass).

Grid provenance: docs/dev/transition/pdv-trigger/data/fkappa_nH_sweep.csv.
"""

import numpy as np
import pytest

from trinity._input.fkappa_auto import (
    F_KAPPA_CEILING,
    fkappa_fire,
    resolve_fkappa_auto,
)
from trinity._input.read_param import read_param


# ---------------------------------------------------------------------------
# pure lookup
# ---------------------------------------------------------------------------
def test_grid_points_return_measured_values():
    # spot checks straight from fkappa_nH_sweep.csv (f_kappa_fire_measured)
    assert fkappa_fire(1e5, 0.03, 1e3) == pytest.approx(12.0)
    assert fkappa_fire(1e5, 0.10, 1e2) == pytest.approx(64.0)
    assert fkappa_fire(1e6, 0.10, 3e3) == pytest.approx(12.0)
    assert fkappa_fire(1e7, 0.03, 3e3) == pytest.approx(1.0)
    assert fkappa_fire(1e5, 0.10, 1e5) == pytest.approx(1.5)


def test_interpolation_lies_between_neighbours():
    # halfway (log space) along nCore between 3e2 (16) and 1e3 (12)
    mid_n = 10 ** ((np.log10(3e2) + np.log10(1e3)) / 2)
    val = fkappa_fire(1e5, 0.03, mid_n)
    assert 12.0 < val < 16.0


def test_out_of_hull_is_clamped():
    # below every axis -> the (1e5, 0.03, 1e2) corner cell, measured 32
    assert fkappa_fire(1e4, 0.01, 10.0) == pytest.approx(32.0)
    # above nCore hull -> the n=1e5 edge
    assert fkappa_fire(1e5, 0.03, 1e7) == pytest.approx(1.0)


def test_censored_corner_returns_ceiling():
    # (1e5, sfe=0.3, n=1e2): nothing up to f_kappa=64 fired in the sweep
    assert fkappa_fire(1e5, 0.30, 1e2) == pytest.approx(F_KAPPA_CEILING)


def test_never_below_one():
    assert fkappa_fire(1e7, 0.03, 1e5) >= 1.0


# ---------------------------------------------------------------------------
# resolver
# ---------------------------------------------------------------------------
def test_numeric_passthrough_is_untouched():
    # byte-identity contract: the default path must not even be re-floated
    for v in (1.0, 4, 2.5):
        assert resolve_fkappa_auto(v, params={}) is v


def test_non_auto_string_passthrough():
    # defensive: unknown strings are left for downstream validation
    assert resolve_fkappa_auto("typo", params={}) == "typo"


# ---------------------------------------------------------------------------
# end-to-end through read_param (Step 7 resolver wiring)
# ---------------------------------------------------------------------------
def _write_param(tmp_path, extra):
    lines = [
        "mCloud    1e5",
        "sfe    0.03",
        "nCore    1e3",
        "rCore    1.0",
        f"path2output    {tmp_path / 'out'}",
        "log_file    False",
    ] + extra
    p = tmp_path / "fkauto_test.param"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_read_param_resolves_auto(tmp_path):
    params = read_param(_write_param(tmp_path, ["cooling_boost_kappa    auto"]))
    # the sweep measured f_kappa_fire = 12 at (mCloud_input=1e5, sfe=0.03, n=1e3);
    # mCloud itself is rebound post-SFE (0.97e5) — the resolver must use mCloud_input
    assert params["cooling_boost_kappa"].value == pytest.approx(12.0)
    assert isinstance(params["cooling_boost_kappa"].value, float)


def test_read_param_default_stays_exactly_one(tmp_path):
    params = read_param(_write_param(tmp_path, []))
    assert params["cooling_boost_kappa"].value == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
