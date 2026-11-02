"""The SPS feedback interpolants must not ring
(docs/dev/phii-identity/weak-wind/early-1b/SPS_TRANSIENTS.md §5).

The cubic spline used before rang across the steps in the bundled Starburst99
table — the supernova switch-on (3.51-3.61 Myr) and switch-off (44.81-44.91
Myr) — and returned negative total mechanical luminosity (FB_thermCoeffWind
0.01: 3.431-3.502 Myr; every coefficient: 44.91-45.79 Myr) and momentum rate
(44.91-45.39 Myr). The fix: monotone PCHIP per column, totals = wind + SN, and
a table with negative or non-finite columns is rejected at load.
"""
import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
from trinity._input.registry import _resolve_sps_bundle
from trinity.sps import read_sps

# read_sps's return order after t
_KEYS = ("Qi", "Li", "Ln", "Lbol", "Lmech_W", "Lmech_SN", "Lmech_total",
         "pdot_W", "pdot_SN", "pdot_total")
_TOTALS = {"Lmech_total": "Lmech_W", "pdot_total": "pdot_W"}


class _Item:
    def __init__(self, value):
        self.value = value


def _params(coef):
    params = {
        "ZCloud": _Item(1.0),
        "SB99_rotation": _Item(1),
        "sps_refmass": _Item("def_value"),
        "FB_mColdWindFrac": _Item(0.0),
        "FB_mColdSNFrac": _Item(0.0),
        "FB_thermCoeffWind": _Item(coef),
        "FB_thermCoeffSN": _Item(1.0),
        "FB_vSN": _Item(1.0e4 * cvt.v_kms2au),
    }
    params["sps_path"] = _Item(_resolve_sps_bundle("def_path", params))
    return params


@pytest.mark.parametrize("coef", [0.01, 0.1, 1.0])
def test_interpolants_never_leave_the_bracketing_rows(coef):
    sps = read_sps.read_sps(1.0, _params(coef))
    f = read_sps.get_interpolation(sps)
    t = np.asarray(sps[0])
    tt = np.linspace(t[0], t[-1], 200001)
    i = np.clip(np.searchsorted(t, tt, side="right") - 1, 0, len(t) - 2)
    for k, y in zip(_KEYS, sps[1:]):
        y = np.asarray(y)
        v = f["f" + k](tt)
        np.testing.assert_allclose(f["f" + k](t), y, rtol=1e-12, atol=0)
        assert np.all(v >= 0), k
        if k in _TOTALS:   # a sum of two PCHIPs: never below its wind part
            assert np.all(v >= f["f" + _TOTALS[k]](tt)), k
            continue
        lo, hi = np.minimum(y[i], y[i + 1]), np.maximum(y[i], y[i + 1])
        tol = 1e-12 * hi + 1e-15 * y.max()
        assert np.all(v >= lo - tol) and np.all(v <= hi + tol), k


@pytest.mark.parametrize("coef", [0.01, 0.1, 1.0])
def test_feedback_positive_across_supernova_switch_on_and_off(coef):
    f = read_sps.get_interpolation(read_sps.read_sps(1.0, _params(coef)))
    for lo, hi in ((3.40, 3.62), (44.80, 45.90)):
        tt = np.linspace(lo, hi, 20001)
        for k in ("Lmech_total", "pdot_total"):
            assert np.all(f["f" + k](tt) > 0), (k, lo, hi)


def test_negative_table_column_is_rejected(tmp_path):
    params = _params(1.0)
    lines = open(params["sps_path"].value).read().splitlines()
    head = next(n for n, ln in enumerate(lines) if ln.startswith("t,"))
    row = lines[head + 3].split(",")
    row[2] = "0.1"   # log10 fi > 0: fi > 1, so Ln = Lbol (1 - fi) < 0
    lines[head + 3] = ",".join(row)
    bad = tmp_path / "bad_fi.csv"
    bad.write_text("\n".join(lines) + "\n")
    params["sps_path"] = _Item(str(bad))
    with pytest.raises(ValueError, match="Ln"):
        read_sps.read_sps(1.0, params)


def test_cubic_option_reproduces_the_old_spline():
    """sps_interp cubic: the pre-2026-11 interpolants, bit for bit (to reproduce older runs)."""
    from scipy.interpolate import interp1d, make_interp_spline

    sps = read_sps.read_sps(1.0, _params(0.01))
    f = read_sps.get_interpolation(sps, "cubic")
    t = np.asarray(sps[0])
    tt = np.linspace(t[0], t[-1], 20001)
    for k, y in zip(_KEYS, sps[1:]):
        assert np.array_equal(f["f" + k](tt), interp1d(t, y, kind="cubic")(tt)), k
    old = make_interp_spline(t, np.asarray(sps[10]), k=3).derivative()
    assert np.array_equal(f["fpdotdot_total"](tt), old(tt))
    assert f["fLmech_total"](3.472) < 0   # the artefact the default removes


def test_unknown_interpolation_is_rejected():
    from trinity._input.errors import ParameterFileError
    from trinity._input.registry import _validate_sps_interp

    sps = read_sps.read_sps(1.0, _params(1.0))
    with pytest.raises(ValueError, match="sps_interp"):
        read_sps.get_interpolation(sps, "linear")
    with pytest.raises(ParameterFileError, match="sps_interp"):
        _validate_sps_interp("linear", {})
    _validate_sps_interp("cubic", {})
