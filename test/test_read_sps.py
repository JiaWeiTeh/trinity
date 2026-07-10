import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
from trinity._input.registry import _resolve_sps_bundle
from trinity.sps import read_sps, sps_columns


class _Item:
    def __init__(self, value):
        self.value = value


_SPS_LABELS = (
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

_CANONICAL_UNITS = {
    "t": "Myr",
    "Qi": "1/Myr",
    "Lbol": "Msun*pc^2/Myr^3",
    "Lmech_W": "Msun*pc^2/Myr^3",
    "Lmech_total": "Msun*pc^2/Myr^3",
    "pdot_W": "Msun*pc/Myr^2",
}

_FIRST_FILE_ROW = 1
_FIRST_ROW_GOLDENS = {
    # Captured 2026-07-10 on Python 3.9.6, numpy 1.26.4, scipy 1.13.1,
    # astropy 6.0.1, pandas 2.3.3, matplotlib 3.9.4, pytest 8.4.2.
    "t": 0.01,
    "Qi": 1.7025660421083365e66,
    "Li": 3214436867076.4536,
    "Ln": 2514907154102.974,
    "Lbol": 5729344021179.428,
    "Lmech_W": 20281725496.49822,
    "Lmech_SN": 0.0,
    "Lmech_total": 20281725496.49822,
    "pdot_W": 10848045.012862816,
    "pdot_SN": 0.0,
    "pdot_total": 10848045.012862816,
}


def _default_sps_params():
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
    return params


def test_default_sps_loader_known_first_row():
    params = _default_sps_params()

    sps_data = read_sps.read_sps(1.0, params)
    assert len(sps_data) == len(_SPS_LABELS)
    arrays = dict(zip(_SPS_LABELS, sps_data))

    assert params["sps_path"].value.endswith("lib/default/sps/starburst99/1e6cluster_default.csv")
    assert params["sps_refmass"].value == 1e6
    assert params["sps_column_map"].value == sps_columns.DEFAULT_SPS_COLUMN_MAP
    for canonical, unit in _CANONICAL_UNITS.items():
        assert sps_columns.CANONICALS[canonical].canonical_au_unit == unit

    assert arrays["t"][0] == 0.0
    assert np.all(np.diff(arrays["t"]) > 0.0)
    for label, expected in _FIRST_ROW_GOLDENS.items():
        assert arrays[label][_FIRST_FILE_ROW] == pytest.approx(expected, rel=1.0e-12, abs=1.0e-30)
