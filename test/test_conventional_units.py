"""
Tests for the user-facing "conventional units" display layer.

These verify that the conversions added for human-readable output are the
*right* conversions (correct factor AND direction), that the underlying
diagnosis is correct (e.g. Qi is stored internally as [1/Myr], not [1/s]),
and that conversion is display-only (stored values stay in internal AU).

Design notes — guarding against "passes by coincidence":
- Conversion factors are checked against astropy's independent unit
  algebra, not against trinity's own frozen numbers.
- Qi is round-tripped through the *real* SPS loader, so the test fails if
  the internal-unit diagnosis (1/Myr via *Myr2s at load) is wrong.
- Non-zero, non-unity sample values are chosen so a missing or wrong-direction
  conversion cannot produce the expected string.
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import pytest

import trinity._functions.unit_conversions as cvt
from trinity._output.run_constants import METADATA_FILENAME


# ---------------------------------------------------------------------------
# T1 — conversion factors match astropy (independent of trinity's constants)
# ---------------------------------------------------------------------------

class TestConversionFactorsVsAstropy:

    def test_energy_au_to_erg(self):
        u = pytest.importorskip("astropy.units")
        expected = (1.0 * u.M_sun * u.pc**2 / u.Myr**2).to(u.erg).value
        assert cvt.INV_CONV.E_au2cgs == pytest.approx(expected, rel=1e-9)

    def test_luminosity_au_to_erg_per_s(self):
        u = pytest.importorskip("astropy.units")
        expected = (1.0 * u.M_sun * u.pc**2 / u.Myr**3).to(u.erg / u.s).value
        assert cvt.INV_CONV.L_au2cgs == pytest.approx(expected, rel=1e-9)

    def test_photon_rate_per_Myr_to_per_s(self):
        # Internal Qi is [1/Myr]; conventional is [1/s]; the factor that takes
        # a rate from per-Myr to per-second is exactly s2Myr.
        u = pytest.importorskip("astropy.units")
        expected = (1.0 / u.Myr).to(1.0 / u.s).value
        assert cvt.s2Myr == pytest.approx(expected, rel=1e-9)

    def test_mass_loss_Msun_per_Myr_to_Msun_per_yr(self):
        u = pytest.importorskip("astropy.units")
        expected = (1.0 * u.M_sun / u.Myr).to(u.M_sun / u.yr).value
        assert cvt.Mdot_au2Msunyr == pytest.approx(expected, rel=1e-6)
        assert cvt.Mdot_au2Msunyr == 1e-6  # 1 Myr = 1e6 yr, exactly


# ---------------------------------------------------------------------------
# T2 — Qi round-trips through the real loader (proves the diagnosis)
# ---------------------------------------------------------------------------

def test_qi_loader_roundtrip_to_log_per_second():
    from trinity.sps.sps_columns import convert_to_canonical_au

    logQ_file = np.array([49.5, 50.0])  # SB99 file column: log10 Qi [1/s]
    Qi_internal = convert_to_canonical_au(logQ_file, "Qi", "1/s", log=True)

    # Internal value is NOT per-second — the loader multiplied by Myr2s
    # (~3.16e13), so it is far larger than the file's per-second value.
    assert np.all(Qi_internal > 10.0 ** logQ_file)

    # The display conversion (* s2Myr, then log10) recovers the physical
    # log Q [1/s] the file started with.
    Qi_per_s = Qi_internal * cvt.s2Myr
    np.testing.assert_allclose(np.log10(Qi_per_s), logQ_file, rtol=1e-12)


# ---------------------------------------------------------------------------
# T4 — termination debug report converts Eb to erg
# ---------------------------------------------------------------------------

def test_critical_params_eb_wired_to_erg():
    from trinity._output.simulation_end import CRITICAL_PARAMS

    eb = next(row for row in CRITICAL_PARAMS if row[0] == "Eb")
    assert eb[2] == "erg"                                     # unit label
    assert eb[3] == pytest.approx(cvt.INV_CONV.E_au2cgs)      # conversion factor


def test_termination_report_renders_eb_in_erg(tmp_path):
    from trinity._output.simulation_end import write_termination_debug_report

    # Two snapshots so the comparison table is built; Eb in internal AU.
    snaps = [{"t_now": 0.10, "Eb": 2.0}, {"t_now": 0.20, "Eb": 2.0}]
    (tmp_path / "dictionary.jsonl").write_text(
        "\n".join(json.dumps(s) for s in snaps) + "\n"
    )

    write_termination_debug_report(str(tmp_path), reason="unit-test")

    md = json.loads((tmp_path / METADATA_FILENAME).read_text())
    comparison = md["termination_debug"]["comparison"]
    eb_row = next(r for r in comparison if r["key"] == "Eb")

    assert eb_row["unit"] == "erg"
    expected = 2.0 * cvt.INV_CONV.E_au2cgs  # ~3.802e43, != raw 2.0
    assert eb_row["old"] == pytest.approx(expected)
    assert eb_row["new"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# T5 — SF-onset feedback summary emits correct conventional-unit values
# ---------------------------------------------------------------------------

class _Item:
    """Minimal stand-in for DescribedItem (only ``.value`` is used)."""

    def __init__(self, value):
        self.value = value


def test_sf_onset_summary_values(caplog):
    from trinity.phase0_init.get_InitPhaseParam import get_y0

    L_au2cgs = cvt.INV_CONV.L_au2cgs

    # Pick internal values that map to clean conventional numbers.
    Qi_internal = 1e50 / cvt.s2Myr        # -> log Q = 50.000 [1/s]
    Lbol_internal = 1e40 / L_au2cgs       # -> 1.000e+40 erg/s
    Lmech_W_internal = 1e38 / L_au2cgs    # -> 1.000e+38 erg/s
    Mdot0_target = 100.0                  # Msun/Myr -> 1.000e-04 Msun/yr
    # Mdot0 = pdot_W**2 / (2 * Lmech_W)  =>  pdot_W = sqrt(2 * Lmech_W * Mdot0)
    pdot_W_internal = math.sqrt(2.0 * Lmech_W_internal * Mdot0_target)

    params = {
        "mu_convert": _Item(1.4),
        "nCore": _Item(1.0e3),
        "bubble_xi_Tb": _Item(0.9),
        "tSF": _Item(0.0),
        "sps_f": _Item({
            "fQi": lambda t: Qi_internal,
            "fLbol": lambda t: Lbol_internal,
            "fLmech_W": lambda t: Lmech_W_internal,
            "fpdot_W": lambda t: pdot_W_internal,
        }),
    }

    with caplog.at_level(logging.INFO,
                         logger="trinity.phase0_init.get_InitPhaseParam"):
        get_y0(params)

    text = caplog.text
    assert "log Q=50.000 [1/s]" in text
    assert "Lbol=1.000e+40 erg/s" in text
    assert "Lmech_wind=1.000e+38 erg/s" in text
    assert "Mdot_wind=1.000e-04 Msun/yr" in text
