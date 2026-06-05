"""Pins the Tavg volume-weighting fix (the intermediate-region sign bug).

Tavg = 3 * Σ(∫T r² dr) / Σ|r_outer³ - r_inner³|. The three regions tile
[R1, R2_coolingswitch]; r_bubble/r_conduction are descending grid slices while
r_interm = linspace(r2Prime, R2_coolingswitch) is ascending, so without abs()
the intermediate term carries the wrong sign and subtracts its volume. With
abs() the terms telescope to the true full-domain volume R2c³ - R1³.
"""
import numpy as np
import pytest


def _vol_signed(r_bubble, r_conduction, r_interm):
    return ((r_bubble[0]**3 - r_bubble[-1]**3) +
            (r_conduction[0]**3 - r_conduction[-1]**3) +
            (r_interm[0]**3 - r_interm[-1]**3))


def _vol_abs(r_bubble, r_conduction, r_interm):
    return (abs(r_bubble[0]**3 - r_bubble[-1]**3) +
            abs(r_conduction[0]**3 - r_conduction[-1]**3) +
            abs(r_interm[0]**3 - r_interm[-1]**3))


def test_abs_volume_telescopes_to_full_domain():
    # radii increase outward: R1 < r_CIEswitch < r2Prime < R2_coolingswitch
    R1, rc, r2P, R2c = 0.05, 0.108, 0.10810, 0.108105
    r_bubble = np.array([rc, R1])        # descending (grid slice)
    r_conduction = np.array([r2P, rc])   # descending (grid slice)
    r_interm = np.array([r2P, R2c])      # ascending (linspace) -> sign trap
    assert _vol_abs(r_bubble, r_conduction, r_interm) == pytest.approx(R2c**3 - R1**3, rel=0, abs=1e-18)


def test_signed_volume_undercounts_due_to_sign_bug():
    R1, rc, r2P, R2c = 0.05, 0.108, 0.10810, 0.108105
    r_bubble = np.array([rc, R1])
    r_conduction = np.array([r2P, rc])
    r_interm = np.array([r2P, R2c])
    # the old (signed) total subtracts the intermediate volume -> too small
    assert _vol_signed(r_bubble, r_conduction, r_interm) < _vol_abs(r_bubble, r_conduction, r_interm)
