"""
Characterization tests for the ``convert2au`` unit-string parser.

The parser had no automated coverage (only an eyeball ``__main__`` harness),
so these pin its behaviour across every code path before/after refactoring:
numerator-only units, integer and negative exponents, single and multiple
``/`` denominators, parenthesised fractional exponents (e.g. the C_thermal
unit ``K**(-7/2)``), dimensionless units, ``None``/empty, and error cases.

Expected factors are composed independently from the base constants (not
copied magic numbers), so a flipped exponent sign, wrong unit routing, or
broken denominator handling fails the test rather than passing by coincidence.
"""

from __future__ import annotations

import pytest

import trinity._functions.unit_conversions as cvt

# Base factors the parser is expected to route to (see ``unit_map`` in
# convert2au): m_H maps to CGS.m_H * g2Msun; K/Zsun/Msun/pc/Myr are 1.0.
_cm = cvt.CONV.cm2pc
_km = cvt.CONV.km2pc
_s = cvt.CONV.s2Myr
_g = cvt.CONV.g2Msun
_erg = cvt.CONV.E_cgs2au
_mH = cvt.CGS.m_H * cvt.CONV.g2Msun


# (unit_string, expected_factor) — expected built from base factors only.
CASES = [
    # single base unit, numerator, no exponent
    ("cm", _cm),
    ("g", _g),
    ("s", _s),
    ("km", _km),
    ("m_H", _mH),
    ("erg", _erg),
    # integer / negative exponents, no slash
    ("cm**2", _cm**2),
    ("cm**-3", _cm**-3),
    # compound numerator with a `/` denominator (exercises denominator loop)
    ("km/s", _km / _s),
    ("K/pc", 1.0),
    # multiple `/` denominators (denominator loop with an exponent)
    ("Msun/Myr**2/pc", 1.0),
    # compound numerator, exponents in numerator only (no slash)
    ("g*cm**2*s**-2", _g * _cm**2 * _s**-2),       # erg
    ("g*cm**2*s**-3", _g * _cm**2 * _s**-3),       # erg/s
    ("g*cm*s**-2", _g * _cm * _s**-2),             # dyne
    ("cm**3 * s**-1", _cm**3 * _s**-1),            # caseB_alpha unit (w/ spaces)
    ("cm**2 * g**-1", _cm**2 * _g**-1),            # dust_KappaIR unit
    ("cm**3 * g**-1 * s**-2", _cm**3 * _g**-1 * _s**-2),  # G unit
    ("erg * K**-1", _erg),                          # k_B unit (K is 1.0)
    ("cm * s**-1", _cm * _s**-1),                   # c_light unit
    # parenthesised fractional exponent (C_thermal); K is 1.0 so 1**(-7/2)==1
    ("erg * s**-1 * cm**-1 * K**(-7/2)", _erg * _s**-1 * _cm**-1),
    # dimensionless / no-unit
    ("K", 1.0),
    ("Zsun", 1.0),
    ("Msun", 1.0),
    ("pc", 1.0),
    ("Myr", 1.0),
    ("", 1.0),
    (None, 1.0),
]


@pytest.mark.parametrize("unit_string, expected", CASES)
def test_convert2au_factor(unit_string, expected):
    assert cvt.convert2au(unit_string) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("bad", ["foo", "cm**bar", "123abc"])
def test_convert2au_rejects_invalid(bad):
    with pytest.raises(cvt.UnitConversionError):
        cvt.convert2au(bad)
