#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:46:23 2024
Rewritten: January 2026 - Improved safety, accuracy, and maintainability

@author: Jia Wei Teh

Unit conversion constants for TRINITY
======================================

Philosophy
----------
We use hardcoded constants (not astropy at runtime) for SPEED - no import overhead
and no repeated computation. However, we protect against accidental modification
and provide verification against astropy for accuracy.

All conversions are to "Astronomy Units" (AU):
    Mass:   Msun (solar masses)
    Length: pc (parsecs)
    Time:   Myr (megayears)

Constants derived from astropy.units and frozen for safety.
See test suite at bottom for accuracy verification.
"""

import re
import ast
from typing import Optional, Dict
from dataclasses import dataclass
from fractions import Fraction


# =============================================================================
# Immutable conversion constants (derived from astropy.units)
# =============================================================================

@dataclass(frozen=True)  # frozen=True makes this immutable
class ConversionConstants:
    """
    Immutable container for unit conversion factors.
    
    All constants convert CGS → Astronomy Units [Msun, pc, Myr].
    Derived using astropy.units (see verification test at bottom).
    
    The @dataclass(frozen=True) decorator prevents accidental modification:
        CONV.cm2pc = 999  # Raises FrozenInstanceError
    """
    
    # -------------------------------------------------------------------------
    # Base units
    # -------------------------------------------------------------------------
    
    # Length
    cm2pc: float = 3.240779289444365e-19
    km2pc: float = 3.240779289444365e-14  # = cm2pc / 1e-5
    
    # Time
    s2Myr: float = 3.168808781402895e-14
    
    # Mass
    g2Msun: float = 5.029144215870041e-34
    
    # -------------------------------------------------------------------------
    # Derived units (commonly used in astrophysics)
    # -------------------------------------------------------------------------
    
    # Number density: 1/cm³ → 1/pc³
    ndens_cgs2au: float = 2.937998946096347e+55
    
    # Photon flux: 1/cm²/s → 1/pc²/Myr
    phi_cgs2au: float = 3.0047272630641653e+50
    
    # Energy: erg → Msun·pc²/Myr²
    E_cgs2au: float = 5.260183968837699e-44
    
    # Luminosity: erg/s → Msun·pc²/Myr³
    L_cgs2au: float = 1.6599878161499254e-30
    
    # Momentum rate: g·cm/s² → Msun·pc/Myr²
    pdot_cgs2au: float = 1.623123174716277e-25
    
    # Momentum rate derivative: g·cm/s³ → Msun·pc/Myr³
    pdotdot_cgs2au: float = 5.122187189842638e-12
    
    # Gravitational constant: cm³/g/s² → pc³/Msun/Myr²
    G_cgs2au: float = 67400.3588611473
    
    # Velocity
    v_kms2au: float = 1.022712165045695     # km/s → pc/Myr
    v_cms2au: float = 1.022712165045695e-05  # cm/s → pc/Myr
    
    # Force: g·cm/s² → Msun·pc/Myr²
    F_cgs2au: float = 1.623123174716277e-25
    
    # Pressure: g/cm/s² → Msun/pc/Myr²
    Pb_cgs2au: float = 1545441495671.806
    
    # Boltzmann constant: erg/K → Msun·pc²/Myr²/K
    k_B_cgs2au: float = 5.260183968837699e-44
    
    # Thermal conduction coefficient: g·cm/s³/K^(7/2) → Msun·pc/Myr³/K^(7/2)
    c_therm_cgs2au: float = 5.122187189842638e-12
    
    # Energy density rate: erg/cm³/s → Msun/pc/Myr³
    dudt_cgs2au: float = 4.877042454381257e+25
    
    # Cooling function: erg·cm³/s → Msun·pc⁵/Myr³
    Lambda_cgs2au: float = 5.650062667161655e-86
    
    # Surface density: g/cm² → Msun/pc²
    tau_cgs2au: float = 4788.452460043275
    
    # Gravitational potential: cm²/s² → pc²/Myr²
    gravPhi_cgs2au: float = 1.045940172532453e-10
    
    # Gravitational force per unit mass: cm/s² → pc/Myr²
    grav_force_m_cgs2au: float = 322743414.19646025
    
    def __post_init__(self):
        """Verify that all constants are positive (sanity check)."""
        for field_name, value in self.__dataclass_fields__.items():
            field_value = getattr(self, field_name)
            if not isinstance(field_value, (int, float)) or field_value <= 0:
                raise ValueError(f"Constant {field_name} must be positive, got {field_value}")


# Global immutable constant container
CONV = ConversionConstants()


# =============================================================================
# Inverse conversions (AU → CGS)
# =============================================================================

@dataclass(frozen=True)
class InverseConversionConstants:
    """Inverse conversions: Astronomy Units → CGS"""
    
    pc2cm: float = 1 / CONV.cm2pc
    Myr2s: float = 1 / CONV.s2Myr
    Msun2g: float = 1 / CONV.g2Msun
    
    ndens_au2cgs: float = 1 / CONV.ndens_cgs2au
    phi_au2cgs: float = 1 / CONV.phi_cgs2au
    E_au2cgs: float = 1 / CONV.E_cgs2au
    L_au2cgs: float = 1 / CONV.L_cgs2au
    pdot_au2cgs: float = 1 / CONV.pdot_cgs2au
    pdotdot_au2cgs: float = 1 / CONV.pdotdot_cgs2au
    G_au2cgs: float = 1 / CONV.G_cgs2au
    v_au2kms: float = 1 / CONV.v_kms2au
    v_au2cms: float = 1 / CONV.v_cms2au
    F_au2cgs: float = 1 / CONV.F_cgs2au
    Pb_au2cgs: float = 1 / CONV.Pb_cgs2au
    k_B_au2cgs: float = 1 / CONV.k_B_cgs2au
    c_therm_au2cgs: float = 1 / CONV.c_therm_cgs2au
    dudt_au2cgs: float = 1 / CONV.dudt_cgs2au
    Lambda_au2cgs: float = 1 / CONV.Lambda_cgs2au
    tau_au2cgs: float = 1 / CONV.tau_cgs2au
    gravPhi_au2cgs: float = 1 / CONV.gravPhi_cgs2au
    grav_force_m_au2cgs: float = 1 / CONV.grav_force_m_cgs2au


INV_CONV = InverseConversionConstants()


# =============================================================================
# Physical constants in CGS units
# =============================================================================
# These are fundamental physical constants, not conversion factors.
# Useful for calculations that need CGS values directly.

@dataclass(frozen=True)
class PhysicalConstantsCGS:
    """
    Fundamental physical constants in CGS units.

    These are the raw constants for calculations in CGS before converting
    to astronomy units. Values from CODATA 2018 / IAU 2015 resolutions.
    """

    # Gravitational constant: [cm³ g⁻¹ s⁻²]
    G: float = 6.67430e-8

    # Boltzmann constant: [erg K⁻¹] = [g cm² s⁻² K⁻¹]
    k_B: float = 1.380649e-16

    # Hydrogen atom mass: [g]
    m_H: float = 1.6735575e-24

    # Proton mass: [g]
    m_p: float = 1.67262192e-24

    # Electron mass: [g]
    m_e: float = 9.1093837e-28

    # Speed of light: [cm s⁻¹]
    c: float = 2.99792458e10

    # Stefan-Boltzmann constant: [erg cm⁻² s⁻¹ K⁻⁴]
    sigma_SB: float = 5.670374e-5

    # Planck constant: [erg s]
    h: float = 6.62607015e-27

    # Elementary charge (esu): [statcoulomb]
    e: float = 4.80320425e-10


CGS = PhysicalConstantsCGS()


# Backward-compatible aliases for CGS constants
G_CGS = CGS.G
K_B_CGS = CGS.k_B
M_H_CGS = CGS.m_H
M_P_CGS = CGS.m_p
M_E_CGS = CGS.m_e
C_CGS = CGS.c
SIGMA_SB_CGS = CGS.sigma_SB
H_CGS = CGS.h


# =============================================================================
# Backward compatibility aliases (deprecated, but kept for old code)
# =============================================================================
# These maintain compatibility with old code but are deprecated

cm2pc = CONV.cm2pc
pc2cm = INV_CONV.pc2cm
s2Myr = CONV.s2Myr
Myr2s = INV_CONV.Myr2s
ndens_cgs2au = CONV.ndens_cgs2au
ndens_au2cgs = INV_CONV.ndens_au2cgs
phi_cgs2au = CONV.phi_cgs2au
phi_au2cgs = INV_CONV.phi_au2cgs
E_cgs2au = CONV.E_cgs2au
E_au2cgs = INV_CONV.E_au2cgs
L_cgs2au = CONV.L_cgs2au
L_au2cgs = INV_CONV.L_au2cgs
pdot_cgs2au = CONV.pdot_cgs2au
pdot_au2cgs = INV_CONV.pdot_au2cgs
pdotdot_cgs2au = CONV.pdotdot_cgs2au
pdotdot_au2cgs = INV_CONV.pdotdot_au2cgs
G_cgs2au = CONV.G_cgs2au
G_au2cgs = INV_CONV.G_au2cgs
v_kms2au = CONV.v_kms2au
v_au2kms = INV_CONV.v_au2kms
v_cms2au = CONV.v_cms2au
v_au2cms = INV_CONV.v_au2cms
g2Msun = CONV.g2Msun
Msun2g = INV_CONV.Msun2g
F_cgs2au = CONV.F_cgs2au
F_au2cgs = INV_CONV.F_au2cgs
Pb_cgs2au = CONV.Pb_cgs2au
Pb_au2cgs = INV_CONV.Pb_au2cgs
k_B_cgs2au = CONV.k_B_cgs2au
k_B_au2cgs = INV_CONV.k_B_au2cgs
c_therm_cgs2au = CONV.c_therm_cgs2au
c_therm_au2cgs = INV_CONV.c_therm_au2cgs
dudt_cgs2au = CONV.dudt_cgs2au
dudt_au2cgs = INV_CONV.dudt_au2cgs
Lambda_cgs2au = CONV.Lambda_cgs2au
Lambda_au2cgs = INV_CONV.Lambda_au2cgs
tau_cgs2au = CONV.tau_cgs2au
tau_au2cgs = INV_CONV.tau_au2cgs
gravPhi_cgs2au = CONV.gravPhi_cgs2au
gravPhi_au2cgs = INV_CONV.gravPhi_au2cgs
grav_force_m_cgs2au = CONV.grav_force_m_cgs2au
grav_force_m_au2cgs = INV_CONV.grav_force_m_au2cgs


# =============================================================================
# Safe unit string parser (NO eval()!)
# =============================================================================

class UnitConversionError(Exception):
    """Raised when unit conversion fails."""
    pass


def convert2au(unit_string: Optional[str]) -> float:
    """
    Convert a unit string to astronomy units [Msun, pc, Myr].
    
    This function parses unit strings like "g*cm**2/s**3" and computes
    the conversion factor WITHOUT using eval() for safety.
    
    Parameters
    ----------
    unit_string : str or None
        Unit string from parameter file (e.g., "cm**3*s**-1")
        If None, returns 1 (dimensionless).
    
    Returns
    -------
    float
        Conversion factor to multiply original value by.
    
    Examples
    --------
    >>> convert2au("cm")
    3.240779289444365e-19
    
    >>> convert2au("g*cm**2/s**2")  # erg
    5.260183968837699e-44
    
    >>> convert2au("km*s**-1")  # velocity
    1.022712165045695
    
    Raises
    ------
    UnitConversionError
        If unit string contains unrecognized units or invalid syntax.
    
    Notes
    -----
    Does NOT use eval() - parses and computes safely using ast.literal_eval
    for exponents only.
    """
    
    # No unit = dimensionless
    if unit_string is None:
        return 1.0
    
    # Remove whitespace
    unit_string = re.sub(r'\s+', '', unit_string)
    
    # Handle empty string
    if not unit_string:
        return 1.0
    
    # Base unit conversion map
    unit_map: Dict[str, float] = {
        'g': CONV.g2Msun,
        's': CONV.s2Myr,
        'cm': CONV.cm2pc,
        'km': CONV.km2pc,
        'erg': CONV.E_cgs2au,
        # Mean molecular weight unit: dimensionless value × m_H [g] → Msun
        # Used for mu parameters (mu_atom, mu_ion, mu_mol, mu_convert)
        'm_H': CGS.m_H * CONV.g2Msun,
        # Dimensionless units (no conversion needed)
        'K': 1.0,
        'Zsun': 1.0,
        'Msun': 1.0,
        'pc': 1.0,
        'Myr': 1.0,
    }
    
    # Split on * and / (but not **)
    # Split into numerator and denominator parts
    # First split by / but not inside parentheses

    # Helper function to split respecting parentheses
    def split_by_slash(s):
        """Split by / but not inside parentheses."""
        parts = []
        current = []
        paren_depth = 0

        for char in s:
            if char == '(':
                paren_depth += 1
                current.append(char)
            elif char == ')':
                paren_depth -= 1
                current.append(char)
            elif char == '/' and paren_depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(char)

        if current:
            parts.append(''.join(current))

        return parts

    # Split into numerator and denominators
    parts = split_by_slash(unit_string)
    numerator = parts[0] if parts else ""
    denominators = parts[1:] if len(parts) > 1 else []

    # Split each part on single * (not **)
    def split_units(s):
        """Split on * but not **."""
        return [u.strip() for u in re.split(r'(?<!\*)\*(?!\*)', s) if u.strip()]

    numerator_units = split_units(numerator)
    denominator_units = []
    for denom in denominators:
        denominator_units.extend(split_units(denom))

    # Track total conversion factor
    total_factor = 1.0

    # Process numerator units
    for unit in numerator_units:
        # Parse: base_unit**exponent (allow underscores in unit names like m_H)
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)(\*\*(.+))?$', unit)

        if not match:
            raise UnitConversionError(f"Cannot parse unit: '{unit}' in '{unit_string}'")

        base_unit = match.group(1)
        exponent_str = match.group(3)

        # Get base conversion factor
        if base_unit not in unit_map:
            raise UnitConversionError(
                f"Unknown unit '{base_unit}'. "
                f"Known units: {', '.join(unit_map.keys())}"
            )

        base_factor = unit_map[base_unit]

        # Handle exponent
        if exponent_str:
            # Strip surrounding parentheses if present (e.g., "(-7/2)" -> "-7/2")
            exponent_str = exponent_str.strip('()')
            # Use Fraction for safe evaluation (handles division like "-7/2")
            try:
                exponent = float(Fraction(exponent_str))
            except (ValueError, ZeroDivisionError):
                raise UnitConversionError(
                    f"Invalid exponent '{exponent_str}' in unit '{unit}'"
                )
        else:
            exponent = 1.0

        # Apply conversion with exponent
        total_factor *= base_factor ** exponent

    # Process denominator units (negative exponent)
    for unit in denominator_units:
        # Parse: base_unit**exponent (allow underscores in unit names like m_H)
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)(\*\*(.+))?$', unit)

        if not match:
            raise UnitConversionError(f"Cannot parse unit: '{unit}' in '{unit_string}'")

        base_unit = match.group(1)
        exponent_str = match.group(3)

        # Get base conversion factor
        if base_unit not in unit_map:
            raise UnitConversionError(
                f"Unknown unit '{base_unit}'. "
                f"Known units: {', '.join(unit_map.keys())}"
            )

        base_factor = unit_map[base_unit]

        # Handle exponent
        if exponent_str:
            # Strip surrounding parentheses if present
            exponent_str = exponent_str.strip('()')
            # Use Fraction for safe evaluation (handles division like "-7/2")
            try:
                exponent = float(Fraction(exponent_str))
            except (ValueError, ZeroDivisionError):
                raise UnitConversionError(
                    f"Invalid exponent '{exponent_str}' in unit '{unit}'"
                )
        else:
            exponent = 1.0

        # Apply conversion with negative exponent (denominator)
        total_factor *= base_factor ** (-exponent)
    
    return total_factor


# =============================================================================
# Test suite (run with: python unit_conversions.py)
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Unit Conversion Test Suite")
    print("=" * 80)
    
    # Test 1: Immutability
    print("\n[Test 1] Immutability of constants")
    try:
        CONV.cm2pc = 999  # Should raise FrozenInstanceError
        print("FAILED: Constants are mutable!")
    except Exception as e:
        print(f"PASSED: Constants are immutable ({type(e).__name__})")
    
    # Test 2: Basic conversions
    print("\n[Test 2] Basic unit conversions")
    tests = [
        ("cm", CONV.cm2pc, "Length"),
        ("s", CONV.s2Myr, "Time"),
        ("g", CONV.g2Msun, "Mass"),
        ("km", CONV.km2pc, "Length (km)"),
    ]
    
    for unit, expected, description in tests:
        result = convert2au(unit)
        passed = abs(result - expected) < 1e-20
        status = "passed" if passed else "failed"
        print(f"{status} {description:20} {unit:10} → {result:.6e}")
    
    # Test 3: Compound units
    print("\n[Test 3] Compound unit conversions")
    compound_tests = [
        ("g*cm**2*s**-2", CONV.E_cgs2au, "Energy (erg)"),
        ("g*cm**2*s**-3", CONV.L_cgs2au, "Luminosity (erg/s)"),
        ("km*s**-1", CONV.v_kms2au, "Velocity (km/s)"),
        ("g*cm*s**-2", CONV.F_cgs2au, "Force (dyne)"),
        ("cm**-3", CONV.ndens_cgs2au, "Number density"),
    ]
    
    for unit, expected, description in compound_tests:
        result = convert2au(unit)
        passed = abs(result - expected) / expected < 1e-10
        status = "passed" if passed else "failed"
        print(f"{status} {description:25} {unit:20} → {result:.6e}")
    
    # Test 4: Dimensionless units
    print("\n[Test 4] Dimensionless units")
    dimensionless_tests = [
        ("K", 1.0, "Temperature"),
        ("Zsun", 1.0, "Metallicity"),
        ("Msun", 1.0, "Solar mass (AU)"),
        ("pc", 1.0, "Parsec (AU)"),
        ("Myr", 1.0, "Megayear (AU)"),
        (None, 1.0, "None (no unit)"),
    ]
    
    for unit, expected, description in dimensionless_tests:
        result = convert2au(unit)
        passed = result == expected
        status = "passed" if passed else "failed"
        print(f"{status} {description:25} {str(unit):10} → {result}")
    
    # Test 5: Error handling
    print("\n[Test 5] Error handling for invalid units")
    invalid_tests = [
        "foo",  # Unknown unit
        "cm**bar",  # Invalid exponent
        "123abc",  # Invalid format
    ]
    
    for unit in invalid_tests:
        try:
            convert2au(unit)
            print(f"FAILED: Should have raised error for '{unit}'")
        except UnitConversionError as e:
            print(f"PASSED: Correctly rejected '{unit}'")
    
    # Test 6: Verify against astropy (if available)
    print("\n[Test 6] Verification against astropy (if installed)")
    try:
        import astropy.units as u
        
        verification_tests = [
            ("cm → pc", u.cm.to(u.pc), CONV.cm2pc),
            ("s → Myr", u.s.to(u.Myr), CONV.s2Myr),
            ("g → Msun", u.g.to(u.M_sun), CONV.g2Msun),
            ("erg → Msun·pc²/Myr²", (u.erg).to(u.M_sun * u.pc**2 / u.Myr**2), CONV.E_cgs2au),
            ("erg/s → Msun·pc²/Myr³", (u.erg/u.s).to(u.M_sun * u.pc**2 / u.Myr**3), CONV.L_cgs2au),
            ("G", (u.cm**3/u.g/u.s**2).to(u.pc**3/u.M_sun/u.Myr**2), CONV.G_cgs2au),
            ("km/s → pc/Myr", (u.km/u.s).to(u.pc/u.Myr), CONV.v_kms2au),
        ]
        
        all_passed = True
        for description, astropy_val, our_val in verification_tests:
            rel_error = abs(astropy_val - our_val) / astropy_val
            passed = rel_error < 1e-10
            all_passed = all_passed and passed
            status = "passed" if passed else "failed"
            print(f"{status} {description:25} Rel.Error: {rel_error:.2e}")
        
        if all_passed:
            print("\n All constants match astropy within 1e-10 relative error!")
        
    except ImportError:
        print("⚠️  astropy not installed - skipping verification")
        print("   Install with: pip install astropy")
    
    # Test 7: Performance (no eval!)
    print("\n[Test 7] Performance test (1000 conversions)")
    import time
    
    start = time.time()
    for _ in range(1000):
        convert2au("g*cm**2*s**-3")
    elapsed = time.time() - start
    
    print(f"1000 conversions in {elapsed*1000:.2f} ms ({elapsed*1000000:.1f} μs per conversion)")
    print(f"   No eval() = SAFE and FAST!")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
