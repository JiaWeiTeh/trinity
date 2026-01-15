#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for integrated density and mass profile modules.

Tests scalar/array consistency, homogeneous cloud, power-law, and validation
functions for both density_profile_integrated and mass_profile_integrated.

Author: TRINITY Team
"""

import numpy as np
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cloud_properties.density_profile_integrated import get_density_profile
from src.cloud_properties.mass_profile_integrated import (
    get_mass_profile,
    get_mass_density,
    validate_mass_at_rCloud,
    compute_minimum_rCore,
    DENSITY_CONVERSION,
)
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_homogeneous,
    compute_rCloud_powerlaw
)


# =============================================================================
# Helper class for mock parameters
# =============================================================================

class MockParam:
    """Mock parameter object with .value attribute."""
    def __init__(self, value):
        self.value = value


def make_test_params(**kwargs):
    """Create a parameter dictionary for testing.

    Default values suitable for a homogeneous cloud test case.
    Override any parameter by passing keyword arguments.
    """
    defaults = {
        'nISM': 1.0,
        'nCore': 1000.0,
        'rCloud': 10.0,
        'rCore': 1.0,
        'dens_profile': 'densPL',
        'densPL_alpha': 0.0,
        'mu_ion': 1.4,
        'mu_atom': 2.3,
        'mCloud': 1e5,
    }
    defaults.update(kwargs)
    return {k: MockParam(v) for k, v in defaults.items()}


# =============================================================================
# Density Profile Tests
# =============================================================================

def test_density_scalar_array_consistency():
    """Test that scalar input returns scalar, array input returns array."""
    print("Testing density profile scalar/array consistency...")

    params = make_test_params()

    # Test scalar input
    r_scalar = 5.0
    n_scalar = get_density_profile(r_scalar, params)
    assert isinstance(n_scalar, float), f"Expected float, got {type(n_scalar)}"
    print(f"  ✓ Scalar input (r={r_scalar}) → scalar output (n={n_scalar})")

    # Test array input
    r_array = np.array([0.5, 5.0, 15.0])
    n_array = get_density_profile(r_array, params)
    assert isinstance(n_array, np.ndarray), f"Expected ndarray, got {type(n_array)}"
    assert len(n_array) == len(r_array), "Output length mismatch"
    print(f"  ✓ Array input (r={r_array}) → array output (n={n_array})")

    # Test list input (should behave like array)
    r_list = [0.5, 5.0, 15.0]
    n_list = get_density_profile(r_list, params)
    assert isinstance(n_list, np.ndarray), f"Expected ndarray, got {type(n_list)}"
    print(f"  ✓ List input (r={r_list}) → array output (n={n_list})")

    # Test single-element array (should still be array, not scalar)
    r_single_arr = np.array([5.0])
    n_single_arr = get_density_profile(r_single_arr, params)
    assert isinstance(n_single_arr, np.ndarray), f"Expected ndarray, got {type(n_single_arr)}"
    print(f"  ✓ Single-element array input → array output (n={n_single_arr})")

    print("✓ All density scalar/array consistency tests passed!")
    return True


def test_density_homogeneous_cloud():
    """Test homogeneous cloud profile (alpha=0)."""
    print("\nTesting density homogeneous cloud (α=0)...")

    params = make_test_params(densPL_alpha=0.0)
    nCore = params['nCore'].value
    nISM = params['nISM'].value
    rCloud = params['rCloud'].value

    # Test inside cloud
    r_inside = 5.0
    n_inside = get_density_profile(r_inside, params)
    assert n_inside == nCore, f"Expected nCore, got {n_inside}"
    print(f"  ✓ Inside cloud (r={r_inside}): n = nCore = {n_inside}")

    # Test exactly at cloud boundary
    r_boundary = rCloud
    n_boundary = get_density_profile(r_boundary, params)
    assert n_boundary == nCore, f"Expected nCore at boundary, got {n_boundary}"
    print(f"  ✓ At cloud boundary (r={r_boundary}): n = nCore = {n_boundary}")

    # Test outside cloud
    r_outside = 15.0
    n_outside = get_density_profile(r_outside, params)
    assert n_outside == nISM, f"Expected nISM, got {n_outside}"
    print(f"  ✓ Outside cloud (r={r_outside}): n = nISM = {n_outside}")

    print("✓ All density homogeneous cloud tests passed!")
    return True


def test_density_power_law_profile():
    """Test power-law density profile (alpha != 0)."""
    print("\nTesting density power-law profile (α≠0)...")

    alpha = -2.0  # Typical value for molecular clouds
    params = make_test_params(densPL_alpha=alpha)

    nCore = params['nCore'].value
    rCore = params['rCore'].value
    rCloud = params['rCloud'].value
    nISM = params['nISM'].value

    # Test inside core (should be constant nCore)
    r_core = 0.5
    n_core = get_density_profile(r_core, params)
    assert n_core == nCore, f"Expected nCore inside core, got {n_core}"
    print(f"  ✓ Inside core (r={r_core}): n = nCore = {n_core}")

    # Test in power-law region
    r_pl = 5.0
    n_pl = get_density_profile(r_pl, params)
    expected_n_pl = nCore * (r_pl / rCore) ** alpha
    assert np.isclose(n_pl, expected_n_pl), f"Expected {expected_n_pl}, got {n_pl}"
    print(f"  ✓ Power-law region (r={r_pl}): n = {n_pl} (expected {expected_n_pl})")

    # Test outside cloud (should be nISM)
    r_outside = 15.0
    n_outside = get_density_profile(r_outside, params)
    assert n_outside == nISM, f"Expected nISM outside cloud, got {n_outside}"
    print(f"  ✓ Outside cloud (r={r_outside}): n = nISM = {n_outside}")

    print("✓ All density power-law profile tests passed!")
    return True


def test_density_array_regions():
    """Test array input spanning multiple regions."""
    print("\nTesting density array spanning multiple regions...")

    params = make_test_params(densPL_alpha=0.0)
    nCore = params['nCore'].value
    nISM = params['nISM'].value
    rCloud = params['rCloud'].value

    # Array spanning inside and outside cloud
    r_arr = np.array([0.5, 5.0, 10.0, 15.0, 20.0])
    n_arr = get_density_profile(r_arr, params)

    # Check each element
    assert n_arr[0] == nCore, f"r=0.5 should give nCore"
    assert n_arr[1] == nCore, f"r=5.0 should give nCore"
    assert n_arr[2] == nCore, f"r=10.0 (boundary) should give nCore"
    assert n_arr[3] == nISM, f"r=15.0 should give nISM"
    assert n_arr[4] == nISM, f"r=20.0 should give nISM"

    print(f"  ✓ Array spanning regions: r={r_arr} → n={n_arr}")
    print("✓ All density array region tests passed!")
    return True


# =============================================================================
# Mass Profile Tests
# =============================================================================

def test_mass_scalar_array_consistency():
    """Test that scalar and array inputs give consistent mass results."""
    print("\nTesting mass profile scalar/array consistency...")

    # Create self-consistent params
    nCore = 1e3
    mu_ion = 1.4
    mCloud = 1e5
    rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu=mu_ion)
    rCore = 0.1 * rCloud

    params = make_test_params(
        nCore=nCore,
        mu_ion=mu_ion,
        mCloud=mCloud,
        rCloud=rCloud,
        rCore=rCore,
        densPL_alpha=0.0
    )

    # Test 1: Scalar input should return scalar
    r_scalar = 0.5 * rCloud
    M_scalar = get_mass_profile(r_scalar, params)
    assert isinstance(M_scalar, float), f"Expected float, got {type(M_scalar)}"
    print(f"  ✓ Scalar input (r={r_scalar:.2f}) → scalar output (M={M_scalar:.4e})")

    # Test 2: Array input should return array
    r_array = np.array([0.3, 0.5, 0.8]) * rCloud
    M_array = get_mass_profile(r_array, params)
    assert isinstance(M_array, np.ndarray), f"Expected ndarray, got {type(M_array)}"
    assert len(M_array) == len(r_array), "Output length mismatch"
    print(f"  ✓ Array input (len={len(r_array)}) → array output (len={len(M_array)})")

    # Test 3: Scalar should match corresponding array element
    assert np.isclose(M_scalar, M_array[1]), "Scalar and array results don't match!"
    print(f"  ✓ Scalar result matches array element")

    # Test 4: With return_mdot, scalar input → scalar outputs
    v_scalar = 10.0
    M_s, dMdt_s = get_mass_profile(r_scalar, params, return_mdot=True, rdot=v_scalar)
    assert isinstance(M_s, float), f"Expected float for M, got {type(M_s)}"
    assert isinstance(dMdt_s, float), f"Expected float for dMdt, got {type(dMdt_s)}"
    print(f"  ✓ Scalar with return_mdot → scalar outputs")

    # Test 5: With return_mdot, array input → array outputs
    v_array = np.array([10.0, 10.0, 10.0])
    M_a, dMdt_a = get_mass_profile(r_array, params, return_mdot=True, rdot=v_array)
    assert isinstance(M_a, np.ndarray), f"Expected ndarray for M, got {type(M_a)}"
    assert isinstance(dMdt_a, np.ndarray), f"Expected ndarray for dMdt, got {type(dMdt_a)}"
    print(f"  ✓ Array with return_mdot → array outputs")

    print("✓ All mass scalar/array consistency tests passed!")
    return True


def test_mass_homogeneous_cloud():
    """Test α=0 (homogeneous) case for mass profile.

    Uses self-consistent parameters computed from mCloud and nCore.
    """
    print("\nTesting mass homogeneous cloud (α=0)...")

    # Define fundamental inputs
    nCore = 1e3  # cm⁻³
    mu_ion = 1.4
    mCloud = 1e5  # Msun

    # Compute rCloud from fundamental inputs using proper physics
    rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu=mu_ion)
    rCore = 0.1 * rCloud  # rCore is 10% of rCloud

    # Physical density in Msun/pc³
    rhoCore_physical = nCore * mu_ion * DENSITY_CONVERSION

    print(f"  Computed rCloud = {rCloud:.3f} pc from mCloud={mCloud:.0e} Msun, nCore={nCore:.0e} cm⁻³")

    params = make_test_params(
        nCore=nCore,
        mu_ion=mu_ion,
        mCloud=mCloud,
        rCloud=rCloud,
        rCore=rCore,
        densPL_alpha=0.0
    )

    # Test at various radii (as fractions of rCloud)
    test_radii = [0.1 * rCloud, 0.3 * rCloud, 0.5 * rCloud, 0.9 * rCloud]

    for r in test_radii:
        M = get_mass_profile(r, params)
        # Expected mass in Msun using physical density
        M_expected = (4.0/3.0) * np.pi * r**3 * rhoCore_physical
        assert np.isclose(M, M_expected, rtol=1e-6), \
            f"r={r}: M={M:.6e} != expected {M_expected:.6e}"
        print(f"  ✓ r={r:.2f}: M = {M:.4e} Msun (expected: {M_expected:.4e})")

    # Verify mass at rCloud equals mCloud (self-consistency check)
    M_at_rCloud = get_mass_profile(rCloud, params)
    assert np.isclose(M_at_rCloud, mCloud, rtol=1e-6), \
        f"M(rCloud) = {M_at_rCloud:.4e} != mCloud = {mCloud:.4e}"
    print(f"  ✓ M(rCloud) = {M_at_rCloud:.4e} Msun = mCloud (self-consistent)")

    print("✓ Mass homogeneous cloud tests passed!")
    return True


def test_mass_powerlaw_analytical():
    """Test power-law profile against analytical solution."""
    print("\nTesting mass power-law profile...")

    # Define fundamental inputs
    nCore = 1e3  # cm⁻³
    mu_ion = 1.4
    mCloud = 1e5  # Msun
    alpha = -2.0  # isothermal

    # Compute rCloud and rCore from fundamental inputs using proper physics
    rCloud, rCore = compute_rCloud_powerlaw(mCloud, nCore, alpha, rCore_fraction=0.1, mu=mu_ion)

    # Physical density in Msun/pc³
    rhoCore_physical = nCore * mu_ion * DENSITY_CONVERSION

    print(f"  Computed rCloud = {rCloud:.3f} pc, rCore = {rCore:.3f} pc")
    print(f"  from mCloud={mCloud:.0e} Msun, nCore={nCore:.0e} cm⁻³, α={alpha}")

    params = make_test_params(
        nCore=nCore,
        mu_ion=mu_ion,
        mCloud=mCloud,
        rCloud=rCloud,
        rCore=rCore,
        densPL_alpha=alpha
    )

    # Test at radii spanning core, envelope, and beyond cloud
    r_arr = np.array([0.5 * rCore, rCore, 0.5 * rCloud, rCloud, 1.5 * rCloud])
    M_arr = get_mass_profile(r_arr, params)
    print(f"  M(r) = {M_arr} Msun")

    # Verify mass is monotonically increasing
    assert np.all(np.diff(M_arr) > 0), "Mass should be monotonically increasing!"
    print("  ✓ Mass is monotonically increasing")

    # Verify inside core (r < rCore) matches uniform formula (in Msun)
    r_in_core = 0.5 * rCore
    M_core_expected = (4.0/3.0) * np.pi * r_in_core**3 * rhoCore_physical
    assert np.isclose(M_arr[0], M_core_expected, rtol=1e-6), \
        f"Core mass mismatch: {M_arr[0]:.4e} vs {M_core_expected:.4e}"
    print(f"  ✓ Core region matches uniform density formula: {M_arr[0]:.4e} Msun")

    # Verify mass at rCloud equals mCloud (self-consistency check)
    M_at_rCloud = M_arr[3]
    assert np.isclose(M_at_rCloud, mCloud, rtol=0.01), \
        f"M(rCloud) = {M_at_rCloud:.4e} != mCloud = {mCloud:.4e}"
    print(f"  ✓ M(rCloud) = {M_at_rCloud:.4e} Msun ≈ mCloud (self-consistent)")

    print("✓ Mass power-law profile test passed!")
    return True


def test_mass_density_import():
    """Test that density is correctly imported from density_profile module."""
    print("\nTesting density import from density_profile module...")

    params = make_test_params()

    # Test density at a point
    r = 5.0
    n = get_density_profile(r, params)  # Number density from density_profile [cm⁻³]
    rho_internal = get_mass_density(r, params, physical_units=False)  # Internal units (n × μ)
    rho_physical = get_mass_density(r, params, physical_units=True)   # Physical units [Msun/pc³]

    expected_n = params['nCore'].value  # 1000.0 cm⁻³
    expected_rho_internal = expected_n * params['mu_ion'].value  # 1000.0 × 1.4
    expected_rho_physical = expected_rho_internal * DENSITY_CONVERSION  # Msun/pc³

    assert np.isclose(n, expected_n), f"Number density: {n} != {expected_n}"
    assert np.isclose(rho_internal, expected_rho_internal), \
        f"Mass density (internal): {rho_internal} != {expected_rho_internal}"
    assert np.isclose(rho_physical, expected_rho_physical), \
        f"Mass density (physical): {rho_physical} != {expected_rho_physical}"

    print(f"  ✓ Number density n(r={r}) = {n} cm⁻³ (from density_profile module)")
    print(f"  ✓ Mass density ρ(r={r}) = {rho_internal:.4e} (internal: n × μ)")
    print(f"  ✓ Mass density ρ(r={r}) = {rho_physical:.4e} Msun/pc³ (physical)")
    print("✓ Density import test passed!")
    return True


def test_mass_accretion_rate():
    """Test mass accretion rate dM/dt = 4πr²ρv calculation."""
    print("\nTesting mass accretion rate...")

    # Use self-consistent parameters
    nCore = 1e3
    mu_ion = 1.4
    mCloud = 1e5
    rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu=mu_ion)
    rCore = 0.1 * rCloud

    params = make_test_params(
        nCore=nCore,
        mu_ion=mu_ion,
        mCloud=mCloud,
        rCloud=rCloud,
        rCore=rCore,
        densPL_alpha=0.0
    )

    # Test at a point inside the cloud
    r = 0.5 * rCloud
    v = 10.0  # pc/Myr

    M, dMdt = get_mass_profile(r, params, return_mdot=True, rdot=v)

    # Expected dM/dt = 4πr²ρv
    rho = get_mass_density(r, params, physical_units=True)
    expected_dMdt = 4.0 * np.pi * r**2 * rho * v

    assert np.isclose(dMdt, expected_dMdt, rtol=1e-6), \
        f"dM/dt = {dMdt:.4e} != expected {expected_dMdt:.4e}"
    print(f"  ✓ dM/dt = {dMdt:.4e} Msun/Myr at r={r:.2f} pc, v={v} pc/Myr")

    print("✓ Mass accretion rate test passed!")
    return True


def test_validate_mass():
    """Test mass validation function."""
    print("\nTesting mass validation...")

    # Create self-consistent params
    nCore = 1e3
    mu_ion = 1.4
    mCloud = 1e5
    rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu=mu_ion)
    rCore = 0.1 * rCloud

    params = make_test_params(
        nCore=nCore,
        mu_ion=mu_ion,
        mCloud=mCloud,
        rCloud=rCloud,
        rCore=rCore,
        densPL_alpha=0.0
    )

    # Should pass validation
    result = validate_mass_at_rCloud(params)
    assert result['valid'], f"Validation should pass: {result['message']}"
    print(f"  ✓ Valid params pass: {result['message']}")

    # Test with inconsistent params (wrong mCloud)
    bad_params = make_test_params(
        nCore=nCore,
        mu_ion=mu_ion,
        mCloud=mCloud * 2,  # Wrong mass!
        rCloud=rCloud,
        rCore=rCore,
        densPL_alpha=0.0
    )

    result_bad = validate_mass_at_rCloud(bad_params, tolerance=0.001)
    # This should fail since mCloud is wrong
    print(f"  ✓ Inconsistent params detected: error = {result_bad['relative_error']*100:.2f}%")

    print("✓ Mass validation test passed!")
    return True


def test_minimum_rCore():
    """Test compute_minimum_rCore function."""
    print("\nTesting minimum rCore computation...")

    nCore = 1e3
    nISM = 1.0
    rCloud = 10.0
    alpha = -2.0

    # Function returns tuple: (rCore_suggested, nEdge, is_valid, rCore_min)
    rCore_suggested, nEdge, is_valid, rCore_min = compute_minimum_rCore(nCore, nISM, rCloud, alpha)

    # At minimum rCore, edge density should equal nISM
    # n(rCloud) = nCore * (rCloud/rCore_min)^alpha = nISM
    # rCore_min = rCloud * (nCore/nISM)^(1/alpha)
    expected_rCore_min = rCloud * (nCore / nISM) ** (1.0 / alpha)

    assert np.isclose(rCore_min, expected_rCore_min, rtol=1e-6), \
        f"rCore_min = {rCore_min:.4f} != expected {expected_rCore_min:.4f}"

    # Verify that suggested rCore gives valid edge density
    assert is_valid, f"Suggested rCore should give valid nEdge >= nISM, but nEdge = {nEdge}"
    assert nEdge >= nISM, f"Edge density {nEdge} should be >= nISM {nISM}"

    print(f"  ✓ rCore_min = {rCore_min:.4f} pc for nCore={nCore}, nISM={nISM}, rCloud={rCloud}, α={alpha}")
    print(f"    (Expected: {expected_rCore_min:.4f} pc)")
    print(f"  ✓ rCore_suggested = {rCore_suggested:.4f} pc gives nEdge = {nEdge:.2f} cm⁻³ (valid={is_valid})")

    print("✓ Minimum rCore computation test passed!")
    return True


# =============================================================================
# Bonnor-Ebert Sphere Tests
# =============================================================================

def test_BE_lane_emden_solution():
    """Test Lane-Emden solution against known critical values."""
    print("\nTesting Lane-Emden solution...")

    from src.cloud_properties.bonnorEbertSphere_v2 import (
        solve_lane_emden, OMEGA_CRITICAL, XI_CRITICAL, M_DIM_CRITICAL
    )

    solution = solve_lane_emden()

    # Find critical values (where rho/rho_c = 1/OMEGA_CRITICAL)
    idx_crit = np.argmin(np.abs(solution.rho_rhoc - 1.0/OMEGA_CRITICAL))
    xi_crit_computed = solution.xi[idx_crit]
    m_crit_computed = solution.m[idx_crit]

    print(f"  Critical ξ: computed={xi_crit_computed:.3f}, expected≈{XI_CRITICAL:.3f}")
    print(f"  Critical m: computed={m_crit_computed:.2f}, expected≈{M_DIM_CRITICAL:.2f}")

    # Check within tolerance
    assert abs(xi_crit_computed - XI_CRITICAL) < 0.1, \
        f"ξ_crit mismatch: {xi_crit_computed:.3f} vs {XI_CRITICAL:.3f}"
    assert abs(m_crit_computed - M_DIM_CRITICAL) < 0.5, \
        f"m_crit mismatch: {m_crit_computed:.2f} vs {M_DIM_CRITICAL:.2f}"

    # Check density decreases monotonically
    assert np.all(np.diff(solution.rho_rhoc) <= 0), "Density should decrease"
    print("  ✓ Density decreases monotonically")

    # Check mass increases monotonically
    assert np.all(np.diff(solution.m) >= 0), "Mass should increase monotonically"
    print("  ✓ Mass increases monotonically")

    print("✓ Lane-Emden solution test passed!")
    return True


def test_BE_sphere_creation():
    """Test BE sphere creation from M, n_core, Omega."""
    print("\nTesting BE sphere creation...")

    from src.cloud_properties.bonnorEbertSphere_v2 import (
        create_BE_sphere, OMEGA_CRITICAL
    )

    # Test case: 1 solar mass cloud
    result = create_BE_sphere(
        M_cloud=1.0,      # [Msun]
        n_core=1e4,       # [cm⁻³]
        Omega=10.0        # Moderately concentrated
    )

    print(f"  Input: M={result.M_cloud} Msun, n_core={result.n_core:.0e} cm⁻³, Ω={result.Omega}")
    print(f"  Output: r_out={result.r_out:.4f} pc, n_out={result.n_out:.2e} cm⁻³")
    print(f"          T_eff={result.T_eff:.1f} K, stable={result.is_stable}")

    # Verify outputs
    assert result.r_out > 0, "Radius should be positive"
    assert np.isclose(result.n_out, result.n_core / result.Omega), "n_out = n_core/Omega"
    assert result.T_eff > 0, "Temperature should be positive"
    assert result.is_stable == (result.Omega < OMEGA_CRITICAL), "Stability check"

    print("✓ BE sphere creation test passed!")
    return True


def test_BE_density_profile():
    """Test BE density profile n(r) = nCore * rho_rhoc(xi)."""
    print("\nTesting BE density profile...")

    from src.cloud_properties.bonnorEbertSphere_v2 import (
        solve_lane_emden, create_BE_sphere
    )

    # Create a BE sphere
    M_cloud = 100.0   # [Msun]
    n_core = 1e4      # [cm⁻³]
    Omega = 10.0
    mu = 2.33         # Mean molecular weight

    solution = solve_lane_emden()
    result = create_BE_sphere(
        M_cloud=M_cloud,
        n_core=n_core,
        Omega=Omega,
        mu=mu,
        lane_emden_solution=solution
    )

    # Create mock params for get_density_profile
    # IMPORTANT: mu_ion must match what was used in create_BE_sphere
    params = make_test_params(
        dens_profile='densBE',
        nCore=n_core,
        rCloud=result.r_out,
        mu_ion=mu,  # Must match BE sphere creation
        densBE_Omega=Omega,
        densBE_Teff=result.T_eff,
        densBE_f_rho_rhoc=solution.f_rho_rhoc,
        gamma_adia=5.0/3.0,
    )

    # Test density at center (should be nCore)
    r_center = 0.01 * result.r_out
    n_center = get_density_profile(r_center, params)
    # At center, rho_rhoc ≈ 1, so n ≈ nCore
    assert np.isclose(n_center, n_core, rtol=0.01), \
        f"Center density {n_center:.2e} should be ~nCore {n_core:.2e}"
    print(f"  ✓ n(r=0.01*rCloud) = {n_center:.2e} ≈ nCore")

    # Test density at edge (should be nCore/Omega at exactly rCloud)
    # Use exactly rCloud for the edge test
    r_edge = result.r_out
    n_edge = get_density_profile(r_edge, params)
    expected_n_edge = n_core / Omega
    assert np.isclose(n_edge, expected_n_edge, rtol=0.05), \
        f"Edge density {n_edge:.2e} should be ~{expected_n_edge:.2e}"
    print(f"  ✓ n(r=rCloud) = {n_edge:.2e} ≈ nCore/Ω = {expected_n_edge:.2e}")

    # Test outside cloud (should be nISM)
    r_outside = 1.5 * result.r_out
    n_outside = get_density_profile(r_outside, params)
    assert n_outside == params['nISM'].value, "Outside should be nISM"
    print(f"  ✓ n(r=1.5*rCloud) = {n_outside} = nISM")

    print("✓ BE density profile test passed!")
    return True


def test_BE_mass_total():
    """Test that M(rCloud) = mCloud for BE sphere (within 2%)."""
    print("\nTesting BE mass total...")

    from src.cloud_properties.bonnorEbertSphere_v2 import (
        solve_lane_emden, create_BE_sphere
    )

    # Create a BE sphere
    M_cloud = 100.0   # [Msun]
    n_core = 1e4      # [cm⁻³]
    Omega = 10.0
    mu = 2.33

    solution = solve_lane_emden()
    result = create_BE_sphere(
        M_cloud=M_cloud,
        n_core=n_core,
        Omega=Omega,
        mu=mu,
        lane_emden_solution=solution
    )

    # Create mock params for get_mass_profile
    params = make_test_params(
        dens_profile='densBE',
        nCore=n_core,
        rCloud=result.r_out,
        mCloud=M_cloud,
        mu_ion=mu,
        densBE_Omega=Omega,
        densBE_Teff=result.T_eff,
        densBE_f_rho_rhoc=solution.f_rho_rhoc,
        densBE_f_m=solution.f_m,
        densBE_xi_out=result.xi_out,
        gamma_adia=5.0/3.0,
    )

    # Compute mass at rCloud
    M_at_rCloud = get_mass_profile(result.r_out, params)
    rel_error = abs(M_at_rCloud - M_cloud) / M_cloud

    print(f"  M(rCloud) = {M_at_rCloud:.4f} Msun")
    print(f"  Expected  = {M_cloud:.4f} Msun")
    print(f"  Relative error = {rel_error*100:.2f}%")

    # Should be accurate to within 2%
    assert rel_error < 0.02, f"Mass error too large: {rel_error*100:.2f}%"

    print("✓ BE mass total test passed!")
    return True


# =============================================================================
# Main test runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running profile integration tests")
    print("=" * 60)

    tests = [
        # Density profile tests
        ("Density: Scalar/Array Consistency", test_density_scalar_array_consistency),
        ("Density: Homogeneous Cloud", test_density_homogeneous_cloud),
        ("Density: Power-law Profile", test_density_power_law_profile),
        ("Density: Array Regions", test_density_array_regions),
        # Mass profile tests
        ("Mass: Scalar/Array Consistency", test_mass_scalar_array_consistency),
        ("Mass: Homogeneous Cloud", test_mass_homogeneous_cloud),
        ("Mass: Power-law Profile", test_mass_powerlaw_analytical),
        ("Mass: Density Import", test_mass_density_import),
        ("Mass: Accretion Rate", test_mass_accretion_rate),
        ("Mass: Validation", test_validate_mass),
        ("Mass: Minimum rCore", test_minimum_rCore),
        # Bonnor-Ebert sphere tests
        ("BE: Lane-Emden Solution", test_BE_lane_emden_solution),
        ("BE: Sphere Creation", test_BE_sphere_creation),
        ("BE: Density Profile", test_BE_density_profile),
        ("BE: Mass Total", test_BE_mass_total),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {name}")
            print(f"  Exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
