#!/usr/bin/env python3
"""
Minimal working example comparing manual Euler vs scipy.integrate.odeint()
Shows why pure ODE function works with your dictionary structure.
"""

import numpy as np
import scipy.integrate
import time


# =============================================================================
# Simulate your dictionary structure
# =============================================================================

class SimpleDict:
    """Simulates your params dictionary."""
    def __init__(self):
        self.data = {
            't_now': 0.0,
            'R2': 1.0,
            'constant': 10.0,  # Some constant parameter
        }

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value


# =============================================================================
# Example 1: IMPURE ODE function (has side effects)
# =============================================================================

def ode_impure(y, t, params):
    """
    IMPURE: Modifies params during evaluation.
    This is what causes dictionary corruption with scipy!
    """
    R, v = y

    # SIDE EFFECT: Writing to params
    params['t_now'] = t  # Problem with scipy!
    params['R2'] = R     # Problem with scipy!

    # Calculate derivatives
    dR_dt = v
    dv_dt = -params['constant'] * R  # Simple harmonic oscillator

    return [dR_dt, dv_dt]


# =============================================================================
# Example 2: PURE ODE function (no side effects)
# =============================================================================

def ode_pure(y, t, params):
    """
    PURE: Only reads from params, never writes.
    Safe for scipy to call any number of times!
    """
    R, v = y

    # ONLY READ from params - no side effects
    constant = params['constant']

    # Calculate derivatives
    dR_dt = v
    dv_dt = -constant * R  # Simple harmonic oscillator

    return [dR_dt, dv_dt]


# =============================================================================
# Method 1: Manual Euler (what you're doing now)
# =============================================================================

def solve_manual_euler(ode_func, y0, t_arr, params, use_impure=True):
    """Manual Euler integration."""
    y = y0.copy()
    results = [y.copy()]

    for i in range(len(t_arr) - 1):
        t = t_arr[i]
        dt = t_arr[i+1] - t_arr[i]

        # Get derivatives
        if use_impure:
            dydt = ode_func(y, t, params)
        else:
            dydt = ode_func(y, t, params)

        # Euler step
        y = y + np.array(dydt) * dt
        results.append(y.copy())

        # If using impure function, we'd update params here
        if use_impure:
            params['t_now'] = t_arr[i+1]
            params['R2'] = y[0]

    return np.array(results)


# =============================================================================
# Method 2: scipy.integrate.odeint() with PURE function
# =============================================================================

def solve_scipy(ode_func, y0, t_arr, params):
    """Solve with scipy.integrate.odeint()."""
    solution = scipy.integrate.odeint(
        ode_func,
        y0,
        t_arr,
        args=(params,),
        rtol=1e-6,
        atol=1e-8
    )

    # Update params AFTER solving (not during!)
    params['t_now'] = t_arr[-1]
    params['R2'] = solution[-1, 0]

    return solution


# =============================================================================
# Test and compare
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("COMPARISON: Manual Euler vs scipy.integrate.odeint()")
    print("=" * 80)

    # Initial conditions: Simple harmonic oscillator
    y0 = np.array([1.0, 0.0])  # [position, velocity]

    # Time array - very fine for Euler stability
    t_euler = np.linspace(0, 10, 10000)  # 10,000 steps for Euler
    t_scipy = np.linspace(0, 10, 100)     # 100 steps for scipy (adaptive!)

    # Create params
    params_euler = SimpleDict()
    params_scipy = SimpleDict()

    # ---------------------------------------------------------------------
    # Method 1: Manual Euler with IMPURE function
    # ---------------------------------------------------------------------

    print("\n1. MANUAL EULER (what you're doing now)")
    print("-" * 80)

    start = time.time()
    sol_euler = solve_manual_euler(ode_impure, y0, t_euler, params_euler, use_impure=True)
    time_euler = time.time() - start

    print(f"   Timesteps: {len(t_euler)}")
    print(f"   Time taken: {time_euler*1000:.2f} ms")
    print(f"   Final position: {sol_euler[-1, 0]:.6f}")
    print(f"   Final velocity: {sol_euler[-1, 1]:.6f}")
    print(f"   params['t_now']: {params_euler['t_now']:.6f}")
    print(f"   params['R2']: {params_euler['R2']:.6f}")

    # ---------------------------------------------------------------------
    # Method 2: scipy with PURE function
    # ---------------------------------------------------------------------

    print("\n2. SCIPY.INTEGRATE.ODEINT() with PURE function")
    print("-" * 80)

    start = time.time()
    sol_scipy = solve_scipy(ode_pure, y0, t_scipy, params_scipy)
    time_scipy = time.time() - start

    print(f"   Timesteps: {len(t_scipy)} (but scipy uses adaptive sub-steps)")
    print(f"   Time taken: {time_scipy*1000:.2f} ms")
    print(f"   Final position: {sol_scipy[-1, 0]:.6f}")
    print(f"   Final velocity: {sol_scipy[-1, 1]:.6f}")
    print(f"   params['t_now']: {params_scipy['t_now']:.6f}")
    print(f"   params['R2']: {params_scipy['R2']:.6f}")

    # ---------------------------------------------------------------------
    # Analytical solution (for verification)
    # ---------------------------------------------------------------------

    print("\n3. ANALYTICAL SOLUTION")
    print("-" * 80)

    # For simple harmonic oscillator: R(t) = cos(sqrt(k)*t)
    # with k = constant, R(0) = 1, v(0) = 0
    t_final = 10.0
    k = params_euler['constant']
    R_analytical = np.cos(np.sqrt(k) * t_final)
    v_analytical = -np.sqrt(k) * np.sin(np.sqrt(k) * t_final)

    print(f"   Final position: {R_analytical:.6f}")
    print(f"   Final velocity: {v_analytical:.6f}")

    # ---------------------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------------------

    print("\n4. ERROR ANALYSIS")
    print("-" * 80)

    error_euler = abs(sol_euler[-1, 0] - R_analytical)
    error_scipy = abs(sol_scipy[-1, 0] - R_analytical)

    print(f"   Euler error: {error_euler:.3e}")
    print(f"   Scipy error: {error_scipy:.3e}")
    print(f"   Scipy is {error_euler/error_scipy:.1f}x more accurate")

    print("\n5. PERFORMANCE")
    print("-" * 80)

    speedup = time_euler / time_scipy
    print(f"   Euler took: {time_euler*1000:.2f} ms with {len(t_euler)} steps")
    print(f"   Scipy took: {time_scipy*1000:.2f} ms with {len(t_scipy)} output points")
    print(f"   Speedup: {speedup:.1f}x faster")

    print("\n6. KEY INSIGHT")
    print("-" * 80)
    print("   ✓ PURE function (ode_pure) works perfectly with scipy")
    print("   ✓ No dictionary corruption - params only updated AFTER solve")
    print("   ✓ Much faster AND more accurate")
    print("   ✓ Time always moves forward in params: t₀ → t₁ → t₂")
    print("\n   ✗ IMPURE function (ode_impure) would break with scipy")
    print("   ✗ But we can't use it with scipy anyway (side effects)")
    print("   ✗ Manual Euler is slow and inaccurate")

    print("\n" + "=" * 80)
    print("CONCLUSION: Use pure ODE function + scipy.integrate.odeint()")
    print("=" * 80)
    print()

    # ---------------------------------------------------------------------
    # Demonstrate dictionary corruption if we tried to use impure with scipy
    # ---------------------------------------------------------------------

    print("\n7. WHAT HAPPENS IF WE USE IMPURE FUNCTION WITH SCIPY?")
    print("-" * 80)
    print("   Let's try it and see the corruption...")

    # Track how many times ODE is evaluated
    call_count = [0]
    time_history = []

    def ode_impure_tracked(y, t, params):
        """Track calls to show scipy evaluates multiple times."""
        call_count[0] += 1
        time_history.append(t)

        # Side effect
        params['t_now'] = t
        params['R2'] = y[0]

        # Calculate
        R, v = y
        dR_dt = v
        dv_dt = -params['constant'] * R

        return [dR_dt, dv_dt]

    params_test = SimpleDict()

    # Solve with scipy
    t_test = np.linspace(0, 1, 10)  # Just 10 output points
    sol_test = scipy.integrate.odeint(ode_impure_tracked, y0, t_test, args=(params_test,))

    print(f"   Requested {len(t_test)} output points")
    print(f"   But scipy called ODE function {call_count[0]} times!")
    print(f"\n   Time values scipy evaluated at:")
    print(f"   {np.array(time_history[:20])}...")  # Show first 20
    print(f"\n   Notice: scipy goes back and forth in time!")
    print(f"   This would corrupt your dictionary if you update based on t")
    print(f"\n   With PURE function: No problem - just reading, not writing")
    print(f"   With IMPURE function: Dictionary gets corrupted!")

    print("\n" + "=" * 80)
