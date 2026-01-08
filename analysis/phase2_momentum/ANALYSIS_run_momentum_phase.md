# COMPREHENSIVE ANALYSIS: run_momentum_phase.py

**File**: `src/phase2_momentum/run_momentum_phase.py`
**Lines**: 272 (157 active, 115 commented-out)
**Purpose**: Momentum-driven phase of shell expansion
**Analysis Date**: 2026-01-08

---

## EXECUTIVE SUMMARY

**Overall Assessment**: 🔴 **CRITICAL ISSUES - 90% Duplicate of run_transition_phase.py**

This file has **identical problems** to `run_transition_phase.py` with **90% code duplication**!

### 🔴 CRITICAL ISSUES:
1. **MANUAL EULER INTEGRATION** (Lines 71-74) - First-order, inaccurate, unstable
2. **42% DEAD CODE** (115/272 lines commented out)
3. **BARE except: pass** (Lines 56-59) - Hides all errors
4. **NO ADAPTIVE STEPPING** - Fixed log-spaced timesteps
5. **CODE DUPLICATION** - 90% identical to `run_transition_phase.py`
6. **HARDCODED Eb=0** (Line 47) - Should be computed from transition

### ⚠️ MAJOR ISSUES:
- Same O(n²) array concatenation performance issue
- Same artifact handling for mass_profile inconsistencies
- Same print statements instead of logging
- Same magic number for timesteps (200 * log10(tmax/tmin))
- Same global state mutation

### ✅ WHAT IT DOES RIGHT:
- Same good event-based termination logic
- Same multiple physical stopping conditions
- Momentum-driven expansion physics (no energy evolution)

---

## WHAT THE SCRIPT DOES

### Purpose:
Momentum-driven phase models shell expansion when **momentum dominates over energy**.

### Physics:
- **Energy**: E = 0 (no energy evolution, all dissipated)
- **Velocity**: Evolves based on momentum conservation (ram pressure vs gravity)
- **Temperature**: Not evolved (dT/dt = 0)
- **Radius**: dr/dt = v

### Key Difference from Transition Phase:
```python
# Transition phase:
dE/dt = -E / t_soundcrossing  # Energy decays

# Momentum phase:
dE/dt = 0  # Energy already dissipated
Eb = 0     # Hard-coded (Line 47)
```

### Termination Conditions:
1. **Time limit**: t > stop_t
2. **Collapse**: v < 0 and R < R_collapse
3. **Large radius**: R > stop_r
4. **Dissolution**: shell density < stop_n_diss
5. **Cloud breakout**: R > R_cloud

*(No energy threshold event because E already 0)*

---

## CRITICAL FINDING: 90% CODE DUPLICATION

**Shocking**: This file is **virtually identical** to `run_transition_phase.py`!

### Line-by-Line Comparison:

| Section | run_transition_phase.py | run_momentum_phase.py | Identical? |
|---------|-------------------------|------------------------|------------|
| Imports | Lines 9-19 | Lines 8-15 | ✓ |
| run_phase function | Lines 23-126 | Lines 20-121 | ✓ (except ODE call) |
| Euler loop | Lines 53-79 | Lines 51-75 | ✓ |
| ODE function | Lines 131-211 | Lines 126-198 | 95% identical |
| check_events | Lines 215-287 | Lines 202-267 | ✓ (except energy event) |
| Commented refinement | Lines 82-125 | Lines 78-119 | ✓ |

**Only Actual Differences**:

1. **Energy initialization** (Line 47):
   ```python
   # Momentum phase:
   Eb = 0  # Hard-coded!

   # Transition phase:
   Eb = params['Eb'].value  # From previous phase
   ```

2. **Energy derivative** (Line 198):
   ```python
   # Momentum phase:
   return [rd, vd, 0, 0]  # dE/dt = 0

   # Transition phase:
   dEdt = -Eb / t_soundcrossing
   return [rd, vd, dEdt, 0]
   ```

3. **No energy event in check_events**:
   ```python
   # Transition phase has (Lines 242-244):
   if Eb_next < 1e3:
       return True

   # Momentum phase: No energy event (energy already 0)
   ```

**That's it!** 3 line differences out of 270+ lines = **90% duplication**! 🔴

---

## DETAILED CODE ANALYSIS

*(Most issues identical to run_transition_phase.py - see that analysis)*

### **UNIQUE ISSUE: Hardcoded Eb=0 (Line 47)**

**Severity**: ⚠️ MODERATE

**Problem**:
```python
# Line 47: Hard-coded!
Eb = 0
```

**Why This Is Questionable**:

1. **Should be computed**: Transition phase should set energy to small but non-zero value when handing off to momentum phase

2. **Inconsistent with y0**:
   ```python
   # Line 25: y0 includes params['Eb']
   y0 = [params['R2'].value, params['v2'].value, params['Eb'].value, params['T0'].value]

   # But then Line 47 overrides:
   Eb = 0  # Ignores params['Eb']!
   ```

3. **Magic value**: Why exactly 0? What if there's residual energy?

**Should Be**:
```python
# Use final energy from transition phase
Eb = params['Eb'].value

# Or validate it's small enough:
if params['Eb'].value > 1e3:
    raise ValueError(f"Energy too large for momentum phase: {params['Eb'].value}")
Eb = 0  # Can safely set to 0
```

---

### **Function 1: run_phase_momentum()** (Lines 20-121)

**IDENTICAL** to `run_phase_transition()` except:
- Calls `ODE_equations_momentum` instead of `ODE_equations_transition`
- Sets `Eb = 0` (Line 47)

**All issues from transition phase apply**:
- ✗ Manual Euler integration (Lines 71-74)
- ✗ Bare except: pass (Lines 56-59)
- ✗ Fixed timesteps (Lines 38-42)
- ✗ 42% dead code (Lines 78-119)

---

### **Function 2: ODE_equations_momentum()** (Lines 126-198)

**95% IDENTICAL** to `ODE_equations_transition()`.

**Only difference**:
```python
# Line 198: Momentum phase
return [rd, vd, 0, 0]  # No energy evolution

# vs Transition phase Line 211:
dEdt = -Eb / t_soundcrossing
return [rd, vd, dEdt, 0]
```

**All other code EXACTLY THE SAME**:
- Print statement (Line 131)
- params dict mutation (Lines 134-138)
- SB99 feedback (Line 153)
- Shell structure (Line 154)
- Array concatenation (Lines 162-166, 188)
- Mass profile (Lines 170-185)
- Artifact handling (Lines 179-185)
- params.save_snapshot() (Line 195)

**This is TERRIBLE software engineering!** 🔴

---

### **Function 3: check_events()** (Lines 202-267)

**IDENTICAL** to `run_transition_phase.check_events()` except:
- No energy threshold event (energy already 0)

**All other events identical**:
- Time limit (Lines 229-233)
- Collapse (Lines 236-240)
- Large radius (Lines 243-247)
- Dissolution (Lines 251-256)
- Cloud breakout (Lines 259-264)

---

## PHYSICS CORRECTNESS

### **Momentum Phase Physics**:

**Goal**: Model shell expansion when momentum dominates.

**Momentum Conservation**:
```python
# Shell momentum: p = m_shell * v_shell
# Force balance:
m_shell * dv/dt = F_ram - F_gravity - F_drag

# Where:
# - F_ram: Ram pressure from winds/SNe
# - F_gravity: Gravity from cloud/cluster
# - F_drag: Drag from ambient medium
```

**Energy**:
```python
# E = 0 (all energy dissipated)
# dE/dt = 0
```

**Is This Correct?** ✓

**Yes**, momentum phase assumes:
1. All thermal energy dissipated by cooling
2. Kinetic energy of shell >> internal energy
3. Expansion driven by momentum injection from winds/SNe

**Validated** by:
- Classic Weaver+ 1977 solution
- Momentum-driven phase is well-established in ISM physics

---

## ARCHITECTURAL PROBLEMS

### **SAME ISSUES** as run_transition_phase.py:

1. ❌ Manual Euler integration (first-order, inaccurate)
2. ❌ No adaptive stepping (fixed timesteps)
3. ❌ No error control (cannot assess accuracy)
4. ❌ Expensive array concatenation (O(n²) performance)
5. ❌ Global state mutation (params dict)
6. ❌ Bare except: pass (hides errors)
7. ❌ Print instead of logging
8. ❌ Magic numbers (200)
9. ❌ 42% dead code (commented refinement)
10. ❌ **90% code duplication with run_transition_phase.py** 🔴

### **PLUS**:

11. ❌ Hard-coded Eb=0 (should be validated from transition phase)

---

## UNIFIED SOLUTION

**Both files should be refactored into a common framework**:

```python
# base_phase.py
from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp

class PhaseIntegrator(ABC):
    """Base class for phase integration."""

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def ode_function(self, t, y):
        """ODE system for this phase."""
        pass

    @abstractmethod
    def get_events(self):
        """List of event functions."""
        pass

    def run(self, y0, t_span):
        """Run phase integration."""
        result = solve_ivp(
            fun=self.ode_function,
            t_span=t_span,
            y0=y0,
            method='RK45',
            events=self.get_events(),
            dense_output=True,
            rtol=1e-6,
            atol=1e-9
        )
        return result

class TransitionPhase(PhaseIntegrator):
    """Transition phase: energy decay."""

    def ode_function(self, t, y):
        R2, v2, Eb, T0 = y
        # ... compute derivatives ...
        t_sc = R2 / self.params['c_sound']
        dEdt = -Eb / t_sc  # Energy decay
        return [rd, vd, dEdt, 0]

    def get_events(self):
        return [
            self.energy_event,
            self.time_event,
            self.radius_event,
            self.collapse_event,
            self.dissolution_event,
            self.breakout_event
        ]

    def energy_event(self, t, y):
        """Energy threshold."""
        return y[2] - 1e3  # E < 1000 erg

class MomentumPhase(PhaseIntegrator):
    """Momentum phase: no energy evolution."""

    def ode_function(self, t, y):
        R2, v2, Eb, T0 = y
        # ... compute derivatives ...
        # No energy evolution
        return [rd, vd, 0, 0]

    def get_events(self):
        return [
            # No energy event
            self.time_event,
            self.radius_event,
            self.collapse_event,
            self.dissolution_event,
            self.breakout_event
        ]

# Usage:
transition = TransitionPhase(params)
result = transition.run(y0=[R2, v2, Eb, T0], t_span=[tmin, tmax])

momentum = MomentumPhase(params)
result = momentum.run(y0=[R2, v2, 0, T0], t_span=[tmin, tmax])
```

**Benefits**:
- ✓ Shared event handling code
- ✓ Shared integration logic
- ✓ Phase-specific only in `ode_function()` and `get_events()`
- ✓ Reduce from ~300 lines × 2 → ~200 lines shared + ~50 lines each
- ✓ scipy.integrate.solve_ivp (accurate, adaptive, stable)

---

## SUMMARY OF ISSUES

| Issue | Severity | Lines | Fix Effort |
|-------|----------|-------|------------|
| Manual Euler integration | 🔴 CRITICAL | 71-74 | 4 hours* |
| No adaptive stepping | 🔴 CRITICAL | 38-42 | 2 hours* |
| Bare except: pass | 🔴 CRITICAL | 56-59 | 15 min* |
| 90% code duplication | 🔴 CRITICAL | All | 1 day |
| Hard-coded Eb=0 | ⚠️ MODERATE | 47 | 30 min |
| Array concatenation O(n²) | ⚠️ MODERATE | 162-188 | 1 hour* |
| Artifact handling | ⚠️ MODERATE | 179-185 | 2 hours* |
| Magic numbers | ⚠️ MODERATE | 38 | 30 min* |
| Print instead of logging | ⚠️ MODERATE | 131, etc | 1 hour* |
| 42% dead code | ⚠️ MODERATE | 78-119 | 15 min* |

*Can be fixed once for both files with unified framework

**Total Issues**: 10 critical/moderate (shared with transition phase)
**Total Fix Effort**: ~1 day for unified refactor

---

## REFACTORING RECOMMENDATIONS

### Priority 1: Create Unified Framework (1 day)
- **Create PhaseIntegrator base class**
- **Refactor both files** to inherit from base
- **Share** event handling, integration logic, output handling
- **Use scipy.integrate.solve_ivp** with RK45

### Priority 2: Fix Shared Issues (~6 hours)
- **Pre-allocate** output arrays (fix O(n²) performance)
- **Fix** bare except
- **Remove** dead code (commented refinement blocks)
- **Replace** print with logging

### Priority 3: Fix Unique Issues (~1 hour)
- **Validate Eb** from transition phase
- **Fix** artifact handling in mass_profile
- **Add** named constants (magic numbers)

### Priority 4: Code Quality (~2 hours)
- **Add** comprehensive docstrings
- **Add** type hints
- **Add** unit tests for ODE functions
- **Add** integration tests for full phases

---

## FINAL VERDICT

**Rating**: ⚠️ 2/10 - Poor Code Quality, Urgent Refactoring Needed

**POSITIVE**:
✓ Good event-based termination logic
✓ Clear momentum-driven physics (E = 0)
✓ Multiple physical stopping conditions

**NEGATIVE**:
✗ Manual Euler integration (inaccurate, unstable)
✗ No adaptive stepping (fixed timesteps)
✗ No error control (cannot validate results)
✗ **90% code duplication with run_transition_phase.py** 🔴🔴🔴
✗ O(n²) performance (array concatenation)
✗ 42% dead code (commented refinement)
✗ Bare except: pass (hides errors)
✗ Hard-coded Eb=0 (should validate)

**RECOMMENDATION**:
**URGENT**: Refactor both files into unified framework:
- PhaseIntegrator base class
- scipy.integrate.solve_ivp
- Shared event handling
- Phase-specific only in ODE functions

**Effort**: 1 day for both files
**Priority**: 🔴 **CRITICAL** (code duplication is unacceptable!)

**Return on Investment**:
- Eliminate 270 lines of duplicate code
- Improve accuracy (RK45 vs Euler)
- Improve performance (pre-allocation)
- Improve maintainability (single implementation)
- Enable unit testing (smaller, focused functions)
