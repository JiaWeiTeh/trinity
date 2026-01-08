# TRINITY Architecture Analysis

## Overall Architecture Pattern

TRINITY follows a **procedural pipeline architecture** with some object-oriented patterns for data containers. The simulation flows linearly through distinct phases, with a central `params` dictionary acting as both configuration and state store.

---

## Core Design Patterns

### 1. DescribedDict/DescribedItem Pattern

**Location**: `src/_input/dictionary.py`

This is the central data container pattern used throughout TRINITY:

```python
class DescribedItem:
    """Container for value + metadata"""
    __slots__ = ("_value", "info", "ori_units", "exclude_from_snapshot")

class DescribedDict(dict):
    """Dictionary of DescribedItems with snapshot capabilities"""
```

**Strengths**:
- Self-documenting: Each parameter carries its own description and units
- Supports arithmetic operations directly (no `.value` needed in many contexts)
- Built-in snapshot/persistence mechanism
- Array simplification for large data

**Weaknesses**:
- Mutable global state passed everywhere
- Side effects hidden within ODE functions (modify `params` during integration)

---

### 2. Phase-Based Pipeline

**Flow**:
```
run.py
    └── main.start_expansion(params)
            ├── Phase 0: get_InitCloudProp, get_InitPhaseParam
            ├── Phase 1a: run_energy_phase.run_energy(params)
            ├── Phase 1b: run_energy_implicit_phase.run_phase_energy(params)
            ├── Phase 1c: run_transition_phase.run_phase_transition(params)
            └── Phase 2: run_momentum_phase.run_phase_momentum(params)
```

**Pattern**: Each phase module is responsible for:
1. Reading initial conditions from `params`
2. Running an integration loop
3. Updating `params` with results
4. Optionally saving snapshots

---

### 3. ODE Integration Pattern

**Current Implementation** (Manual Euler):

```python
# From src/phase1_energy/run_energy_phase.py
for ii, time in enumerate(t_arr):
    y = [R2, v2, Eb, T0]
    rd, vd, Ed, Td = energy_phase_ODEs.get_ODE_Edot(y, time, params)
    R2 += rd * dt_min
    v2 += vd * dt_min
    # ...
```

**Issues**:
- First-order Euler integration (low accuracy)
- Fixed timestep (inefficient)
- ODE function modifies `params` (impure, breaks scipy.integrate compatibility)

**Ideal Pattern** (from `analysis/` refactored code):

```python
def get_ODE_Edot_pure(y, t, params):
    """Pure function: only reads params, never writes"""
    R2, v2, Eb, T0 = y
    # Calculate derivatives (no side effects)
    return [dRdt, dvdt, dEdt, dTdt]

sol = scipy.integrate.odeint(get_ODE_Edot_pure, y0, t_arr, args=(params,))
```

---

## Module Dependency Graph

```
run.py
    │
    ├── src/_input/read_param.py
    │       └── src/_input/dictionary.py
    │       └── src/_functions/unit_conversions.py
    │
    └── src/main.py
            ├── src/phase0_init/
            │       ├── get_InitCloudProp.py
            │       └── get_InitPhaseParam.py
            │
            ├── src/sb99/
            │       ├── read_SB99.py
            │       └── update_feedback.py
            │
            ├── src/phase1_energy/
            │       ├── run_energy_phase.py
            │       └── energy_phase_ODEs.py
            │           ├── bubble_structure/
            │           ├── shell_structure/
            │           └── cooling/
            │
            ├── src/phase1b_energy_implicit/
            │       ├── run_energy_implicit_phase.py
            │       └── get_betadelta.py
            │
            ├── src/phase1c_transition/
            │       └── run_transition_phase.py
            │
            └── src/phase2_momentum/
                    └── run_momentum_phase.py
```

---

## State Management

### Current Approach: Mutable Global State

The `params` DescribedDict is passed through all functions and modified in place:

```python
def some_function(params):
    # Read state
    R2 = params['R2'].value

    # Do calculations...

    # Write state (side effect!)
    params['R2'].value = new_R2
    params['t_now'].value = time
```

**Problems**:
1. Hard to track what functions modify which parameters
2. Order of operations matters but isn't explicit
3. Can't easily parallelize
4. Breaks scipy.integrate (requires deepcopy workarounds)

### Proposed Approach: Immutable State

```python
@dataclass(frozen=True)
class SimulationState:
    t: float
    R2: float
    v2: float
    Eb: float
    T0: float

def evolve_state(state: SimulationState, params: Config) -> SimulationState:
    """Pure function returning new state"""
    # Calculate new values
    return SimulationState(t=new_t, R2=new_R2, ...)
```

---

## File Organization Analysis

### Current Structure

```
src/
├── _input/          # Private: Input handling
├── _output/         # Private: Output formatting
├── _functions/      # Private: Utilities
├── _plots/          # Private: Visualization
├── phase0_init/     # Public: Phase 0
├── phase1_energy/   # Public: Phase 1a
├── phase1b_energy_implicit/  # Phase 1b
├── phase1c_transition/       # Phase 1c
├── phase2_momentum/ # Public: Phase 2
├── bubble_structure/# Physics module
├── shell_structure/ # Physics module
├── cloud_properties/# Physics module
├── cooling/         # Physics module
├── cloudy/          # External interface
├── sb99/            # External interface
└── phase_general/   # Shared phase utilities
```

### Observations

**Positive**:
- Clear separation between phases
- Private modules (underscore prefix) for internal utilities
- Physics modules are somewhat isolated

**Concerns**:
1. **Deep nesting**: Some imports go 3-4 levels deep
2. **Circular dependencies**: `energy_phase_ODEs` calls `shell_structure` which may need `energy_phase_ODEs` values
3. **Old code artifacts**: Many `_old.py`, `_legacy.py`, `_before*.py` files
4. **No clear API surface**: Everything imports from everywhere

---

## Numerical Integration Architecture

### Timestep Management

```
Phase 1 (Energy): dt_min = 1e-6 Myr, ~30 steps per loop
Phase 1b (Implicit): Adaptive via get_betadelta optimization
Phase 1c (Transition): Similar to Phase 1
Phase 2 (Momentum): Logarithmic timesteps from tmin to tmax
```

### Event Detection

Current approach uses manual checking after each step:

```python
def check_events(params, dt_params):
    # Check terminating conditions
    if t_next > params['stop_t'].value:
        return True  # Terminate
    if R2_next > params['stop_r'].value:
        return True  # Terminate
    # ...
    return False  # Continue
```

A better approach would use scipy's event handling:

```python
def shell_dissolved(t, y, params):
    """Event: shell density drops below threshold"""
    return params['shell_nMax'].value - params['stop_n_diss'].value

shell_dissolved.terminal = True
shell_dissolved.direction = -1

sol = scipy.integrate.solve_ivp(ode_func, t_span, y0, events=[shell_dissolved])
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        INITIALIZATION                            │
├─────────────────────────────────────────────────────────────────┤
│  .param file → read_param.py → DescribedDict(params)            │
│                                     ↓                            │
│  lib/sps/*.dat → read_SB99.py → SB99f interpolation functions   │
│                                     ↓                            │
│  lib/cooling/*.dat → cooling module → interpolation functions   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MAIN SIMULATION LOOP                        │
├─────────────────────────────────────────────────────────────────┤
│  For each timestep:                                              │
│    1. Update feedback (SB99f(t) → Lwind, Qi, etc.)              │
│    2. Calculate shell structure → fabs, tau, nmax               │
│    3. Calculate bubble structure → Tavg, Lcool                   │
│    4. Solve ODEs for [R2, v2, Eb, T0]                           │
│    5. Update params                                              │
│    6. Check termination conditions                               │
│    7. Save snapshot (periodic)                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                  │
├─────────────────────────────────────────────────────────────────┤
│  params.flush() → dictionary.jsonl (JSONL format)               │
│  Terminal output via print() statements                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Bottlenecks

Identified in `analysis/README_REFACTORED_CODE.md`:

| Component | Issue | Impact |
|-----------|-------|--------|
| `get_betadelta.py` | Manual 5x5 grid search | 3.7x slower than scipy.optimize |
| `run_energy_phase.py` | Manual Euler (100k steps) | 10-100x slower than scipy.integrate |
| ODE functions | Impure (modify params) | Requires deepcopy, breaks vectorization |
| `shell_structure.py` | Missing mu factors | Physics errors of 40-230% |

---

## Summary

TRINITY uses a straightforward procedural architecture that is easy to understand but has accumulated technical debt:

1. **Mutable state** makes debugging and parallelization difficult
2. **Manual integration** is slow and inaccurate
3. **No clear module boundaries** lead to circular dependencies
4. **Old code artifacts** clutter the repository

The `analysis/` directory contains refactored versions that address many of these issues, achieving 3-100x performance improvements.
