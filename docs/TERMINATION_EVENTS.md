# TRINITY Termination Events Overview

This document describes all termination events used across simulation phases.
Events are handled by the centralized module `src/phase_general/phase_events.py`.

## Event Categories

Events are categorized by their consequence:

| Category | Description | Example |
|----------|-------------|---------|
| **Simulation-Ending** | Stops entire simulation | `min_radius`, `max_radius`, `velocity_runaway` |
| **Phase-Ending** | Ends current phase, moves to next | `cloud_boundary`, `energy_floor`, `cooling_balance` |
| **Monitoring** | Records event but continues | `velocity_sign` (collapse onset) |

---

## Events by Phase

### Energy Phase (`run_energy_phase_modified.py`)

| Event | Type | Trigger Condition | Direction | Consequence |
|-------|------|-------------------|-----------|-------------|
| `cloud_boundary` | Phase-Ending | R2 > rCloud | +1 (rising) | -> Implicit phase |
| `min_radius` | Simulation-Ending | R2 < coll_r * 1.5 | -1 (falling) | End simulation |
| `velocity_runaway` | Simulation-Ending | v2 < -500 pc/Myr | -1 (falling) | End simulation |

**Notes:**
- Energy phase is very short (~3000 years)
- Collapse is rare in this early phase
- Main termination is reaching cloud boundary

---

### Implicit Phase (`run_energy_implicit_phase_modified.py`)

| Event | Type | Trigger Condition | Direction | Consequence |
|-------|------|-------------------|-----------|-------------|
| `velocity_sign` | Monitoring | v2 crosses zero | -1 (falling) | Records collapse onset |
| `min_radius` | Simulation-Ending | R2 < coll_r * 1.5 | -1 (falling) | End simulation |
| `max_radius` | Simulation-Ending | R2 > stop_r | +1 (rising) | End simulation |
| `velocity_runaway` | Simulation-Ending | v2 < -500 pc/Myr | -1 (falling) | End simulation |
| `cooling_balance`* | Phase-Ending | (Lgain - Lloss)/Lgain < 0.05 | -1 (falling) | -> Transition phase |

**Notes:**
- *`cooling_balance` is checked at segment level (not via solve_ivp events) because Lgain/Lloss are computed between segments
- This phase solves for beta/delta cooling parameters
- Collapse detection sets `isCollapse` flag

---

### Transition Phase (`run_transition_phase_modified.py`)

| Event | Type | Trigger Condition | Direction | Consequence |
|-------|------|-------------------|-----------|-------------|
| `energy_floor` | Phase-Ending | Eb < 1e3 erg | -1 (falling) | -> Momentum phase |
| `min_radius` | Simulation-Ending | R2 < coll_r * 1.5 | -1 (falling) | End simulation |
| `max_radius` | Simulation-Ending | R2 > stop_r | +1 (rising) | End simulation |
| `velocity_runaway` | Simulation-Ending | v2 < -500 pc/Myr | -1 (falling) | End simulation |

**Notes:**
- Energy decays on sound-crossing timescale: dE/dt = -Eb / t_sound
- When Eb drops below floor, thermal pressure is negligible
- Phase ends when bubble energy is depleted

---

### Momentum Phase (`run_momentum_phase_modified.py`)

| Event | Type | Trigger Condition | Direction | Consequence |
|-------|------|-------------------|-----------|-------------|
| `min_radius` | Simulation-Ending | R2 < coll_r * 1.5 | -1 (falling) | End simulation |
| `max_radius` | Simulation-Ending | R2 > stop_r | +1 (rising) | End simulation |
| `velocity_runaway` | Simulation-Ending | v2 < -500 pc/Myr | -1 (falling) | End simulation |

**Notes:**
- Final phase with Eb = 0 (no thermal pressure)
- Most likely phase for collapse during rapid shell contraction
- Events prevent LSODA crashes by stopping before R2 -> 0

---

## Post-Integration Termination Checks

In addition to solve_ivp events, each phase performs post-integration checks:

| Check | Condition | Sets |
|-------|-----------|------|
| `reached_tmax` | t_now > stop_t | `EndSimulationDirectly = True` |
| `small_radius` | R2 < coll_r (if isCollapse) | `EndSimulationDirectly = True` |
| `large_radius` | R2 > stop_r | `EndSimulationDirectly = True` |
| `dissolved` | shell_nMax < stop_n_diss | `EndSimulationDirectly = True` |
| `cloud_boundary` | R2 > rCloud (if !expansionBeyondCloud) | `EndSimulationDirectly = True` |
| `max_segments` | segment_count >= MAX_SEGMENTS | `termination_reason = "max_segments"` |

---

## Safety Thresholds

Defined in `phase_events.py`:

```python
MIN_RADIUS_SAFETY = 0.01       # pc - absolute minimum radius
MIN_RADIUS_FACTOR = 1.5        # Factor above coll_r for early termination
MAX_VELOCITY_COLLAPSE = 500.0  # pc/Myr (~490 km/s) - extreme inward velocity
MAX_VELOCITY_EXPANSION = 1000.0  # pc/Myr (~978 km/s) - extreme outward velocity
```

The minimum radius threshold is: `max(coll_r * MIN_RADIUS_FACTOR, MIN_RADIUS_SAFETY)`

---

## Event Result Structure

When an event triggers, `check_event_termination()` returns:

```python
@dataclass
class EventResult:
    triggered: bool           # Whether any event triggered
    name: str                 # Event name (e.g., "min_radius")
    index: int                # Index in events list (-1 if none)
    t: float                  # Time of event
    y: np.ndarray             # State vector at event
    is_simulation_ending: bool  # True if simulation should end
    reason_code: str          # Short code (e.g., "small_radius_event")
    reason_message: str       # Human-readable message
```

---

## Usage Example

```python
from src.phase_general.phase_events import (
    build_momentum_phase_events,
    check_event_termination,
    apply_event_result,
)

# Build events for the phase
ode_events = build_momentum_phase_events(params)

# Pass to solve_ivp
sol = scipy.integrate.solve_ivp(..., events=ode_events)

# Check result
event_result = check_event_termination(sol, ode_events)
if event_result.triggered:
    apply_event_result(params, event_result, event_result.t, event_result.y,
                      state_keys=['R2', 'v2'])
    if event_result.is_simulation_ending:
        break  # Exit simulation loop
```

---

## Phase Flow Diagram

```
Energy Phase (short, ~3000 yr)
    |
    | cloud_boundary event OR time limit
    v
Implicit Phase (cooling phase)
    |
    | cooling_balance (Lloss ~ Lgain) OR small_radius/velocity_runaway
    v
Transition Phase (energy decay)
    |
    | energy_floor (Eb < 1e3) OR small_radius/velocity_runaway
    v
Momentum Phase (final, Eb = 0)
    |
    | small_radius/large_radius/dissolved/tmax
    v
Simulation End
```
