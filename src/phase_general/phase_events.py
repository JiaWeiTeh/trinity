#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Event Functions for TRINITY
=================================

Centralized module for ODE event functions used across all simulation phases.
These events are passed to scipy.integrate.solve_ivp to enable safe termination
during integration.

Event Types
-----------
Events are categorized by their consequence:

1. **Simulation-Ending Events** (EndSimulationDirectly=True):
   - min_radius: R2 < coll_r (shell collapse)
   - max_radius: R2 > stop_r (expansion limit)
   - velocity_runaway: |v2| > threshold (numerical instability)

2. **Phase-Ending Events** (move to next phase):
   - cloud_boundary: R2 > rCloud (energy phase -> implicit)
   - cooling_balance: L_cool ~ L_gain (implicit -> transition)
   - energy_floor: Eb < threshold (transition -> momentum)
   - velocity_sign: v2 crosses zero (collapse onset detection)

Usage
-----
Events are created via factory functions that capture phase-specific parameters:

    from src.phase_general.phase_events import (
        make_min_radius_event,
        make_max_radius_event,
        EventResult,
        check_event_termination,
    )

    # Create events for a phase
    events = [
        make_min_radius_event(coll_r * 1.5),
        make_max_radius_event(stop_r),
    ]

    # Pass to solve_ivp
    sol = solve_ivp(..., events=events)

    # Check results
    result = check_event_termination(sol, events)
    if result.triggered:
        print(f"Event '{result.name}' triggered at t={result.t}")

@author: TRINITY Team
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default safety thresholds
MIN_RADIUS_SAFETY = 0.01       # pc - absolute minimum radius
MIN_RADIUS_FACTOR = 1.5        # Factor above coll_r for early termination
MAX_VELOCITY_COLLAPSE = 500.0  # pc/Myr (~490 km/s) - extreme inward velocity
MAX_VELOCITY_EXPANSION = 1000.0  # pc/Myr (~978 km/s) - extreme outward velocity


# =============================================================================
# Event Result Container
# =============================================================================

@dataclass
class EventResult:
    """Container for event detection results."""
    triggered: bool
    name: str
    index: int  # Which event in the list triggered (-1 if none)
    t: float    # Time of event (NaN if not triggered)
    y: np.ndarray  # State at event (empty if not triggered)
    is_simulation_ending: bool  # True if simulation should end
    reason_code: str  # Short code for termination_reason
    reason_message: str  # Human-readable message for SimulationEndReason


# =============================================================================
# Simulation-Ending Event Factories
# =============================================================================

def make_min_radius_event(min_r: float, name: str = "min_radius"):
    """
    Create event that triggers when R2 falls below min_r.

    This prevents LSODA from crashing when R2 approaches zero during
    rapid collapse. The event is terminal - integration stops immediately.

    Parameters
    ----------
    min_r : float
        Minimum allowed radius (pc). Typically coll_r * factor or MIN_RADIUS_SAFETY.
    name : str
        Name for identifying this event in results.

    Returns
    -------
    event : callable
        Event function for solve_ivp with terminal=True, direction=-1.
        Has additional attributes: event.name, event.is_simulation_ending,
        event.reason_code, event.reason_message
    """
    def event(t, y):
        R2 = y[0]
        return R2 - min_r

    event.terminal = True
    event.direction = -1  # Only trigger when R2 crosses min_r from above
    event.name = name
    event.is_simulation_ending = True
    event.reason_code = "small_radius_event"
    event.reason_message = "Small radius reached (event)"
    return event


def make_max_radius_event(max_r: float, name: str = "max_radius"):
    """
    Create event that triggers when R2 exceeds max_r.

    Used to stop simulation when shell expands beyond stop_r limit.

    Parameters
    ----------
    max_r : float
        Maximum allowed radius (pc). Typically stop_r from params.
    name : str
        Name for identifying this event in results.

    Returns
    -------
    event : callable
        Event function for solve_ivp with terminal=True, direction=1.
    """
    def event(t, y):
        R2 = y[0]
        return R2 - max_r

    event.terminal = True
    event.direction = 1  # Only trigger when R2 crosses max_r from below
    event.name = name
    event.is_simulation_ending = True
    event.reason_code = "large_radius_event"
    event.reason_message = "Large radius reached (event)"
    return event


def make_velocity_runaway_event(v_max: float = MAX_VELOCITY_COLLAPSE,
                                 direction: str = "collapse",
                                 name: str = "velocity_runaway"):
    """
    Create event that triggers on extreme velocity magnitude.

    This catches runaway dynamics before the solver becomes numerically unstable.

    Parameters
    ----------
    v_max : float
        Maximum velocity magnitude (pc/Myr). Default 500 pc/Myr for collapse.
    direction : str
        "collapse" for inward (v2 < -v_max), "expansion" for outward (v2 > v_max),
        or "both" for either direction.
    name : str
        Name for identifying this event in results.

    Returns
    -------
    event : callable
        Event function for solve_ivp with terminal=True.
    """
    if direction == "collapse":
        def event(t, y):
            v2 = y[1]
            return v2 + v_max  # Triggers when v2 < -v_max
        event.direction = -1
        event.reason_message = "Collapse velocity runaway (event)"
    elif direction == "expansion":
        def event(t, y):
            v2 = y[1]
            return v_max - v2  # Triggers when v2 > v_max
        event.direction = -1
        event.reason_message = "Expansion velocity runaway (event)"
    else:  # both
        def event(t, y):
            v2 = y[1]
            return v_max - abs(v2)  # Triggers when |v2| > v_max
        event.direction = -1
        event.reason_message = "Velocity runaway (event)"

    event.terminal = True
    event.name = name
    event.is_simulation_ending = True
    event.reason_code = "velocity_runaway_event"
    return event


# =============================================================================
# Phase-Ending Event Factories
# =============================================================================

def make_cloud_boundary_event(rCloud: float, name: str = "cloud_boundary"):
    """
    Create event that triggers when R2 reaches cloud edge.

    Used in energy phase to detect when shell reaches cloud boundary,
    triggering transition to implicit phase.

    Parameters
    ----------
    rCloud : float
        Cloud radius (pc).
    name : str
        Name for identifying this event in results.

    Returns
    -------
    event : callable
        Event function for solve_ivp with terminal=True, direction=1.
    """
    def event(t, y):
        R2 = y[0]
        return R2 - rCloud

    event.terminal = True
    event.direction = 1  # Only trigger when R2 crosses rCloud from below
    event.name = name
    event.is_simulation_ending = False  # Phase ending, not simulation ending
    event.reason_code = "cloud_boundary"
    event.reason_message = "Shell reached cloud boundary"
    return event


def make_energy_floor_event(energy_floor: float, y_index: int = 2,
                            name: str = "energy_floor"):
    """
    Create event that triggers when bubble energy falls below threshold.

    Used in transition phase to detect when thermal energy is negligible,
    triggering transition to momentum phase.

    Parameters
    ----------
    energy_floor : float
        Minimum energy threshold (erg). Below this, transition to momentum phase.
    y_index : int
        Index of Eb in state vector y. Default 2 for [R2, v2, Eb, ...].
    name : str
        Name for identifying this event in results.

    Returns
    -------
    event : callable
        Event function for solve_ivp with terminal=True, direction=-1.
    """
    def event(t, y):
        Eb = y[y_index]
        return Eb - energy_floor

    event.terminal = True
    event.direction = -1  # Only trigger when Eb crosses threshold from above
    event.name = name
    event.is_simulation_ending = False  # Phase ending, not simulation ending
    event.reason_code = "energy_floor"
    event.reason_message = "Bubble energy below threshold"
    return event


def make_velocity_sign_event(y_index: int = 1, name: str = "velocity_sign"):
    """
    Create event that triggers when velocity changes sign.

    Used to detect collapse onset (v2 going from positive to negative).

    Parameters
    ----------
    y_index : int
        Index of v2 in state vector y. Default 1 for [R2, v2, ...].
    name : str
        Name for identifying this event in results.

    Returns
    -------
    event : callable
        Event function for solve_ivp with terminal=False (monitoring only),
        direction=-1 (only triggers on positive-to-negative crossing).
    """
    def event(t, y):
        v2 = y[y_index]
        return v2

    event.terminal = False  # Non-terminal by default - just records the crossing
    event.direction = -1  # Only trigger when v2 goes positive -> negative
    event.name = name
    event.is_simulation_ending = False
    event.reason_code = "velocity_sign_change"
    event.reason_message = "Velocity changed sign (collapse onset)"
    return event


def make_cooling_balance_event(threshold: float = 0.05, name: str = "cooling_balance"):
    """
    Create event factory for cooling balance detection.

    Returns a factory that creates the actual event given current Lgain/Lloss.
    This is needed because Lgain/Lloss change each segment.

    NOTE: This event requires segment-level checking since Lgain/Lloss
    are computed during segment setup, not available to solve_ivp.

    Parameters
    ----------
    threshold : float
        Ratio threshold. Event triggers when (Lgain - Lloss) / Lgain < threshold.
    name : str
        Name for identifying this event.

    Returns
    -------
    factory : callable
        Function that takes (Lgain, Lloss) and returns an event function.
    """
    def factory(Lgain: float, Lloss: float):
        def event(t, y):
            if Lgain <= 0:
                return 1.0  # No event if no gain
            ratio = (Lgain - Lloss) / Lgain
            return ratio - threshold

        event.terminal = True
        event.direction = -1  # Trigger when ratio falls below threshold
        event.name = name
        event.is_simulation_ending = False  # Phase ending
        event.reason_code = "cooling_balance"
        event.reason_message = f"Cooling balance reached (L_cool ~ L_gain)"
        return event

    return factory


# =============================================================================
# Event Result Checking
# =============================================================================

def check_event_termination(sol, events: List[Callable]) -> EventResult:
    """
    Check solve_ivp solution for event termination.

    Parameters
    ----------
    sol : OdeResult
        Solution object from solve_ivp.
    events : list of callable
        List of event functions passed to solve_ivp.

    Returns
    -------
    result : EventResult
        Container with event detection results.
    """
    if sol.t_events is None:
        return EventResult(
            triggered=False,
            name="",
            index=-1,
            t=np.nan,
            y=np.array([]),
            is_simulation_ending=False,
            reason_code="",
            reason_message=""
        )

    # Check each event
    for i, (t_ev, y_ev) in enumerate(zip(sol.t_events, sol.y_events)):
        if len(t_ev) > 0:
            event = events[i]
            return EventResult(
                triggered=True,
                name=getattr(event, 'name', f'event_{i}'),
                index=i,
                t=float(t_ev[0]),
                y=y_ev[0].copy(),
                is_simulation_ending=getattr(event, 'is_simulation_ending', True),
                reason_code=getattr(event, 'reason_code', 'unknown_event'),
                reason_message=getattr(event, 'reason_message', 'Event triggered')
            )

    return EventResult(
        triggered=False,
        name="",
        index=-1,
        t=np.nan,
        y=np.array([]),
        is_simulation_ending=False,
        reason_code="",
        reason_message=""
    )


# =============================================================================
# Event List Builders for Each Phase
# =============================================================================

def build_energy_phase_events(params) -> List[Callable]:
    """
    Build event list for energy phase.

    Events:
    - cloud_boundary: R2 > rCloud (phase ending)
    - min_radius: R2 < safety threshold (simulation ending)
    - velocity_runaway: |v2| too large (simulation ending)

    Parameters
    ----------
    params : dict
        Parameter dictionary with rCloud, coll_r, etc.

    Returns
    -------
    events : list
        List of event functions for solve_ivp.
    """
    rCloud = params['rCloud'].value
    coll_r = params['coll_r'].value

    min_r = max(coll_r * MIN_RADIUS_FACTOR, MIN_RADIUS_SAFETY)

    events = [
        make_cloud_boundary_event(rCloud),
        make_min_radius_event(min_r),
        make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction="collapse"),
    ]

    logger.debug(f"Energy phase events: cloud_boundary={rCloud:.2f} pc, "
                 f"min_radius={min_r:.4f} pc")
    return events


def build_implicit_phase_events(params) -> Tuple[List[Callable], Callable]:
    """
    Build event list for implicit (cooling) phase.

    Events:
    - velocity_sign: v2 crosses zero (monitoring, non-terminal)
    - min_radius: R2 < safety threshold (simulation ending)
    - max_radius: R2 > stop_r (simulation ending)
    - velocity_runaway: |v2| too large (simulation ending)

    Also returns cooling_balance factory for segment-level checking.

    Parameters
    ----------
    params : dict
        Parameter dictionary with coll_r, stop_r, etc.

    Returns
    -------
    events : list
        List of event functions for solve_ivp.
    cooling_balance_factory : callable
        Factory function to create cooling_balance event for each segment.
    """
    coll_r = params['coll_r'].value
    stop_r = params['stop_r'].value

    min_r = max(coll_r * MIN_RADIUS_FACTOR, MIN_RADIUS_SAFETY)

    events = [
        make_velocity_sign_event(),
        make_min_radius_event(min_r),
        make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction="collapse"),
    ]

    # Only add max_radius event if stop_r is set
    if stop_r is not None and stop_r > 0:
        events.append(make_max_radius_event(stop_r))

    cooling_factory = make_cooling_balance_event(threshold=0.05)

    logger.debug(f"Implicit phase events: min_radius={min_r:.4f} pc, "
                 f"stop_r={stop_r}")
    return events, cooling_factory


def build_transition_phase_events(params, energy_floor: float = 1e3) -> List[Callable]:
    """
    Build event list for transition phase.

    Events:
    - energy_floor: Eb < threshold (phase ending -> momentum)
    - min_radius: R2 < safety threshold (simulation ending)
    - max_radius: R2 > stop_r (simulation ending)
    - velocity_runaway: |v2| too large (simulation ending)

    Parameters
    ----------
    params : dict
        Parameter dictionary with coll_r, stop_r, etc.
    energy_floor : float
        Minimum energy threshold (erg). Default 1e3.

    Returns
    -------
    events : list
        List of event functions for solve_ivp.
    """
    coll_r = params['coll_r'].value
    stop_r = params['stop_r'].value

    min_r = max(coll_r * MIN_RADIUS_FACTOR, MIN_RADIUS_SAFETY)

    events = [
        make_energy_floor_event(energy_floor, y_index=2),
        make_min_radius_event(min_r),
        make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction="collapse"),
    ]

    # Only add max_radius event if stop_r is set
    if stop_r is not None and stop_r > 0:
        events.append(make_max_radius_event(stop_r))

    logger.debug(f"Transition phase events: energy_floor={energy_floor:.2e} erg, "
                 f"min_radius={min_r:.4f} pc")
    return events


def build_momentum_phase_events(params) -> List[Callable]:
    """
    Build event list for momentum phase.

    Events:
    - min_radius: R2 < safety threshold (simulation ending)
    - max_radius: R2 > stop_r (simulation ending)
    - velocity_runaway: |v2| too large (simulation ending)

    Parameters
    ----------
    params : dict
        Parameter dictionary with coll_r, stop_r, etc.

    Returns
    -------
    events : list
        List of event functions for solve_ivp.
    """
    coll_r = params['coll_r'].value
    stop_r = params['stop_r'].value

    min_r = max(coll_r * MIN_RADIUS_FACTOR, MIN_RADIUS_SAFETY)

    events = [
        make_min_radius_event(min_r),
        make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction="collapse"),
    ]

    # Only add max_radius event if stop_r is set
    if stop_r is not None and stop_r > 0:
        events.append(make_max_radius_event(stop_r))

    logger.debug(f"Momentum phase events: min_radius={min_r:.4f} pc, "
                 f"stop_r={stop_r}")
    return events


# =============================================================================
# Helper for Applying Event Results to Params
# =============================================================================

def apply_event_result(params, result: EventResult, t: float, y: np.ndarray,
                       state_keys: List[str] = ['R2', 'v2']) -> None:
    """
    Apply event result to params dictionary.

    Updates params with final state and termination info if event triggered.

    Parameters
    ----------
    params : dict
        Parameter dictionary to update.
    result : EventResult
        Event detection result from check_event_termination.
    t : float
        Time at event.
    y : np.ndarray
        State vector at event.
    state_keys : list of str
        Keys for state variables in order matching y. Default ['R2', 'v2'].
    """
    if not result.triggered:
        return

    # Update time
    params['t_now'].value = t

    # Update state variables
    for i, key in enumerate(state_keys):
        if i < len(y) and key in params:
            params[key].value = float(y[i])

    # Set termination info if simulation-ending
    if result.is_simulation_ending:
        params['SimulationEndReason'].value = result.reason_message
        params['EndSimulationDirectly'].value = True

        # Mark collapse if it's a collapse-related event
        if 'radius' in result.reason_code.lower() or 'collapse' in result.reason_code.lower():
            if 'isCollapse' in params:
                params['isCollapse'].value = True

    logger.info(f"Event '{result.name}' applied: {result.reason_message}")
