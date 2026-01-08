#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/phase1_energy/run_energy_phase.py

Purpose
-------
Advance the *energy-driven* (Weaver-like) phase of the bubble/shell evolution.

This routine **mutates** the passed-in ``params`` (a DescribedDict-like container)
by updating the *current* state keys (e.g. ``t_now, R2, v2, Eb, T0``) and by
appending coarse history arrays (e.g. ``array_t_now``).

Why this file exists
--------------------
The original implementation used a fixed-step forward Euler loop because the RHS
(energy_phase_ODEs.get_ODE_Edot) was written with *side effects* on ``params``.
Adaptive ODE solvers (SciPy) evaluate the RHS multiple times per step and may
repeat evaluations at the same time, which corrupts a side-effecting dictionary.

This rewrite keeps the convenient dictionary-driven code **and** enables robust
adaptive ODE integration by:

1) Treating the ODE state as a plain vector ``y = [R2, v2, Eb, T0]``.
2) Calling the RHS through a lightweight *sandbox* view that prevents accidental
   mutation of the real ``params`` during solver trial evaluations.
3) **Committing** results to ``params`` only at monotonic output times.

Notes
-----
- Units are assumed to be the project internal units: [Msun, pc, Myr].
- This file intentionally avoids heavy deep-copies inside the RHS. If you must
  isolate RHS side effects, use the sandbox below rather than deepcopy(params).
"""

import copy
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.integrate import solve_ivp

import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.cloud_properties.mass_profile as mass_profile
import src.cooling.non_CIE.read_cloudy as non_CIE
import src.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
import src.bubble_structure.bubble_luminosity as bubble_luminosity
import src._functions.operations as operations
from src._input.dictionary import updateDict
from src.sb99.update_feedback import get_currentSB99feedback


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_value(params: Mapping, key: str, default):
    """Return params[key].value if present, else default."""
    try:
        return params[key].value
    except Exception:
        return default


def _clone_item(item):
    """Best-effort clone of a DescribedItem-like object.

    We copy metadata fields when present and shallow/deep-copy values only when
    they are obviously mutable. This is intentionally conservative to avoid
    materializing large HDF5-backed arrays.
    """
    cls = type(item)
    # Value extraction (DescribedItem implements .value)
    try:
        val = item.value
    except Exception:
        val = item

    # Copy value (avoid deepcopy for callables / interpolators / proxies)
    if isinstance(val, np.ndarray):
        val2 = val.copy()
    elif isinstance(val, (list, dict)):
        val2 = copy.deepcopy(val)
    else:
        val2 = val

    # Reconstruct if it looks like DescribedItem
    try:
        return cls(
            val2,
            info=getattr(item, "info", None),
            ori_units=getattr(item, "ori_units", None),
            exclude_from_snapshot=getattr(item, "exclude_from_snapshot", False),
            isPersistent=getattr(item, "isPersistent", False),
        )
    except Exception:
        # Fallback: shallow copy the object, then overwrite .value if possible
        out = copy.copy(item)
        try:
            out.value = val2
        except Exception:
            pass
        return out


class ParamsSandbox(MutableMapping):
    """A lightweight overlay that prevents accidental mutation of the real params.

    - Reads fall back to the underlying params.
    - Keys in ``mutable_keys`` are *cloned* on first access; mutations affect only
      the clone, not the underlying params.
    - If ``sandbox_all=True``, every accessed key is cloned (safer, slower).
    """

    def __init__(
        self,
        base_params: Mapping,
        *,
        mutable_keys: Optional[Iterable[str]] = None,
        sandbox_all: bool = False,
    ):
        self._base = base_params
        self._local = {}
        self._mutable_keys = set(mutable_keys or [])
        self._sandbox_all = bool(sandbox_all)

    def __getitem__(self, key: str):
        if key in self._local:
            return self._local[key]
        if self._sandbox_all or key in self._mutable_keys:
            item = _clone_item(self._base[key])
            self._local[key] = item
            return item
        return self._base[key]

    def __setitem__(self, key: str, value) -> None:
        self._local[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._local:
            del self._local[key]
        else:
            raise KeyError(key)

    def __iter__(self):
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)


def _append_array(params, key: str, values: Sequence[float]) -> None:
    """Append values to params[key].value if the key exists."""
    if key not in params:
        return
    arr = np.asarray(params[key].value)
    vals = np.asarray(values, dtype=float)
    params[key].value = np.concatenate([arr, vals])


@dataclass
class EnergyPhaseConfig:
    """Tunable integration controls."""

    t_end: float = 3e-3
    cooling_update_interval: float = 5e-2
    dt_chunk: float = 3e-5
    rtol: float = 1e-6
    atol: Tuple[float, float, float, float] = (1e-10, 1e-10, 1e-10, 1e-10)
    method: str = "Radau"


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def run_energy(params) -> None:
    """Advance the energy-driven phase in-place."""

    # Current state
    t_now = float(params["t_now"])
    R2 = float(params["R2"])
    v2 = float(params["v2"])
    Eb = float(params["Eb"])
    T0 = float(params["T0"])

    # Cloud / shell temperatures
    rCloud = float(params["rCloud"])
    t_neu = float(params["TShell_neu"])
    t_ion = float(params["TShell_ion"])

    # Controls (prefer params overrides if available)
    cfg = EnergyPhaseConfig(
        t_end=float(_get_value(params, "tEnd_energy", _get_value(params, "t_end_energy", 3e-3))),
        cooling_update_interval=float(_get_value(params, "cooling_update_interval", 5e-2)),
        dt_chunk=float(_get_value(params, "dt_energy_chunk", 3e-5)),
        rtol=float(_get_value(params, "ode_rtol_energy", 1e-6)),
        atol=tuple(_get_value(params, "ode_atol_energy", (1e-10, 1e-10, 1e-10, 1e-10))),
        method=str(_get_value(params, "ode_method_energy", "Radau")),
    )

    # Initial feedback / derived quantities
    Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot = get_currentSB99feedback(t_now, params)

    # Inner shock radius (guard bracketing)
    a = max(1e-6, 1e-3 * R2)
    b = max(a * 1.01, R2)
    try:
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, a=a, b=b, args=([LWind, Eb, vWind, R2]))
    except Exception:
        R1 = a

    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, float(params["gamma_adia"]))

    # Shell mass at current radius
    try:
        Msh0 = float(mass_profile.get_mass_profile(np.array([R2]), params, return_mdot=False)[0])
    except Exception:
        Msh0 = float(_get_value(params, "shell_mass", 0.0))

    updateDict(
        params,
        ["R1", "R2", "v2", "Eb", "T0", "t_now", "Pb", "shell_mass"],
        [R1, R2, v2, Eb, T0, t_now, Pb, Msh0],
    )

    loop_count = 0
    continue_weaver = True

    t_prev_cool = float(_get_value(params, "t_previousCoolingUpdate", t_now - 1e99))

    rhs_mutable_keys = {
        "t_now", "t_next", "R2", "v2", "Eb", "T0", "R1", "Pb", "shell_mass",
    }

    def event_reach_cloud(t, y):
        return y[0] - rCloud

    event_reach_cloud.terminal = True
    event_reach_cloud.direction = 1.0

    while (R2 < rCloud) and ((cfg.t_end - t_now) > 1e-6) and continue_weaver:
        # refresh cooling structure (infrequent)
        if (t_now - t_prev_cool) >= cfg.cooling_update_interval:
            cooling_nonCIE, heating_nonCIE, netcool_interp = non_CIE.get_coolingStructure(params)
            params["cStruc_cooling_nonCIE"].value = cooling_nonCIE
            params["cStruc_heating_nonCIE"].value = heating_nonCIE
            params["cStruc_net_nonCIE_interpolation"].value = netcool_interp
            t_prev_cool = t_now
            if "t_previousCoolingUpdate" in params:
                params["t_previousCoolingUpdate"].value = t_prev_cool

        # refresh bubble/shell derived properties at the chunk start (skip first)
        if loop_count > 0:
            _ = bubble_luminosity.get_bubbleproperties(params)
            if "bubble_T_rgoal" in params:
                params["T0"].value = float(params["bubble_T_rgoal"])
                T0 = float(params["T0"])

        t_chunk_end = min(cfg.t_end, t_now + cfg.dt_chunk)
        y0 = np.array([R2, v2, Eb, T0], dtype=float)

        def rhs(t, y):
            p = ParamsSandbox(params, mutable_keys=rhs_mutable_keys, sandbox_all=False)
            p["t_now"].value = float(t)
            p["t_next"].value = float(t)

            p["R2"].value = float(y[0])
            p["v2"].value = float(y[1])
            p["Eb"].value = float(y[2])
            p["T0"].value = float(y[3])

            rd, vd, Ed, Td = energy_phase_ODEs.get_ODE_Edot(y, t, p)
            return np.array([rd, vd, Ed, Td], dtype=float)

        sol = solve_ivp(
            rhs,
            t_span=(t_now, t_chunk_end),
            y0=y0,
            method=cfg.method,
            rtol=cfg.rtol,
            atol=np.asarray(cfg.atol, dtype=float),
            events=[event_reach_cloud],
            max_step=cfg.dt_chunk,
        )

        if sol.status < 0:
            raise RuntimeError(f"Energy-phase ODE solver failed: {sol.message}")

        t_now = float(sol.t[-1])
        R2, v2, Eb, T0 = (float(sol.y[0, -1]), float(sol.y[1, -1]), float(sol.y[2, -1]), float(sol.y[3, -1]))

        R2 = max(R2, 0.0)
        Eb = max(Eb, 0.0)

        # update derived terms at committed state
        Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot = get_currentSB99feedback(t_now, params)

        a = max(1e-6, 1e-3 * R2)
        b = max(a * 1.01, R2)
        try:
            R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, a=a, b=b, args=([LWind, Eb, vWind, R2]))
        except Exception:
            R1 = a

        Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, float(params["gamma_adia"]))

        try:
            Msh0 = float(mass_profile.get_mass_profile(np.array([R2]), params, return_mdot=False)[0])
        except Exception:
            Msh0 = float(_get_value(params, "shell_mass", 0.0))

        updateDict(
            params,
            ["R1", "R2", "v2", "Eb", "T0", "t_now", "Pb", "shell_mass"],
            [R1, R2, v2, Eb, T0, t_now, Pb, Msh0],
        )

        f_abs_ion = float(_get_value(params, "shell_fAbsorbedIon", 1.0))
        T_shell = t_neu if f_abs_ion < 0.99 else t_ion
        params["c_sound"].value = operations.get_soundspeed(T_shell, params)

        _append_array(params, "array_t_now", [t_now])
        _append_array(params, "array_R2", [R2])
        _append_array(params, "array_R1", [R1])
        _append_array(params, "array_v2", [v2])
        _append_array(params, "array_T0", [T0])
        _append_array(params, "array_mShell", [Msh0])

        if hasattr(params, "save_snapshot"):
            params.save_snapshot()

        loop_count += 1
        if R2 >= rCloud:
            break

    return
