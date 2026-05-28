#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 23:14:53 2025

@author: Jia Wei Teh

Evaluate SPS feedback values at a given time and update the params dictionary.
"""

from dataclasses import dataclass
from typing import Iterator, Any
from src._input.dictionary import updateDict
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)


@dataclass
class SPSFeedback:
    """
    Container for SPS stellar feedback parameters at a single time.

    Supports both attribute access and unpacking for backward compatibility:
        # New style (recommended):
        feedback = get_current_sps_feedback(t, params)
        print(feedback.Lbol, feedback.Qi)

        # Old style (still works):
        (t, Qi, Li, Ln, Lbol, ...) = get_current_sps_feedback(t, params)

    Attributes
    ----------
    All luminosities are in code units [Msun·pc²/Myr³] (multiply by
    INV_CONV.L_au2cgs to get erg/s); raw cgs values are converted to AU
    at load time in ``read_sps.py``.

    t : float
        Current time [Myr]
    Qi : float
        Ionizing photon rate [s⁻¹]
    Li : float
        Ionizing luminosity [Msun·pc²/Myr³]
    Ln : float
        Non-ionizing luminosity [Msun·pc²/Myr³]
    Lbol : float
        Bolometric luminosity [Msun·pc²/Myr³]
    Lmech_W : float
        Wind mechanical luminosity [Msun·pc²/Myr³]
    Lmech_SN : float
        SN mechanical luminosity [Msun·pc²/Myr³]
    Lmech_total : float
        Total mechanical luminosity [Msun·pc²/Myr³]
    pdot_W : float
        Wind momentum rate [M_sun·pc/Myr²]
    pdot_SN : float
        SN momentum rate [M_sun·pc/Myr²]
    pdot_total : float
        Total momentum rate [M_sun·pc/Myr²]
    pdotdot_total : float
        Time derivative of total momentum rate
    v_mech_total : float
        Wind velocity [pc/Myr]
    """
    t: float
    Qi: float
    Li: float
    Ln: float
    Lbol: float
    Lmech_W: float
    Lmech_SN: float
    Lmech_total: float
    pdot_W: float
    pdot_SN: float
    pdot_total: float
    pdotdot_total: float
    v_mech_total: float

    def __iter__(self) -> Iterator[float]:
        """Allow unpacking: t, Qi, Li, ... = feedback"""
        return iter([
            self.t, self.Qi, self.Li, self.Ln, self.Lbol,
            self.Lmech_W, self.Lmech_SN, self.Lmech_total,
            self.pdot_W, self.pdot_SN, self.pdot_total,
            self.pdotdot_total, self.v_mech_total
        ])

    def __getitem__(self, idx: int) -> float:
        """Allow indexing: feedback[0] == feedback.t"""
        return list(self)[idx]

    def __len__(self) -> int:
        """Return number of fields."""
        return 13


def get_current_sps_feedback(t, params) -> SPSFeedback:
    """
    Get stellar feedback parameters at time t from the SPS interpolators.

    Interpolates the SPS feedback time-series at the given time and returns
    an SPSFeedback dataclass. The interpolators are built by
    `read_sps.get_interpolation` from data loaded via `read_sps.read_sps`;
    both branches (legacy SB99 positional and user-defined sps_path) feed
    the same `params['sps_f']` dict.

    Parameters
    ----------
    t : float
        Current time [Myr]
    params : DescribedDict
        Global parameters dictionary containing the `sps_f` interpolators.

    Returns
    -------
    SPSFeedback
        Dataclass containing all feedback parameters. Supports both:
        - Attribute access: feedback.Lbol, feedback.Qi, etc.
        - Unpacking: (t, Qi, Li, ...) = get_current_sps_feedback(t, params)

        Fields (luminosities in AU [Msun·pc²/Myr³]; convert to erg/s
        with INV_CONV.L_au2cgs):
        - t : float, current time [Myr]
        - Qi : float, ionizing photon rate [s⁻¹]
        - Li : float, ionizing luminosity [Msun·pc²/Myr³]
        - Ln : float, non-ionizing luminosity [Msun·pc²/Myr³]
        - Lbol : float, bolometric luminosity [Msun·pc²/Myr³]
        - Lmech_W : float, wind mechanical luminosity [Msun·pc²/Myr³]
        - Lmech_SN : float, SN mechanical luminosity [Msun·pc²/Myr³]
        - Lmech_total : float, total mechanical luminosity [Msun·pc²/Myr³]
        - pdot_W : float, wind momentum rate [M_sun·pc/Myr²]
        - pdot_SN : float, SN momentum rate [M_sun·pc/Myr²]
        - pdot_total : float, total momentum rate [M_sun·pc/Myr²]
        - pdotdot_total : float, time derivative of pdot_total
        - v_mech_total : float, wind velocity [pc/Myr]

    Notes
    -----
    Wind velocity (v_mech_total) uses total quantities:
      v_mech_total = 2 * Lmech_total / pdot_total
    This is an effective velocity such that pRam = L/(2*pi*r^2*v) yields
    the correct total ram pressure: pdot_total / (4*pi*r^2).

    Naming convention:
    - Wind components: _W suffix (Lmech_W, pdot_W, fLmech_W, fpdot_W)
    - SN components: _SN suffix (Lmech_SN, pdot_SN, fLmech_SN, fpdot_SN)
    - Total components: _total suffix (Lmech_total, pdot_total)
    """

    sps_f = params['sps_f'].value

    t_min = float(sps_f['fQi'].x[0])
    t_max = float(sps_f['fQi'].x[-1])

    if not (t_min <= t <= t_max):
        raise ValueError(
            f"Time t={t:.6f} outside SPS range [{t_min:.6f}, {t_max:.6f}] Myr"
        )

    # Interpolate all raw SPS values. The interpolators were built in
    # read_sps.py from arrays already converted to code units (AU);
    # luminosities here are [Msun*pc^2/Myr^3], not erg/s.
    Qi = sps_f['fQi'](t)[()]                    # Ionizing photon rate [s⁻¹]
    Li = sps_f['fLi'](t)[()]                    # Ionizing luminosity
    Ln = sps_f['fLn'](t)[()]                    # Non-ionizing luminosity
    Lbol = sps_f['fLbol'](t)[()]                # Bolometric luminosity

    Lmech_W = sps_f['fLmech_W'](t)[()]          # Wind mechanical luminosity
    Lmech_SN = sps_f['fLmech_SN'](t)[()]        # SN mechanical luminosity
    Lmech_total = sps_f['fLmech_total'](t)[()]  # Total mechanical luminosity

    pdot_W = sps_f['fpdot_W'](t)[()]            # Wind momentum rate
    pdot_SN = sps_f['fpdot_SN'](t)[()]          # SN momentum rate
    pdot_total = sps_f['fpdot_total'](t)[()]    # Total momentum rate

    # =========================================================================
    # DERIVED VALUES
    # =========================================================================
    # Effective mechanical velocity: v = 2L/pdot gives P_ram = pdot_total/(4*pi*r^2)
    v_mech_total = (2. * Lmech_total / pdot_total)[()]

    # Numerical derivative of total momentum rate for time evolution
    dt = 1e-9  # Myr (small timestep for derivative)
    pdotdot_total = (sps_f['fpdot_total'](t + dt)[()] - sps_f['fpdot_total'](t - dt)[()]) / (2.0 * dt)

    # Return SPSFeedback dataclass (supports both attribute access and unpacking)
    return SPSFeedback(
        t=t,
        Qi=Qi,
        Li=Li,
        Ln=Ln,
        Lbol=Lbol,
        Lmech_W=Lmech_W,
        Lmech_SN=Lmech_SN,
        Lmech_total=Lmech_total,
        pdot_W=pdot_W,
        pdot_SN=pdot_SN,
        pdot_total=pdot_total,
        pdotdot_total=pdotdot_total,
        v_mech_total=v_mech_total,
    )
