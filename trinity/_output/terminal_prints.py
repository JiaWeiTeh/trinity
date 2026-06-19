#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:39:35 2023

@author: Jia Wei Teh

This script contains functions that handle printing information in the terminal.
Uses logging for consistent output formatting across the TRINITY simulation.
"""

import logging
import math

import trinity._functions.unit_conversions as cvt

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Segments between INFO progress heartbeats in the long phase loops.
HEARTBEAT_EVERY = 50


def _format_banner(message: str, width: int = 50) -> str:
    """Format a message as a banner with dashes."""
    separator = "-" * width
    return f"\n{separator}\n{message}\n{separator}"


def bubble():
    """Log message for bubble structure calculation."""
    logger.info(_format_banner("Calculating bubble structure"))


def phase0(time):
    """Log initialization message with timestamp."""
    logger.info(_format_banner(f"{time}: Initialising bubble"))


def phase(string: str):
    """Log phase transition message."""
    logger.info(_format_banner(string))


def shell():
    """Log message for shell structure calculation."""
    logger.info(_format_banner("Calculating shell structure"))


class cprint:
    """
    A class that deals with printing with colours in terminal.

    Example usage:
        print(f'{cprint.BOLD}This text is bolded{cprint.END}  but this isnt.')

    Attributes
    ----------
    BOLD : str
        ANSI escape code for bold cyan text (used for file saves)
    SAVE : str
        Alias for BOLD
    FILE : str
        Alias for BOLD
    LINK : str
        ANSI escape code for green text (used for links)
    WARN : str
        ANSI escape code for bold blue text (used for warnings)
    BLINK : str
        ANSI escape code for blinking text
    FAIL : str
        ANSI escape code for bold red text (used for errors)
    END : str
        ANSI escape code to reset all formatting
    """

    # Symbol prefix for file operations
    symbol = '\u27B3 '

    # Bolded text to signal that a file is being saved
    BOLD = symbol + '\033[1m\033[96m'
    # aliases
    SAVE = BOLD
    FILE = BOLD

    # Link (green)
    LINK = '\033[32m'

    # Warning message, but code runs still (bold blue)
    WARN = '\033[1m\033[94m'

    # Blink
    BLINK = '\033[5m'

    # FAIL (bold red)
    FAIL = '\033[1m\033[91m'

    # END and clear all colours. This should be included in the end of every operation.
    END = '\033[0m'


# =============================================================================
# Logging helper functions
# =============================================================================

def log_file_saved(filepath: str, description: str = "File saved"):
    """Log a file save operation with distinctive formatting."""
    logger.info(f"{cprint.SAVE}{description}: {filepath}{cprint.END}")


def log_warning(message: str):
    """Log a warning message with distinctive formatting."""
    logger.warning(f"{cprint.WARN}{message}{cprint.END}")


def log_error(message: str):
    """Log an error message with distinctive formatting."""
    logger.error(f"{cprint.FAIL}{message}{cprint.END}")


# =============================================================================
# Physical-state log lines
# =============================================================================
# A single source of truth for the "where is the bubble now" snapshot that gets
# logged at phase entry/exit, the end-of-run report, and the progress heartbeat.
# Values are read straight from the central `params` (DescribedDict) and shown
# in conventional astro/CGS display units so a researcher can read trinity.log
# without doing unit conversions in their head.

# (display label, params key, internal->display factor, format spec, unit label)
_STATE_FIELDS = [
    ("t",      "t_now",      1.0,                ".6f", "Myr"),
    ("R2",     "R2",         1.0,                ".4f", "pc"),
    ("v2",     "v2",         cvt.v_au2kms,       ".4f", "km/s"),
    ("Eb",     "Eb",         cvt.E_au2cgs,       ".4e", "erg"),
    ("Pb",     "Pb",         cvt.Pb_au2_KcmInv,  ".4e", "K/cm3"),
    ("T0",     "T0",         1.0,                ".4e", "K"),
    ("R1",     "R1",         1.0,                ".4f", "pc"),
    ("Mshell", "shell_mass", 1.0,                ".4e", "Msun"),
]


def _phys(params, key, conv=1.0, fmt=".4e"):
    """Read params[key].value in display units; tolerant of missing/bad values.

    Returns 'n/a' if the key is absent or its value is None/non-numeric, and the
    literal 'nan'/'inf' if the value is non-finite. Eb=0 (momentum phase) is
    finite and renders normally.
    """
    item = params.get(key)
    value = getattr(item, "value", None)
    if value is None:
        return "n/a"
    try:
        val = float(value) * conv
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(val):
        return "nan" if math.isnan(val) else ("inf" if val > 0 else "-inf")
    return format(val, fmt)


def format_state(params, label=None, *, oneline=False):
    """Format the core bubble state as a log string.

    With ``oneline=False`` (default) returns a multi-line aligned block intended
    to be passed to a single ``logger.info`` call (one timestamp prefix, then the
    raw aligned body, same as the banners). With ``oneline=True`` returns one
    dense line for the progress heartbeat.
    """
    cells = [
        f"{lab} = {_phys(params, key, conv, fmt)} {unit}"
        for (lab, key, conv, fmt, unit) in _STATE_FIELDS
    ]
    if oneline:
        body = "  ".join(cells)
        return f"{label} | {body}" if label else body
    width = max(len(c) for c in cells)
    rows = [
        "  " + "  ".join(c.ljust(width) for c in cells[i:i + 3]).rstrip()
        for i in range(0, len(cells), 3)
    ]
    head = f"State [{label}]:" if label else "State:"
    return head + "\n" + "\n".join(rows)


def heartbeat(params, tag, segment, tmin, tmax):
    """Emit a throttled one-line progress heartbeat for a long-phase loop.

    Call unconditionally inside the outer segment loop (after the post-ODE state
    writes). It logs only every ``HEARTBEAT_EVERY``-th segment, so it never floods
    the log even though the loop runs hundreds of segments. ``tag`` names the phase
    (e.g. "1b implicit"); ``tmin``/``tmax`` bound the simulated-time progress bar.
    """
    if segment % HEARTBEAT_EVERY != 0:
        return
    if tmax is not None and tmax > tmin:
        t_now = getattr(params.get("t_now"), "value", tmax)
        prog = f"{(t_now - tmin) / (tmax - tmin) * 100:.1f}% of t {tmin:.4g}->{tmax:.4g} Myr"
    else:
        prog = f"t since {tmin:.4g} Myr"
    logger.info(f"[{tag}] seg {segment} ({prog}) | " + format_state(params, oneline=True))


def format_end_report(params):
    """One INFO block: the stopping fate in words, then the final-state block.

    Reads the numeric SimulationEndCode + verbatim SimulationEndReason that the
    phase runners set, so the actual fate (why the bubble stopped) is visible in
    trinity.log rather than only in metadata.json.
    """
    # Lazy import to avoid a module-load cycle (simulation_end imports unit conv).
    from trinity._output.simulation_end import SimulationEndCode

    code_value = getattr(params.get("SimulationEndCode"), "value", None)
    try:
        end = SimulationEndCode.from_code(int(code_value))
    except (TypeError, ValueError):
        end = SimulationEndCode.UNKNOWN
    reason = getattr(params.get("SimulationEndReason"), "value", None) or "unknown"

    if end.is_clean():
        headline = "Simulation ended"
    elif end.is_error():
        headline = "Simulation FAILED"
    else:
        headline = "Simulation ended (inspection required)"

    head = f"{headline}: {end.name} (code {end.code}) — {reason}"
    return head + "\n" + format_state(params, label="final state")


