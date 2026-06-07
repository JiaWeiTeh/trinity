#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``python -m trinity._output.show_run <run_dir>`` — human-readable summary
of a TRINITY output directory.

Replaces the ``cat simulationEnd.txt`` workflow.  Reads
``metadata.json`` (v3+ schema preferred; v1/v2 + text fallback
supported) and pretty-prints a curated subset of run-context +
termination + final-state.  Full JSON dump available via
``--json``.

Designed as a thin, pure-stdlib + numpy + INV_CONV diagnostic.  No
matplotlib, no plot infrastructure imports — fast enough for
shell-loop use in batch jobs::

    for d in outputs/sweep_*/*/; do
        python -m trinity._output.show_run --quiet "$d" || echo "BAD: $d"
    done

Exit codes
----------
* 0 — pretty-print succeeded (or, with ``--quiet``, the run is
  successful per ``output.is_successful_run``).
* 1 — run directory not found, or both ``metadata.json`` and
  ``simulationEnd.txt`` missing.
* 2..9 — with ``--quiet``, the run's own ``exit_code`` from the
  termination block (capped at 9 so it fits in POSIX).
"""

from __future__ import annotations

import argparse
import json
import sys as _sys
from pathlib import Path
from typing import Optional

# Repo-root sys.path shim so direct invocation works without
# ``pip install -e .``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from trinity._functions.unit_conversions import INV_CONV, Pb_au2_KcmInv
from trinity._output.run_constants import METADATA_FILENAME
from trinity._output.simulation_end import read_simulation_end
from trinity._output.trinity_reader import TrinityOutput, find_data_path


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_WIDTH = 60
_HR_HEAVY = "=" * _WIDTH
_HR_LIGHT = "-" * _WIDTH


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _status_line(termination: Optional[dict],
                 is_successful: Optional[bool]) -> str:
    """One-line "Status : ✓ SUCCESS  (outcome)" header."""
    if termination is None or is_successful is None:
        return "Status   : ? UNKNOWN  (no termination block — legacy or aborted run)"
    glyph = "✓" if is_successful else "✗"
    label = "SUCCESS" if is_successful else "ERROR"
    outcome = termination.get("outcome") or "unknown"
    return f"Status   : {glyph} {label}  ({outcome})"


def _fmt_or_na(value, fmt: str = ".4e", default: str = "n/a") -> str:
    """Format ``value`` with ``fmt`` or return ``default``."""
    if value is None:
        return default
    try:
        return format(value, fmt)
    except (TypeError, ValueError):
        return str(value)


def _cloud_section(md: dict) -> list[str]:
    """Render the 'Cloud' section from metadata.json's run-constants."""
    lines = [_HR_LIGHT, "Cloud", _HR_LIGHT]
    # mCluster is derived from mCloud * sfe; show it if both are present.
    mCluster = md.get("mCluster")
    if mCluster is None and "mCloud" in md and "sfe" in md:
        mCluster = md["mCloud"] * md["sfe"]
    # Each row carries a conv factor applied to numeric values for display.
    # nCore/nISM are stored internally in pc⁻³; show them in cm⁻³ (the input
    # unit) via ndens_au2cgs. conv is only applied in the numeric branch, so
    # string rows (dens_profile) keep conv=1.0 and the None guard runs first.
    rows = [
        ("mCloud",        md.get("mCloud"),  ".2e", "Msun", 1.0),
        ("nCore",         md.get("nCore"),   ".2e", "cm⁻³", INV_CONV.ndens_au2cgs),
        ("rCloud",        md.get("rCloud"),  ".3f", "pc",   1.0),
        ("rCore",         md.get("rCore"),   ".3f", "pc",   1.0),
        ("dens_profile",  md.get("dens_profile"), "", "",   1.0),
        ("densPL_alpha",  md.get("densPL_alpha"), ".2f", "", 1.0),
        ("nISM",          md.get("nISM"),    ".2e", "cm⁻³", INV_CONV.ndens_au2cgs),
        ("sfe",           md.get("sfe"),     ".4f", "",     1.0),
        ("mCluster",      mCluster,          ".2e", "Msun", 1.0),
        ("ZCloud",        md.get("ZCloud"),  ".2f", "Z☉",   1.0),
    ]
    for name, value, fmt, units, conv in rows:
        if value is None:
            continue
        formatted = _fmt_or_na(value * conv, fmt) if fmt else str(value)
        unit_str = f"  {units}" if units else ""
        lines.append(f"  {name:14s}: {formatted}{unit_str}")
    return lines


def _final_state_section(final_state: Optional[dict]) -> list[str]:
    """Render the 'Final state' section.

    Applies unit conversions for human reading (km/s for ``v2``,
    cm⁻³ for ``shell_nMax``) — same convention the legacy
    ``simulationEnd.txt`` used.  The internal value is shown in
    parentheses for traceability.
    """
    if not final_state:
        return [_HR_LIGHT, "Final state", _HR_LIGHT,
                "  (no final_state block — legacy or aborted run)"]
    t_now = final_state.get("t_now")
    t_label = f"  [t = {_fmt_or_na(t_now, '.3f')} Myr]"
    lines = [_HR_LIGHT, f"Final state{t_label}", _HR_LIGHT]

    rows: list[tuple[str, str]] = []
    # R2 in pc
    R2 = final_state.get("R2")
    if R2 is not None:
        rows.append(("R2", f"{_fmt_or_na(R2, '.3f')} pc"))
    # v2 in km/s with internal in parens
    v2 = final_state.get("v2")
    if v2 is not None:
        v2_kms = v2 * INV_CONV.v_au2kms
        rows.append(("v2", f"{_fmt_or_na(v2_kms, '.3f')} km/s    "
                          f"({_fmt_or_na(v2, '.3e')} pc/Myr)"))
    # shell_mass in Msun
    sm = final_state.get("shell_mass")
    if sm is not None:
        rows.append(("shell_mass", f"{_fmt_or_na(sm, '.3e')} Msun"))
    # shell_nMax in cm⁻³ with internal in parens
    snm = final_state.get("shell_nMax")
    if snm is not None:
        snm_cgs = snm * INV_CONV.ndens_au2cgs
        rows.append(("shell_nMax", f"{_fmt_or_na(snm_cgs, '.3e')} cm⁻³    "
                                  f"({_fmt_or_na(snm, '.3e')} pc⁻³)"))
    # Pb as P/k_B in K cm⁻³ with internal in parens (like v2 / shell_nMax)
    pb = final_state.get("Pb")
    if pb is not None:
        pb_KcmInv = pb * Pb_au2_KcmInv
        rows.append(("Pb", f"{_fmt_or_na(pb_KcmInv, '.3e')} K cm⁻³ (P/k_B)    "
                          f"({_fmt_or_na(pb, '.3e')} internal)"))
    # Eb in erg with internal in parens (like Pb / shell_nMax / v2)
    eb = final_state.get("Eb")
    if eb is not None:
        eb_cgs = eb * INV_CONV.E_au2cgs
        rows.append(("Eb", f"{_fmt_or_na(eb_cgs, '.3e')} erg    "
                          f"({_fmt_or_na(eb, '.3e')} internal)"))
    # T0, bubble_Tavg (internal units)
    for key, fmt, units in [
        ("T0", ".3e", "K"),
        ("bubble_Tavg", ".3e", "K"),
    ]:
        val = final_state.get(key)
        if val is not None:
            rows.append((key, f"{_fmt_or_na(val, fmt)} {units}"))
    # Current phase
    phase = final_state.get("current_phase")
    if phase is not None:
        rows.append(("phase", str(phase)))
    # Collapse / dissolved booleans
    if "isCollapse" in final_state:
        rows.append(("collapsed",
                     "yes" if final_state["isCollapse"] else "no"))
    if "isDissolved" in final_state:
        rows.append(("dissolved",
                     "yes" if final_state["isDissolved"] else "no"))

    for name, val in rows:
        lines.append(f"  {name:14s}: {val}")
    return lines


def _resolve_run_status(run_dir: Path) -> dict:
    """
    Gather all the bits ``format_run_summary`` and ``main`` need.

    Walks the same fallback chain (v3 metadata block → v1/v2 metadata
    only → legacy ``simulationEnd.txt`` text-parse) and returns
    everything in one dict so the formatter and the ``--quiet`` exit
    path don't duplicate file reads.

    Returns a dict with keys ``metadata``, ``termination``,
    ``final_state``, ``termination_debug``, ``is_successful``,
    ``model_name`` — all values may be ``None`` if the relevant source
    was absent.
    """
    md: dict = {}
    termination: Optional[dict] = None
    final_state: Optional[dict] = None
    termination_debug: Optional[dict] = None
    is_successful: Optional[bool] = None
    model_name = run_dir.name

    # Try the structured path via TrinityOutput first (v3 schema gives
    # everything we need from metadata.json without text parsing).
    try:
        jsonl_path = find_data_path(run_dir)
        output = TrinityOutput.open(jsonl_path)
        md = output.metadata
        termination = output.termination
        final_state = output.final_state
        termination_debug = output.termination_debug
        is_successful = output.is_successful_run
        if md.get("model_name"):
            model_name = md["model_name"]
    except FileNotFoundError:
        # No dictionary.jsonl — try metadata.json directly
        metadata_path = run_dir / METADATA_FILENAME
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    md = json.load(f)
                termination = md.get("termination")
                final_state = md.get("final_state")
                td_block = md.get("termination_debug")
                termination_debug = td_block if isinstance(td_block, dict) else None
                if md.get("model_name"):
                    model_name = md["model_name"]
                if isinstance(termination, dict):
                    ec = termination.get("exit_code")
                    if ec is not None:
                        try:
                            is_successful = 0 <= int(ec) <= 9
                        except (TypeError, ValueError):
                            is_successful = None
            except (json.JSONDecodeError, OSError):
                pass

    # Legacy fallback: if we still don't have termination, try the
    # text file.  ``read_simulation_end`` itself prefers the JSON
    # block; passing through here covers pre-Phase-2 runs.
    if termination is None:
        legacy = read_simulation_end(str(run_dir))
        if legacy is not None:
            termination = {
                "exit_code": legacy.get("exit_code"),
                "outcome": legacy.get("outcome"),
                "detail": legacy.get("detail"),
                "timestamp": legacy.get("timestamp"),
                "model_name": legacy.get("model"),
            }
            ec = termination["exit_code"]
            if ec is not None:
                try:
                    is_successful = 0 <= int(ec) <= 9
                except (TypeError, ValueError):
                    is_successful = None
            if termination["model_name"]:
                model_name = termination["model_name"]

    return {
        "metadata": md,
        "termination": termination,
        "final_state": final_state,
        "termination_debug": termination_debug,
        "is_successful": is_successful,
        "model_name": model_name,
    }


def _termination_debug_section(td: Optional[dict]) -> list[str]:
    """Render only the actionable bits of the termination_debug block.

    Shows flagged comparisons, NaN/Inf inventory, and failing sanity
    checks.  A clean run with no warnings collapses to a single line
    confirming the diagnostic ran.
    """
    if not td:
        return []
    lines = [_HR_LIGHT, "Termination diagnostics", _HR_LIGHT]
    reason = td.get("reason")
    if reason:
        lines.append(f"  reason        : {reason}")
    warnings = td.get("warnings") or []
    invalid = td.get("invalid_values") or {}
    nans = invalid.get("nan") or []
    infs = invalid.get("inf") or []
    failed = [c for c in (td.get("sanity_checks") or [])
              if not c.get("passed", True)]

    if not warnings and not nans and not infs and not failed:
        lines.append("  (no flagged changes, no NaN/Inf, all sanity checks passed)")
        return lines

    if warnings:
        lines.append(f"  large changes : {len(warnings)}")
        for w in warnings[:5]:
            lines.append(f"    - {w.get('label', w.get('key'))}: {w.get('change')}")
        if len(warnings) > 5:
            lines.append(f"    ... and {len(warnings) - 5} more")
    if nans:
        head = ", ".join(nans[:8])
        more = f" (+{len(nans) - 8} more)" if len(nans) > 8 else ""
        lines.append(f"  NaN values    : {head}{more}")
    if infs:
        head = ", ".join(infs[:8])
        more = f" (+{len(infs) - 8} more)" if len(infs) > 8 else ""
        lines.append(f"  Inf values    : {head}{more}")
    if failed:
        lines.append(f"  failed checks : {len(failed)}")
        for c in failed:
            lines.append(f"    - {c.get('check')}: {c.get('detail')}")
    return lines


def format_run_summary(run_dir: Path) -> str:
    """
    Build the multi-line pretty-printed summary string.

    Reads ``metadata.json`` (v3+ preferred) and falls back to
    ``read_simulation_end()`` for runs that pre-date the
    metadata-source-of-truth migration.  Pure function — no I/O
    side effects beyond the file reads.

    Parameters
    ----------
    run_dir : Path
        Directory containing ``metadata.json`` + ``dictionary.jsonl``
        (v4+).  Pre-Phase-5 runs with a legacy ``simulationEnd.txt``
        also work via the back-compat fallback.

    Returns
    -------
    str
        Multi-line text ready for printing.
    """
    status = _resolve_run_status(run_dir)
    md = status["metadata"]
    termination = status["termination"]
    final_state = status["final_state"]
    termination_debug = status["termination_debug"]
    is_successful = status["is_successful"]
    model_name = status["model_name"]

    lines = [
        _HR_HEAVY,
        f"TRINITY run: {model_name}",
        _HR_HEAVY,
        _status_line(termination, is_successful),
    ]
    if termination:
        if termination.get("detail"):
            lines.append(f"Detail   : {termination['detail']}")
        ts = termination.get("timestamp")
        if ts:
            lines.append(f"At       : {ts}")

    if md:
        lines.append("")
        lines.extend(_cloud_section(md))

    lines.append("")
    lines.extend(_final_state_section(final_state))

    debug_lines = _termination_debug_section(termination_debug)
    if debug_lines:
        lines.append("")
        lines.extend(debug_lines)

    lines.append(_HR_HEAVY)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="show_run",
        description=(
            "Pretty-print the metadata + termination + final-state "
            "summary for a TRINITY run directory."
        ),
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Path to the run directory (containing metadata.json "
             "and/or dictionary.jsonl).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print metadata.json verbatim instead of the pretty view.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Print only the status line; exit with the run's exit_code "
             "(0=success, non-zero=failure).  Suitable for shell loops.",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"show_run: not a directory: {run_dir}", file=_sys.stderr)
        return 1

    # --- --json passthrough ------------------------------------------
    if args.json:
        metadata_path = run_dir / METADATA_FILENAME
        if not metadata_path.is_file():
            print(f"show_run: {metadata_path} not found", file=_sys.stderr)
            return 1
        print(metadata_path.read_text(), end="")
        return 0

    # --- --quiet: one-line status + meaningful exit code -------------
    if args.quiet:
        status = _resolve_run_status(run_dir)
        print(_status_line(status["termination"], status["is_successful"]))
        if status["is_successful"] is True:
            return 0
        # Failure path: propagate the run's exit_code if we have one,
        # capped to [1, 9] so the value fits in POSIX.
        t = status["termination"]
        if t and t.get("exit_code") is not None:
            try:
                return min(max(int(t["exit_code"]), 1), 9)
            except (TypeError, ValueError):
                return 1
        return 1

    # --- Default: full pretty-print ---------------------------------
    print(format_run_summary(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
