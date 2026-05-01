"""
Load a TRINITY run directory into a typed bundle the trinity_to_cloudy
driver can consume.

A run directory has this shape (see ``outputs/mockOutput/mockFullrun/``)::

    <run_dir>/
    ├── <model>.param                    # raw input config (not parsed here)
    ├── <model>_summary.txt              # full resolved config (parsed)
    ├── dictionary.jsonl                 # snapshot stream (opened via TrinityOutput)
    ├── metadata.json                    # run-invariant data (parsed)
    ├── simulationEnd.txt                # success / failure (parsed)
    └── ...                              # plots, debug logs (ignored)

Public API::

    from src._output.cloudy.run_loader import load_run, RunBundle, RunLoadError
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src._output.trinity_reader import TrinityOutput, find_data_path


# Canonical TRINITY density-profile enum (mirrors src/_input/read_param.py:286).
VALID_DENS_PROFILES = frozenset({"densBE", "densPL"})


class RunLoadError(ValueError):
    """Raised when a run directory cannot be loaded into a RunBundle."""


@dataclass(frozen=True)
class RunBundle:
    """Everything trinity_to_cloudy needs about one TRINITY run."""

    run_dir: Path
    model_name: str
    metadata: Mapping[str, Any]      # parsed metadata.json
    summary: Mapping[str, Any]       # parsed <model>_summary.txt
    end_state: Mapping[str, Any]     # parsed simulationEnd.txt
    output: TrinityOutput            # opened from find_data_path(run_dir)


def load_run(run_dir: str | Path) -> RunBundle:
    """
    Parse a TRINITY run directory and return a RunBundle.

    Parameters
    ----------
    run_dir
        Path to the directory containing metadata.json, the summary file,
        simulationEnd.txt, and dictionary.jsonl.

    Raises
    ------
    RunLoadError
        If any expected file is missing, malformed, or carries an
        unknown ``dens_profile``.
    FileNotFoundError
        If ``run_dir`` itself does not exist.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # --- metadata.json (run-invariant config) -------------------------------
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise RunLoadError(f"metadata.json missing from {run_dir}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as e:
        raise RunLoadError(f"metadata.json malformed: {e}") from e
    if "model_name" not in metadata:
        raise RunLoadError("metadata.json lacks a 'model_name' field")
    model_name = metadata["model_name"]

    dens_profile = metadata.get("dens_profile")
    if dens_profile not in VALID_DENS_PROFILES:
        raise RunLoadError(
            f"unknown dens_profile {dens_profile!r}; "
            f"expected one of {sorted(VALID_DENS_PROFILES)}"
        )

    # --- <model>_summary.txt (resolved config + ZCloud) ---------------------
    summary_path = run_dir / f"{model_name}_summary.txt"
    if not summary_path.is_file():
        raise RunLoadError(
            f"summary file missing: expected {summary_path.name} in {run_dir}"
        )
    summary = _parse_summary_txt(summary_path.read_text())

    # --- simulationEnd.txt (status gate) ------------------------------------
    end_path = run_dir / "simulationEnd.txt"
    if not end_path.is_file():
        raise RunLoadError(f"simulationEnd.txt missing from {run_dir}")
    end_state = _parse_simulation_end(end_path.read_text())

    # --- dictionary.jsonl (per-snapshot data) -------------------------------
    try:
        jsonl_path = find_data_path(run_dir)
    except FileNotFoundError as e:
        raise RunLoadError(f"snapshot stream not found in {run_dir}: {e}") from e
    output = TrinityOutput.open(jsonl_path)

    return RunBundle(
        run_dir=run_dir,
        model_name=model_name,
        metadata=metadata,
        summary=summary,
        end_state=end_state,
        output=output,
    )


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #

def _parse_summary_txt(text: str) -> dict[str, Any]:
    """
    Parse a ``<model>_summary.txt`` produced by the TRINITY driver.

    Format: ``<key><whitespace><value>`` per line; comments start with ``#``;
    blank lines ignored. Values are coerced (in order): bool, ``None``,
    ``nan``/``inf``, int, float, Python-literal (lists, tuples), else string.
    """
    out: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # split on first run of whitespace
        parts = line.split(None, 1)
        if len(parts) == 1:
            key, value_str = parts[0], ""
        else:
            key, value_str = parts
        out[key] = _coerce_scalar(value_str)
    return out


def _parse_simulation_end(text: str) -> dict[str, Any]:
    """
    Pull the status flag, end reason, and final-state snapshot from
    ``simulationEnd.txt``. Section headers (``-----``, ``=====``) are ignored;
    we look for known ``key: value`` lines wherever they appear.

    Returned keys (units in the key name where they differ from the summary's
    AU = (Msun, pc, Myr) convention)::

        model_name, status, end_reason, exit_code, raw_reason,
        t_now_myr, R2_pc, shell_nMax_cm3, shell_v_kms,
        mCloud_msun, nCore_cm3, rCloud_pc, rCore_pc, alpha, nISM_cm3
    """
    out: dict[str, Any] = {}

    # Map "Key" → (out_key, parser) — parser pulls the value past the colon.
    flat = {
        "Model":      ("model_name", str),
        "Status":     ("status", str),
        "End Reason": ("end_reason", str),
        "Raw Reason": ("raw_reason", str),
        "Exit Code":  ("exit_code", _safe_int),
    }
    # Final-state numeric fields with units stripped from the value
    numeric_units = {
        "Time":            ("t_now_myr", "Myr"),
        "Radius (R2)":     ("R2_pc", "pc"),
        "Shell nMax":      ("shell_nMax_cm3", "cm^-3"),
        "Shell Velocity":  ("shell_v_kms", "km/s"),
        "mCloud":          ("mCloud_msun", "Msun"),
        "nCore":           ("nCore_cm3", "cm^-3"),
        "rCloud":          ("rCloud_pc", "pc"),
        "rCore":           ("rCore_pc", "pc"),
        "alpha":           ("alpha", ""),
        "nISM":            ("nISM_cm3", "cm^-3"),
    }

    for raw in text.splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key in flat:
            out_key, conv = flat[key]
            out[out_key] = conv(value) if value else None
            continue
        if key in numeric_units:
            out_key, _unit = numeric_units[key]
            # value is "<number> <unit>" — take the first whitespace-split token
            tok = value.split()[0] if value else ""
            try:
                out[out_key] = float(tok)
            except (ValueError, IndexError):
                out[out_key] = None
    return out


# --------------------------------------------------------------------------- #
# Scalar coercion (used by _parse_summary_txt)
# --------------------------------------------------------------------------- #

def _coerce_scalar(s: str) -> Any:
    """
    Parse a summary-file value string into the most specific Python type
    that fits. Falls through to ``str``.
    """
    if s == "":
        return ""
    if s == "True":
        return True
    if s == "False":
        return False
    if s == "None":
        return None
    # nan / inf are NOT Python literals — float() handles them but ast doesn't.
    low = s.lower()
    if low in ("nan", "inf", "+inf", "-inf"):
        return float(s)
    # int (only if it looks like one — avoid "1.0" → ValueError fallthrough)
    if _looks_like_int(s):
        try:
            return int(s)
        except ValueError:
            pass
    # float (handles scientific notation, signed)
    try:
        return float(s)
    except ValueError:
        pass
    # Python literal (lists, tuples, dicts) — must start with a recognisable
    # literal opener, otherwise we'd accept arbitrary expressions.
    if s.startswith(("[", "(", "{")):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass
    return s


def _looks_like_int(s: str) -> bool:
    t = s[1:] if s[:1] in "+-" else s
    return t.isdigit()


def _safe_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None


# Re-exported helpers (for tests)
__all__ = [
    "RunBundle",
    "RunLoadError",
    "VALID_DENS_PROFILES",
    "load_run",
]
