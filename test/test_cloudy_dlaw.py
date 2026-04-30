"""
Tests for ``src._output.cloudy.dlaw.build_dlaw_block``.

Covers:
- Format: header / continue-prefixed rows / footer.
- Sort + dedup of unsorted, duplicate-r input.
- Bracket-check error path (r_in below or r_out above the supplied range).
- IF preservation: a synthetic profile with one steep pair densifies the
  smooth spans only, leaving the edge-pair endpoints untouched.
- Ambient splicing: shell + ambient produces a strictly-increasing block
  with no duplicate row at the join.
- All-edges-no-densification fallback (warns, returns input).
- NaN/Inf rejection.
- Unit conversion against hand-computed values from INV_CONV.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src._functions.unit_conversions import INV_CONV
from src._output.cloudy.dlaw import (
    DEFAULT_DLAW_CLOSE,
    DEFAULT_DLAW_OPEN,
    DEFAULT_DLAW_ROW_PREFIX,
    DlawError,
    build_dlaw_block,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_block(block: str) -> tuple[str, list[tuple[float, float]], str]:
    """Split a dlaw block into (header, [(log_r, log_n), ...], footer)."""
    lines = block.split("\n")
    header = lines[0]
    footer = lines[-1]
    rows = []
    for line in lines[1:-1]:
        assert line.startswith(DEFAULT_DLAW_ROW_PREFIX), f"bad row: {line!r}"
        body = line[len(DEFAULT_DLAW_ROW_PREFIX):]
        parts = body.split()
        assert len(parts) == 2, f"row needs 2 fields: {line!r}"
        rows.append((float(parts[0]), float(parts[1])))
    return header, rows, footer


# --------------------------------------------------------------------------- #
# Basic format + monotonicity + dedup + brackets
# --------------------------------------------------------------------------- #

def test_dlaw_basic_format_and_monotonicity():
    # 12 well-resolved points covering [1.0, 2.0] pc
    r_pc = np.linspace(1.0, 2.0, 12)
    log_n_pc3 = 58.0 - 0.5 * np.log10(r_pc)  # mild PL
    block = build_dlaw_block(
        r_pc, log_n_pc3,
        r_in_pc=1.0, r_out_pc=2.0, min_rows=10,
    )
    header, rows, footer = _parse_block(block)
    assert header == DEFAULT_DLAW_OPEN
    assert footer == DEFAULT_DLAW_CLOSE
    assert len(rows) == 12  # already ≥ min_rows; no densification
    # Strictly increasing in r
    log_rs = [r for r, _ in rows]
    assert all(b > a for a, b in zip(log_rs, log_rs[1:]))


def test_dlaw_unit_conversion_matches_inv_conv():
    # Single, hand-checked point: r = 1 pc, n = 1 pc^-3
    # log10(r/cm) = log10(pc2cm); log10(n/cm^-3) = log10(ndens_au2cgs)
    r_pc = np.array([1.0, 2.0])
    log_n_pc3 = np.array([0.0, 0.0])  # constant 1 pc^-3
    block = build_dlaw_block(
        r_pc, log_n_pc3, r_in_pc=1.0, r_out_pc=2.0, min_rows=2,
    )
    _, rows, _ = _parse_block(block)
    expected_log_r0 = math.log10(INV_CONV.pc2cm)               # 18.4892
    expected_log_n = math.log10(INV_CONV.ndens_au2cgs)         # -55.4682
    assert rows[0][0] == pytest.approx(expected_log_r0, abs=1e-5)
    assert rows[0][1] == pytest.approx(expected_log_n, abs=1e-3)


def test_dlaw_sorts_and_dedups_input():
    # Out of order, with one exact duplicate r (different n; the LATER value wins)
    r_pc =       np.array([2.0, 1.0, 1.5, 1.5, 1.2])
    log_n_pc3 =  np.array([55.0, 58.0, 57.0, 56.5, 57.5])  # idx 3 wins at r=1.5
    block = build_dlaw_block(
        r_pc, log_n_pc3, r_in_pc=1.0, r_out_pc=2.0, min_rows=2,
    )
    _, rows, _ = _parse_block(block)
    log_rs = [r for r, _ in rows]
    # Sorted, no duplicates, strictly increasing
    assert log_rs == sorted(log_rs)
    assert len(set(log_rs)) == len(log_rs)
    # The kept value at r=1.5 is the second (56.5), not the first (57.0).
    # In CGS log: log_n_cgs = 56.5 + log10(ndens_au2cgs)
    expected_log_n_at_1p5 = 56.5 + math.log10(INV_CONV.ndens_au2cgs)
    log_r_at_1p5 = math.log10(1.5) + math.log10(INV_CONV.pc2cm)
    matched = [n for r, n in rows if abs(r - log_r_at_1p5) < 1e-6]
    assert matched, f"row at r=1.5 pc not found: {rows}"
    assert matched[0] == pytest.approx(expected_log_n_at_1p5, abs=1e-3)


def test_dlaw_bracket_check_below_range():
    r_pc = np.array([1.0, 2.0])
    log_n_pc3 = np.array([58.0, 57.0])
    with pytest.raises(DlawError, match="below dlaw range start"):
        build_dlaw_block(
            r_pc, log_n_pc3, r_in_pc=0.5, r_out_pc=2.0, min_rows=2,
        )


def test_dlaw_bracket_check_above_range_no_ambient():
    r_pc = np.array([1.0, 2.0])
    log_n_pc3 = np.array([58.0, 57.0])
    with pytest.raises(DlawError, match="past dlaw range end"):
        build_dlaw_block(
            r_pc, log_n_pc3, r_in_pc=1.0, r_out_pc=3.0, min_rows=2,
        )


# --------------------------------------------------------------------------- #
# IF preservation
# --------------------------------------------------------------------------- #

def test_dlaw_if_preservation():
    """
    Synthetic 4-row profile in pc, with a near-instantaneous 2-dex jump
    between rows 1 and 2 — the ionisation-front pattern.

    With min_rows=10, the densifier must:
      - leave the four input rows intact in the output, and
      - distribute new rows ONLY in the smooth spans (0→1, 2→3),
        not across the IF pair (1→2).
    """
    r_pc = np.array([1.500, 1.610, 1.611, 1.700])
    log_n_pc3 = np.array([57.8, 57.9, 60.1, 60.2])  # IF between idx 1 and 2

    block = build_dlaw_block(
        r_pc, log_n_pc3,
        r_in_pc=1.500, r_out_pc=1.700,
        min_rows=10,
    )
    _, rows, _ = _parse_block(block)
    assert len(rows) >= 10

    # Compute the (log_r_cm, log_n_cgs) values the four input rows must keep
    log_pc_cm = math.log10(INV_CONV.pc2cm)
    log_ndens = math.log10(INV_CONV.ndens_au2cgs)
    expected_originals = [
        (math.log10(r) + log_pc_cm, n + log_ndens)
        for r, n in zip(r_pc, log_n_pc3)
    ]
    log_rs = [r for r, _ in rows]
    log_ns = [n for _, n in rows]
    for er, en in expected_originals:
        # Each original (r, n) must appear in the output with high precision
        match_idx = min(range(len(log_rs)), key=lambda i: abs(log_rs[i] - er))
        assert abs(log_rs[match_idx] - er) < 1e-5, f"original r={er} missing"
        assert abs(log_ns[match_idx] - en) < 1e-3, f"original n at r={er} altered"

    # The IF rows (originals 1 and 2) must be ADJACENT in the output —
    # nothing inserted between them.
    if_r_lo = math.log10(r_pc[1]) + log_pc_cm
    if_r_hi = math.log10(r_pc[2]) + log_pc_cm
    idx_lo = min(range(len(log_rs)), key=lambda i: abs(log_rs[i] - if_r_lo))
    idx_hi = min(range(len(log_rs)), key=lambda i: abs(log_rs[i] - if_r_hi))
    assert idx_hi == idx_lo + 1, (
        f"IF pair not adjacent in output: lo@{idx_lo}, hi@{idx_hi}; "
        f"densifier inserted into the IF span"
    )


def test_dlaw_all_edges_warns_and_passes_through():
    """If every pair is steep, densification is impossible — warn, return input."""
    r_pc = np.array([1.000, 1.001, 1.002])
    log_n_pc3 = np.array([50.0, 53.0, 56.0])  # 3 dex per Δlog_r ≈ 4e-4 → ratio ~1e4
    with pytest.warns(UserWarning, match="dlaw densification skipped"):
        block = build_dlaw_block(
            r_pc, log_n_pc3,
            r_in_pc=1.000, r_out_pc=1.002,
            min_rows=10,
        )
    _, rows, _ = _parse_block(block)
    assert len(rows) == 3  # untouched


# --------------------------------------------------------------------------- #
# Ambient splicing
# --------------------------------------------------------------------------- #

def test_dlaw_ambient_extension_no_duplicate_at_join():
    """
    Shell covers [1, 2] pc; ambient covers [2, 5] pc with one point exactly
    at the join (r=2). The splice must DROP the duplicate (we only take
    ambient points strictly past the shell tail).
    """
    shell_r =     np.array([1.0, 1.5, 2.0])
    shell_log_n = np.array([58.0, 57.5, 57.0])
    ambient_r =     np.array([2.0, 3.0, 4.0, 5.0])  # 2.0 must NOT be re-added
    ambient_log_n = np.array([56.0, 55.0, 54.5, 54.0])

    block = build_dlaw_block(
        shell_r, shell_log_n,
        ambient_r_pc=ambient_r,
        ambient_log_n_pc3=ambient_log_n,
        r_in_pc=1.0, r_out_pc=5.0,
        min_rows=2,
    )
    _, rows, _ = _parse_block(block)
    log_rs = [r for r, _ in rows]
    # No duplicate r values
    assert len(set(log_rs)) == len(log_rs)
    # Strictly increasing
    assert all(b > a for a, b in zip(log_rs, log_rs[1:]))
    # Should have shell (3 rows) + ambient past 2.0 (3 rows: 3, 4, 5) = 6 rows
    assert len(rows) == 6


def test_dlaw_ambient_not_used_when_r_out_within_shell():
    shell_r =     np.array([1.0, 1.5, 2.0])
    shell_log_n = np.array([58.0, 57.5, 57.0])
    ambient_r =     np.array([3.0, 4.0])
    ambient_log_n = np.array([55.0, 54.5])

    block = build_dlaw_block(
        shell_r, shell_log_n,
        ambient_r_pc=ambient_r, ambient_log_n_pc3=ambient_log_n,
        r_in_pc=1.0, r_out_pc=2.0,  # within shell
        min_rows=2,
    )
    _, rows, _ = _parse_block(block)
    assert len(rows) == 3  # shell only


def test_dlaw_ambient_required_when_r_out_past_shell():
    shell_r =     np.array([1.0, 2.0])
    shell_log_n = np.array([58.0, 57.0])
    # Ambient stops at 3 pc; user requested r_out = 5 pc → bracket fails
    ambient_r =     np.array([2.5, 3.0])
    ambient_log_n = np.array([55.0, 54.5])
    with pytest.raises(DlawError, match="past dlaw range end"):
        build_dlaw_block(
            shell_r, shell_log_n,
            ambient_r_pc=ambient_r, ambient_log_n_pc3=ambient_log_n,
            r_in_pc=1.0, r_out_pc=5.0, min_rows=2,
        )


# --------------------------------------------------------------------------- #
# Input rejection
# --------------------------------------------------------------------------- #

def test_dlaw_rejects_nan():
    r_pc = np.array([1.0, np.nan, 2.0])
    log_n_pc3 = np.array([58.0, 57.5, 57.0])
    with pytest.raises(DlawError, match="non-finite"):
        build_dlaw_block(r_pc, log_n_pc3, r_in_pc=1.0, r_out_pc=2.0, min_rows=2)


def test_dlaw_rejects_short_input():
    with pytest.raises(DlawError, match="≥2 points"):
        build_dlaw_block(
            np.array([1.0]), np.array([58.0]),
            r_in_pc=1.0, r_out_pc=1.0, min_rows=2,
        )


def test_dlaw_rejects_length_mismatch():
    with pytest.raises(DlawError, match="length mismatch"):
        build_dlaw_block(
            np.array([1.0, 2.0]), np.array([58.0, 57.0, 56.0]),
            r_in_pc=1.0, r_out_pc=2.0, min_rows=2,
        )


def test_dlaw_rejects_nonpositive_radius():
    with pytest.raises(DlawError, match="must be positive"):
        build_dlaw_block(
            np.array([0.0, 1.0]), np.array([58.0, 57.0]),
            r_in_pc=0.0, r_out_pc=1.0, min_rows=2,
        )


def test_dlaw_rejects_partial_ambient():
    """ambient_r and ambient_log_n must both be given or both None."""
    with pytest.raises(DlawError, match="both be given or both None"):
        build_dlaw_block(
            np.array([1.0, 2.0]), np.array([58.0, 57.0]),
            ambient_r_pc=np.array([3.0]),
            r_in_pc=1.0, r_out_pc=2.0, min_rows=2,
        )
