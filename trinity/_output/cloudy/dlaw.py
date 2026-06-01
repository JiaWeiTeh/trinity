"""
Build a CLOUDY ``dlaw table radius`` block from a TRINITY shell density profile.

The block converts (r [pc], log10 n [pc^-3]) pairs into the (log10 r [cm],
log10 n [cm^-3]) form CLOUDY expects, with optional IF-preserving densification
when the profile is too sparse, and optional ambient-ISM splicing past rShell.

Output format (defaults — best-guess for CLOUDY C17/C22; see Step 5 smoke test)::

    dlaw table radius
    continue {log10(r/cm):.6f}  {log10(n_H/cm^-3):.4f}
    continue ...
    end of dlaw

Public API::

    from trinity._output.cloudy.dlaw import build_dlaw_block, DlawError
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np

from trinity._functions.unit_conversions import INV_CONV


# Best-guess CLOUDY syntax (Step 0 / Option B). Override at call site if a
# live smoke test reveals a different working form.
DEFAULT_DLAW_OPEN = "dlaw table radius"
DEFAULT_DLAW_ROW_PREFIX = "continue "
DEFAULT_DLAW_CLOSE = "end of dlaw"

DEFAULT_MIN_ROWS = 10

# |Δlog n / Δlog r| above this counts as an IF-like discontinuity. PL profiles
# are O(1); transition-phase IFs in TRINITY snapshots are O(1e5). 50 separates
# them with margin.
DEFAULT_EDGE_THRESHOLD = 50.0


class DlawError(ValueError):
    """Raised when dlaw construction fails validation."""


def build_dlaw_block(
    shell_r_pc: Sequence[float],
    shell_log_n_pc3: Sequence[float],
    *,
    ambient_r_pc: Sequence[float] | None = None,
    ambient_log_n_pc3: Sequence[float] | None = None,
    r_in_pc: float,
    r_out_pc: float,
    min_rows: int = DEFAULT_MIN_ROWS,
    dens_profile: str = "densPL",
    dlaw_open: str = DEFAULT_DLAW_OPEN,
    dlaw_row_prefix: str = DEFAULT_DLAW_ROW_PREFIX,
    dlaw_close: str = DEFAULT_DLAW_CLOSE,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> str:
    """
    Construct a CLOUDY dlaw block from TRINITY shell + (optional) ambient profiles.

    Parameters
    ----------
    shell_r_pc, shell_log_n_pc3
        Shell radius (pc) and log10 number density (1/pc^3). ≥2 points.
    ambient_r_pc, ambient_log_n_pc3
        Optional ambient ISM profile, used only when ``r_out_pc`` extends
        past the shell tail. Same units as the shell arrays.
    r_in_pc, r_out_pc
        Inner / outer radius CLOUDY will integrate over (pc). Must lie within
        the union of shell + ambient r-range.
    min_rows
        If the post-splice profile has fewer rows, densify by inserting points
        in non-edge spans. Edge (IF-like) pairs are preserved verbatim.
    dens_profile
        TRINITY profile shape; reserved for future PCHIP-on-densBE support.
        Currently unused; densification is linear-in-(log r, log n).
    dlaw_open, dlaw_row_prefix, dlaw_close
        CLOUDY syntax knobs. See module-level defaults.
    edge_threshold
        |Δlog n / Δlog r| above which a pair is locked as an "edge".

    Returns
    -------
    str
        Multi-line dlaw block, no trailing newline.
    """
    # --- 0. Validate scalar parameters --------------------------------------
    if not (math.isfinite(r_in_pc) and math.isfinite(r_out_pc)):
        raise DlawError(
            f"r_in_pc and r_out_pc must be finite; got {r_in_pc}, {r_out_pc}"
        )
    if r_in_pc <= 0:
        raise DlawError(f"r_in_pc must be positive; got {r_in_pc:.3e}")
    if r_out_pc <= r_in_pc:
        raise DlawError(
            f"r_out_pc ({r_out_pc:.3e}) must exceed r_in_pc ({r_in_pc:.3e})"
        )
    if min_rows < 2:
        raise DlawError(f"min_rows must be >= 2; got {min_rows}")
    if edge_threshold <= 0:
        raise DlawError(f"edge_threshold must be positive; got {edge_threshold}")

    # --- 1. Validate and arrayify inputs ------------------------------------
    r_pc = np.asarray(shell_r_pc, dtype=float)
    log_n_pc3 = np.asarray(shell_log_n_pc3, dtype=float)
    if r_pc.shape != log_n_pc3.shape:
        raise DlawError(
            f"shell_r_pc/shell_log_n_pc3 length mismatch: "
            f"{r_pc.shape} vs {log_n_pc3.shape}"
        )
    if r_pc.size < 2:
        raise DlawError(f"shell profile needs ≥2 points; got {r_pc.size}")
    if not np.all(np.isfinite(r_pc)) or not np.all(np.isfinite(log_n_pc3)):
        raise DlawError("shell profile contains non-finite values")
    if np.any(r_pc <= 0):
        raise DlawError(f"shell radii must be positive; min={r_pc.min():.3e} pc")

    # --- 2. Sort and dedup adjacent duplicates (keep last) ------------------
    order = np.argsort(r_pc, kind="stable")
    r_pc = r_pc[order]
    log_n_pc3 = log_n_pc3[order]
    keep = np.ones(r_pc.size, dtype=bool)
    keep[:-1] = r_pc[:-1] != r_pc[1:]
    r_pc = r_pc[keep]
    log_n_pc3 = log_n_pc3[keep]

    # --- 3. Optionally splice ambient past the shell tail -------------------
    if ambient_r_pc is not None or ambient_log_n_pc3 is not None:
        if ambient_r_pc is None or ambient_log_n_pc3 is None:
            raise DlawError(
                "ambient_r_pc and ambient_log_n_pc3 must both be given or both None"
            )
        a_r = np.asarray(ambient_r_pc, dtype=float)
        a_n = np.asarray(ambient_log_n_pc3, dtype=float)
        if a_r.shape != a_n.shape:
            raise DlawError(
                f"ambient length mismatch: {a_r.shape} vs {a_n.shape}"
            )
        if a_r.size and (
            not np.all(np.isfinite(a_r)) or not np.all(np.isfinite(a_n))
        ):
            raise DlawError("ambient profile contains non-finite values")
        if r_out_pc > r_pc[-1] and a_r.size:
            ord2 = np.argsort(a_r, kind="stable")
            a_r, a_n = a_r[ord2], a_n[ord2]
            # dedup ambient (keep last value at each unique r), same recipe as shell
            keep_a = np.ones(a_r.size, dtype=bool)
            keep_a[:-1] = a_r[:-1] != a_r[1:]
            a_r, a_n = a_r[keep_a], a_n[keep_a]
            mask = (a_r > r_pc[-1]) & (a_r <= r_out_pc)
            if mask.any():
                r_pc = np.concatenate([r_pc, a_r[mask]])
                log_n_pc3 = np.concatenate([log_n_pc3, a_n[mask]])

    # --- 4. Bracket check (with tiny float tolerance) -----------------------
    rel_tol = 1e-12
    if r_pc[0] > r_in_pc * (1.0 + rel_tol):
        raise DlawError(
            f"r_in_pc ({r_in_pc:.6e}) below dlaw range start ({r_pc[0]:.6e})"
        )
    if r_pc[-1] < r_out_pc * (1.0 - rel_tol):
        raise DlawError(
            f"r_out_pc ({r_out_pc:.6e}) past dlaw range end ({r_pc[-1]:.6e}); "
            f"supply ambient_* arrays extending to ≥ r_out_pc to splice"
        )

    # --- 5. Convert pc → cm and pc^-3 → cm^-3 in log space ------------------
    log_pc_per_cm = math.log10(INV_CONV.pc2cm)        # +18.4892
    log_ndens_offset = math.log10(INV_CONV.ndens_au2cgs)  # -55.4682
    log_r_cm = np.log10(r_pc) + log_pc_per_cm
    log_n_cm3 = log_n_pc3 + log_ndens_offset

    # --- 6. IF-preserving densification (only if too sparse) ----------------
    if log_r_cm.size < min_rows:
        log_r_cm, log_n_cm3 = _densify_preserving_edges(
            log_r_cm, log_n_cm3,
            target_rows=min_rows,
            edge_threshold=edge_threshold,
        )

    # --- 7. Final validation -------------------------------------------------
    if log_r_cm.size < 2:
        raise DlawError(
            f"dlaw must have ≥2 rows after construction; got {log_r_cm.size}"
        )
    if not np.all(np.diff(log_r_cm) > 0):
        raise DlawError("dlaw rows are not strictly increasing in r after construction")
    if not np.all(np.isfinite(log_r_cm)) or not np.all(np.isfinite(log_n_cm3)):
        raise DlawError("dlaw contains non-finite values after construction")

    # --- 8. Format ----------------------------------------------------------
    lines = [dlaw_open]
    for lr, ln in zip(log_r_cm, log_n_cm3):
        lines.append(f"{dlaw_row_prefix}{lr:.6f}  {ln:.4f}")
    lines.append(dlaw_close)
    return "\n".join(lines)


def _densify_preserving_edges(
    log_r: np.ndarray,
    log_n: np.ndarray,
    *,
    target_rows: int,
    edge_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert linearly-interpolated points into non-edge spans until
    ``len >= target_rows``. Edge (steep) pairs are preserved verbatim.

    If every pair is an edge, the input is returned unchanged with a warning.
    """
    n = log_r.size
    extra_needed = target_rows - n
    if extra_needed <= 0:
        return log_r, log_n

    dlog_r = np.diff(log_r)
    dlog_n = np.diff(log_n)
    slopes = np.abs(dlog_n / dlog_r)
    is_smooth = slopes <= edge_threshold
    smooth_idx = np.where(is_smooth)[0]

    if smooth_idx.size == 0:
        warnings.warn(
            f"dlaw densification skipped: all {n - 1} pair(s) exceed "
            f"edge_threshold={edge_threshold}; returning {n} rows "
            f"(below requested {target_rows}).",
            stacklevel=3,
        )
        return log_r, log_n

    # Distribute extra_needed slots across smooth pairs proportional to dlog_r
    smooth_lengths = dlog_r[smooth_idx]
    weights = smooth_lengths / smooth_lengths.sum()
    raw = weights * extra_needed
    k = np.floor(raw).astype(int)
    remainder = extra_needed - int(k.sum())
    if remainder > 0:
        # Hand out the leftover slots to pairs with the largest fractional part
        frac = raw - k
        # Stable order ensures determinism for ties
        rank = np.argsort(-frac, kind="stable")
        k[rank[:remainder]] += 1

    # Build output: original rows, plus inserted inner points in smooth pairs
    new_r: list[float] = []
    new_n: list[float] = []
    smooth_k = dict(zip(smooth_idx.tolist(), k.tolist()))
    for i in range(n):
        new_r.append(float(log_r[i]))
        new_n.append(float(log_n[i]))
        if i < n - 1 and smooth_k.get(i, 0) > 0:
            ki = smooth_k[i]
            inner = np.linspace(log_r[i], log_r[i + 1], ki + 2)[1:-1]
            inner_n = np.interp(
                inner,
                [log_r[i], log_r[i + 1]],
                [log_n[i], log_n[i + 1]],
            )
            new_r.extend(inner.tolist())
            new_n.extend(inner_n.tolist())

    return np.asarray(new_r, dtype=float), np.asarray(new_n, dtype=float)
