"""
Pure transformation: TRINITY snapshot dict + RunBundle → CLOUDY template
substitution dict.

No file I/O, no template rendering. The CLI (Step 4) calls this, then
substitutes the returned dict into the bundled .in template and writes the
deck and sidecar dlaw .txt.

Public API::

    from src._output.cloudy.snapshot_to_deck import (
        snapshot_to_values, SnapshotInvalid,
    )
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Mapping

import numpy as np

from src._functions.unit_conversions import INV_CONV
from src._output.cloudy.dlaw import DEFAULT_MIN_ROWS, build_dlaw_block
from src._output.cloudy.run_loader import RunBundle


REQUIRED_SNAPSHOT_KEYS = (
    "t_now", "R2", "rShell", "Qi",
    "shell_r_arr", "log_shell_n_arr", "current_phase",
)

# Sentinel for "key absent" — needed because TrinityOutput.Snapshot has
# __getitem__ but no __contains__/__iter__, so `key in snap` falls back to
# integer-indexed iteration that never terminates. Use snap.get(k, _MISSING).
_MISSING = object()

DEFAULT_AGE_MIN_YR = 1.0e5
DEFAULT_AGE_MAX_YR = 1.0e8


class SnapshotInvalid(ValueError):
    """Raised when a snapshot fails validation for CLOUDY conversion."""


def snapshot_to_values(
    snap: Mapping[str, Any],
    bundle: RunBundle,
    *,
    z_override: float | None = None,
    radius_out_pc: float | None = None,
    age_min_yr: float = DEFAULT_AGE_MIN_YR,
    age_max_yr: float = DEFAULT_AGE_MAX_YR,
    hard_age_bounds: bool = False,
    min_rows: int = DEFAULT_MIN_ROWS,
    extend_with_ambient: bool = True,
) -> dict[str, Any]:
    """
    Validate a snapshot and compute the values a CLOUDY .in template needs.

    Returns
    -------
    dict
        Substitution keys (TITLE, AGE_YR, LOG_QH, LOG_RIN, LOG_ROUT, ZREL,
        DLAW_BLOCK, DLAW_ROWS) plus a "_diagnostics" sub-dict for the
        manifest / summary print / sidecar writer.

        DLAW_BLOCK is the full dlaw block (header + rows + footer);
        DLAW_ROWS is rows only. Templates can use either.

    Z handling
    ----------
    bundle.summary["ZCloud"] by default; z_override (>0, finite) wins.

    Ambient splice
    --------------
    Only when radius_out_pc > rShell AND extend_with_ambient is True.
    Pulls metadata.initial_cloud_{r,n}_arr (linear, in pc / pc^-3) and
    converts the density to log10 before handing to dlaw.

    Validation
    ----------
    1. Required keys present.
    2. No NaN / Inf in used scalars / arrays.
    3. t_now > tSF (strict).
    4. R2 > 0, rShell > R2, Qi > 0.
    5. Shell array lengths match and are >= 2.
    6. shell_r_arr endpoints match R2 / rShell (rel_tol=1e-12) — simplify
       preserves them by contract; an exact-equality drift would indicate
       upstream regression.
    7. Cluster age in [age_min_yr, age_max_yr]: warn unless hard_age_bounds.

    All steps except 7 raise ``SnapshotInvalid``.
    """
    # 1. Required keys (use sentinel — see _MISSING comment above)
    missing = [
        k for k in REQUIRED_SNAPSHOT_KEYS
        if snap.get(k, _MISSING) is _MISSING
    ]
    if missing:
        raise SnapshotInvalid(f"snapshot missing required keys: {missing}")

    # 2. Pull and finite-check used scalars
    t_now = float(snap["t_now"])
    R2 = float(snap["R2"])
    rShell = float(snap["rShell"])
    Qi = float(snap["Qi"])
    for name, v in [("t_now", t_now), ("R2", R2), ("rShell", rShell), ("Qi", Qi)]:
        if not math.isfinite(v):
            raise SnapshotInvalid(f"snapshot.{name} is not finite ({v})")

    shell_r = np.asarray(snap["shell_r_arr"], dtype=float)
    shell_log_n = np.asarray(snap["log_shell_n_arr"], dtype=float)
    if not np.all(np.isfinite(shell_r)) or not np.all(np.isfinite(shell_log_n)):
        raise SnapshotInvalid(
            "shell_r_arr or log_shell_n_arr contains non-finite values"
        )

    if "tSF" not in bundle.metadata:
        raise SnapshotInvalid("bundle.metadata lacks 'tSF'")
    tSF = float(bundle.metadata["tSF"])
    if not math.isfinite(tSF):
        raise SnapshotInvalid(f"bundle.metadata['tSF'] is not finite ({tSF})")

    # 3. t_now > tSF (strict)
    if t_now <= tSF:
        raise SnapshotInvalid(
            f"t_now ({t_now}) must exceed tSF ({tSF}); cluster age "
            f"is non-positive"
        )

    # 4. Geometry / source positivity
    if R2 <= 0:
        raise SnapshotInvalid(f"R2 must be positive; got {R2:.3e}")
    if rShell <= R2:
        raise SnapshotInvalid(
            f"rShell ({rShell:.3e}) must exceed R2 ({R2:.3e})"
        )
    if Qi <= 0:
        raise SnapshotInvalid(f"Qi must be positive; got {Qi:.3e}")

    # 5. Shell array sizes
    if shell_r.size != shell_log_n.size:
        raise SnapshotInvalid(
            f"shell_r_arr / log_shell_n_arr length mismatch: "
            f"{shell_r.size} vs {shell_log_n.size}"
        )
    if shell_r.size < 2:
        raise SnapshotInvalid(
            f"shell arrays need >= 2 points; got {shell_r.size}"
        )

    # 6. Shell endpoints
    if not math.isclose(float(shell_r[0]), R2, rel_tol=1e-12):
        raise SnapshotInvalid(
            f"shell_r_arr[0] ({shell_r[0]:.6e}) does not match R2 "
            f"({R2:.6e}); upstream simplify endpoint guarantee broken?"
        )
    if not math.isclose(float(shell_r[-1]), rShell, rel_tol=1e-12):
        raise SnapshotInvalid(
            f"shell_r_arr[-1] ({shell_r[-1]:.6e}) does not match rShell "
            f"({rShell:.6e}); upstream simplify endpoint guarantee broken?"
        )

    # 7. SB99 age band (cluster age in years)
    age_myr = t_now - tSF
    age_yr = age_myr * 1.0e6
    if not (age_min_yr <= age_yr <= age_max_yr):
        msg = (
            f"cluster age {age_yr:.3e} yr outside SB99 band "
            f"[{age_min_yr:.1e}, {age_max_yr:.1e}] yr; CLOUDY may "
            f"extrapolate or fail"
        )
        if hard_age_bounds:
            raise SnapshotInvalid(msg)
        warnings.warn(msg, stacklevel=2)

    # --- Compute template values --------------------------------------------
    log_pc_per_cm = math.log10(INV_CONV.pc2cm)
    log_qh = math.log10(Qi) - math.log10(INV_CONV.Myr2s)  # ph/Myr → ph/s
    log_rin = math.log10(R2) + log_pc_per_cm

    # Outer radius: rShell unless user requested extension
    if radius_out_pc is None:
        r_out_pc = rShell
    else:
        if not math.isfinite(radius_out_pc):
            raise SnapshotInvalid(
                f"radius_out_pc must be finite; got {radius_out_pc}"
            )
        if radius_out_pc < rShell:
            raise SnapshotInvalid(
                f"radius_out_pc ({radius_out_pc:.3e}) must be >= rShell "
                f"({rShell:.3e}); use rShell or larger to extend dlaw"
            )
        r_out_pc = float(radius_out_pc)
    log_rout = math.log10(r_out_pc) + log_pc_per_cm

    # Z scale
    if z_override is not None:
        if not (math.isfinite(z_override) and z_override > 0):
            raise SnapshotInvalid(
                f"z_override must be positive and finite; got {z_override}"
            )
        zrel = float(z_override)
    else:
        if "ZCloud" not in bundle.summary:
            raise SnapshotInvalid(
                "bundle.summary lacks 'ZCloud' (and no z_override)"
            )
        zrel = float(bundle.summary["ZCloud"])

    # Ambient splice from metadata, only when actually extending past rShell
    ambient_r_pc = None
    ambient_log_n_pc3 = None
    if r_out_pc > rShell and not extend_with_ambient:
        # Without ambient data we cannot satisfy the dlaw bracket check past
        # rShell — surface this here rather than as a deeper "past dlaw range
        # end" error from build_dlaw_block.
        raise SnapshotInvalid(
            f"radius_out_pc ({r_out_pc:.3e}) > rShell ({rShell:.3e}) requires "
            f"extend_with_ambient=True (currently False)"
        )
    if extend_with_ambient and r_out_pc > rShell:
        amb_r = bundle.metadata.get("initial_cloud_r_arr")
        amb_n = bundle.metadata.get("initial_cloud_n_arr")
        if amb_r is None or amb_n is None:
            raise SnapshotInvalid(
                "ambient extension requested but metadata lacks "
                "'initial_cloud_r_arr' / 'initial_cloud_n_arr'"
            )
        amb_r = np.asarray(amb_r, dtype=float)
        amb_n = np.asarray(amb_n, dtype=float)
        if amb_r.size == 0:
            raise SnapshotInvalid(
                "ambient extension requested but initial_cloud arrays empty"
            )
        # Convert linear pc^-3 → log10 pc^-3 (eps guard for safety; TRINITY
        # writes positive values, but defends against future regressions).
        eps = np.finfo(float).tiny
        ambient_r_pc = amb_r
        ambient_log_n_pc3 = np.log10(np.maximum(amb_n, eps))

    # Build dlaw (header + rows + footer)
    dlaw_block = build_dlaw_block(
        shell_r, shell_log_n,
        ambient_r_pc=ambient_r_pc,
        ambient_log_n_pc3=ambient_log_n_pc3,
        r_in_pc=R2,
        r_out_pc=r_out_pc,
        min_rows=min_rows,
        dens_profile=str(bundle.metadata.get("dens_profile", "densPL")),
    )
    # Rows-only view: strip the first (header) and last (footer) lines.
    dlaw_lines = dlaw_block.split("\n")
    dlaw_rows_only = "\n".join(dlaw_lines[1:-1])
    n_dlaw_rows = len(dlaw_lines) - 2  # excluding open/close

    title = (
        f"TRINITY {bundle.model_name} "
        f"phase={snap['current_phase']} "
        f"age={age_myr:.4f}Myr"
    )

    return {
        # Substitution keys (pre-formatted; predictable string output in deck)
        "TITLE": title,
        "AGE_YR": f"{age_yr:.4e}",
        "LOG_QH": f"{log_qh:.4f}",
        "LOG_RIN": f"{log_rin:.4f}",
        "LOG_ROUT": f"{log_rout:.4f}",
        "ZREL": f"{zrel:.4f}",
        "DLAW_BLOCK": dlaw_block,
        "DLAW_ROWS": dlaw_rows_only,
        # Diagnostics — manifest, closing summary, sidecar
        "_diagnostics": {
            "age_myr": age_myr,
            "age_yr": age_yr,
            "t_now_myr": t_now,
            "tSF_myr": tSF,
            "phase": str(snap["current_phase"]),
            "n_dlaw_rows": n_dlaw_rows,
            "R2_pc": R2,
            "rShell_pc": rShell,
            "r_out_pc": r_out_pc,
            "ambient_extended": ambient_r_pc is not None,
            "z_used": zrel,
        },
    }


__all__ = [
    "DEFAULT_AGE_MAX_YR",
    "DEFAULT_AGE_MIN_YR",
    "REQUIRED_SNAPSHOT_KEYS",
    "SnapshotInvalid",
    "snapshot_to_values",
]
