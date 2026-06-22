"""Shared helper: blowout time (R2 first exceeds rCloud) per config, + a marker.

rCloud is NOT hardcoded and NOT in the snapshots (the `rCloud` CSV column is empty):
it is RECONSTRUCTED from each config's `.param` (mCloud, sfe, nCore, profile) using
TRINITY's own pipeline -- `read_param` + `validate_gmc_from_params` -- so it is exactly
the rCloud the simulation uses (gas mass = mCloud/(1-sfe) handling, mu_convert, the BE
Lane-Emden solve, all included). t_blowout is the first t_now where R2 > rCloud (the
shell exits the cloud into the low-density ambient).

Blowout is drawn as a vertical line in each config's own colour with a distinct
dash-dot style, on time-axis figures, so coincidence with the dip/surge is visible.
"""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BLOWOUT_LS = (0, (7, 2, 1, 2))  # distinct dash-dot, reserved for legacy blowout lines
BLOWOUT_MARKER = "*"            # blowout shown as a star ON the trajectory (cleaner)
_BLOWOUT_LABEL = r"blowout ($R_2>r_{\rm cloud}$)"
_rc_cache = {}
_tb_cache = {}


def rcloud(config):
    """rCloud [pc] reconstructed from configs/<config>.param via TRINITY's validate_gmc."""
    if config in _rc_cache:
        return _rc_cache[config]
    rC = None
    try:
        from trinity._input import read_param
        from trinity.cloud_properties.validate_gmc import validate_gmc_from_params
        params = read_param.read_param(str(HERE / "configs" / f"{config}.param"))
        rC = float(validate_gmc_from_params(params).rCloud)
    except Exception:  # noqa: BLE001 -- if trinity import/solve fails, marker is just skipped
        rC = None
    _rc_cache[config] = rC
    return rC


def t_blowout(config):
    """First t_now [Myr] where R2 > rCloud (shell exits the cloud), or None."""
    if config in _tb_cache:
        return _tb_cache[config]
    rC = rcloud(config)
    p = HERE / "data" / f"c0_{config}_h0.csv"
    t = None
    if rC is not None and rC == rC and p.exists():
        for r in csv.DictReader(open(p)):
            try:
                if float(r["R2"]) > rC:
                    t = float(r["t_now"]); break
            except (ValueError, KeyError, TypeError):
                continue
    _tb_cache[config] = t
    return t


def _interp_y(t_b, t_arr, y_arr):
    """y of the trajectory (t_arr, y_arr) at time t_b, by linear interp. Returns None if
    t_b is outside the curve's finite span (so the marker is simply skipped, never faked)."""
    ts, ys = [], []
    for ti, yi in zip(t_arr, y_arr):
        if ti is not None and yi is not None and np.isfinite(ti) and np.isfinite(yi):
            ts.append(ti)
            ys.append(yi)
    if len(ts) < 2:
        return None
    order = np.argsort(ts)
    ts = np.asarray(ts)[order]
    ys = np.asarray(ys)[order]
    if not (ts[0] <= t_b <= ts[-1]):
        return None
    return float(np.interp(t_b, ts, ys))


def mark_point(ax, t_b, t_arr, y_arr, color="0.35", label=False, ms=13, **kw):
    """Drop a blowout star ON the trajectory (t_arr, y_arr) at an explicit epoch t_b.

    Use this when the blowout time is computed by the caller (e.g. a frozen run's own
    R2 column). Interpolates the curve's y at t_b so the star sits on the line, replacing
    the full-height vertical line. No-op if t_b is None or off the curve's span."""
    if t_b is None:
        return None
    y_b = _interp_y(t_b, t_arr, y_arr)
    if y_b is None:
        return None
    ax.plot([t_b], [y_b], marker=BLOWOUT_MARKER, ms=ms, color=color, mec="white", mew=0.8,
            ls="none", zorder=5, label=(_BLOWOUT_LABEL if label else None), **kw)
    return t_b


def mark(ax, config, t=None, y=None, color="0.35", label=False, lw=1.3, ms=13, **kw):
    """Blowout indicator at t_blowout(config), in the config's colour.

    If a trajectory (t, y) is supplied, drop a star marker ON that curve at the blowout
    epoch (clean, uncluttered) -- preferred for multi-config time-axis figures. With no
    trajectory it falls back to the legacy full-height dash-dot vertical line."""
    tb = t_blowout(config)
    if tb is None:
        return None
    if t is not None and y is not None:
        return mark_point(ax, tb, t, y, color=color, label=label, ms=ms, **kw)
    ax.axvline(tb, color=color, ls=BLOWOUT_LS, lw=lw, zorder=1.4, alpha=0.9, **kw)
    if label:
        ax.text(tb, 0.97, " blowout", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=6.5, color=color, rotation=90)
    return tb


if __name__ == "__main__":  # quick self-check: reconstructed rCloud + blowout epoch
    for c in ("small_dense_highsfe", "simple_cluster", "midrange_pl0", "pl2_steep",
              "be_sphere", "large_diffuse_lowsfe"):
        print(f"{c:22s} rCloud={rcloud(c)!s:>8.8} pc   t_blowout={t_blowout(c)} Myr")
