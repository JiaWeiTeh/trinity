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

HERE = Path(__file__).resolve().parent
BLOWOUT_LS = (0, (7, 2, 1, 2))  # distinct dash-dot, reserved for blowout lines
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


def mark(ax, config, color="0.35", label=False, lw=1.3, **kw):
    """Vertical blowout line at t_blowout(config), in the config's colour, dash-dot."""
    t = t_blowout(config)
    if t is None:
        return None
    ax.axvline(t, color=color, ls=BLOWOUT_LS, lw=lw, zorder=1.4, alpha=0.9, **kw)
    if label:
        ax.text(t, 0.97, " blowout", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=6.5, color=color, rotation=90)
    return t


if __name__ == "__main__":  # quick self-check: reconstructed rCloud + blowout epoch
    for c in ("small_dense_highsfe", "simple_cluster", "midrange_pl0", "pl2_steep",
              "be_sphere", "large_diffuse_lowsfe"):
        print(f"{c:22s} rCloud={rcloud(c)!s:>8.8} pc   t_blowout={t_blowout(c)} Myr")
