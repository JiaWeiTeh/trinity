#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pedrini+2026 emergence-timescale comparison.

For each run in a TRINITY sweep, compute:
  - tau_TOT: time at which R2 first crosses rCloud (cluster emergence),
             linearly interpolated between the two straddling snapshots.
             For runs that never reach rCloud, tau_TOT = t_max (lower limit).

Output (under <FIG_DIR>/<sweep_dir.name>/):
  - pedrini_emergence_timescales.pdf
  - pedrini_emergence_timescales_summary.csv

Usage
-----
Run from the project root:

    python src/_plots/pedrini_emergence_timescales.py \
        --sweep_dir outputs/pedrini_sweep_grid

The Pedrini+2026 overlay is optional. Pass `--pedrini_csv mock` to use the
hand-digitised reference data embedded in this script:

    python src/_plots/pedrini_emergence_timescales.py \
        --sweep_dir outputs/pedrini_sweep_grid \
        --pedrini_csv mock

Or pass a real CSV path:

    python src/_plots/pedrini_emergence_timescales.py \
        --sweep_dir outputs/pedrini_sweep_grid \
        --pedrini_csv path/to/pedrini2026.csv

The CSV needs columns log_Mstar, tau_TOT, tau_TOT_err
(errors are 1-sigma symmetric in Myr).

Stdout
------
While running, one progress line per simulation is printed:

    [pedrini_tau] (i/N) <run_name>

Plus a one-line notice for each run that breaks out, e.g.:

    [pedrini_tau] <run_name>: rCloud crossing interpolated at t=... Myr
        (snapshots i=k-1/k, R2=...->... pc, t=...->... Myr)

Runs without that notice did not break out, and their tau_TOT is a lower
limit (t_max), plotted as an open/filled triangle in the figure.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src._plots.plot_base import FIG_DIR
from src._output.trinity_reader import (
    TrinityOutput,
    find_all_simulations,
)
from src._functions.unit_conversions import INV_CONV


# Wong 2011 colourblind-safe palette, ordered to match paper_densityProfile.py
# for the first four entries; remaining entries extend to the full eight-colour
# Wong set so we never run out when a sweep has many sfe values.
WONG = ["#0072B2", "#E69F00", "#CC79A7", "#009E73",
        "#D55E00", "#56B4E9", "#F0E442", "#000000"]

STYLE_PATH = Path(__file__).parent / "trinity.mplstyle"

# Hand-digitised stand-in for Pedrini+2026 Fig. X, used when the real CSV
# isn't available. Activate with `--pedrini_csv mock`.
MOCK_PEDRINI_CSV = """\
log_Mstar,tau_TOT,tau_TOT_err
2.25,3.80,0.40
2.35,3.90,0.40
2.45,4.20,0.40
2.55,4.90,0.40
2.65,5.20,0.40
2.75,5.80,0.40
2.85,6.70,0.40
2.95,6.90,0.40
3.05,7.20,0.40
3.15,7.40,0.40
3.25,7.50,0.40
3.35,7.50,0.40
3.45,7.40,0.40
3.55,7.00,0.40
3.72,6.00,0.50
3.92,5.95,0.50
4.62,4.90,0.50
"""


# ---------------------------------------------------------------------------
# Per-run extraction
# ---------------------------------------------------------------------------

def parse_param_file(param_path: Path) -> dict[str, str]:
    """Parse a TRINITY .param file (whitespace-separated `key value` lines)."""
    out: dict[str, str] = {}
    for raw in param_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        out[parts[0]] = parts[1].strip()
    return out


def get_run_sfe(run_dir: Path) -> float:
    """Read `sfe` from the .param file inside a run directory.

    More robust than parsing the folder name (which encodes sfe with rounding,
    e.g. sfe=0.0085 → 'sfe001').
    """
    candidates = list(run_dir.glob("*.param"))
    if not candidates:
        raise FileNotFoundError(f"No .param file in {run_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple .param files in {run_dir}: {candidates}")
    params = parse_param_file(candidates[0])
    if "sfe" not in params:
        raise KeyError(f"`sfe` not in {candidates[0]}")
    return float(params["sfe"])


def parse_raw_reason(run_dir: Path) -> str:
    """Return the `Raw Reason:` line from simulationEnd.txt (empty if missing)."""
    end_path = run_dir / "simulationEnd.txt"
    if not end_path.exists():
        return ""
    for line in end_path.read_text().splitlines():
        if line.startswith("Raw Reason:"):
            return line.split(":", 1)[1].strip()
    return ""


def find_rcloud_crossing(t: np.ndarray, R2: np.ndarray,
                         rCloud: float, run_name: str = "") -> float | None:
    """First time R2 reaches rCloud, linearly interpolated. None if never.

    Prints a one-line notice on every interpolation, matching the
    [TrinityOutput] convention at trinity_reader.py:607-611, so the caller
    sees that tau_TOT is an interpolated value rather than a snapshot time.
    """
    crossed = R2 >= rCloud
    if not crossed.any():
        return None
    j = int(np.argmax(crossed))
    tag = f"[pedrini_tau] {run_name}" if run_name else "[pedrini_tau]"
    if j == 0:
        # R2 already at/above rCloud at the first snapshot — no interpolation.
        return float(t[0])
    R0, R1 = R2[j - 1], R2[j]
    if R1 == R0:
        return float(t[j])
    frac = (rCloud - R0) / (R1 - R0)
    t_cross = float(t[j - 1] + frac * (t[j] - t[j - 1]))
    print(f"{tag}: rCloud crossing interpolated at t={t_cross:.6f} Myr "
          f"(snapshots i={j-1}/{j}, R2={R0:.4f}->{R1:.4f} pc, "
          f"t={t[j-1]:.6f}->{t[j]:.6f} Myr)")
    return t_cross


def collect_run(run_dir: Path) -> dict:
    data_path = run_dir / "dictionary.jsonl"
    if not data_path.exists():
        data_path = run_dir / "dictionary.json"
    out = TrinityOutput.open(data_path)

    t   = np.asarray(out.get("t_now"),  dtype=float)
    R2  = np.asarray(out.get("R2"),     dtype=float)

    mCloud   = float(out[0].get("mCloud"))
    rCloud   = float(out[0].get("rCloud"))
    nCore_au = float(out[0].get("nCore"))
    nCore_cgs = nCore_au * INV_CONV.ndens_au2cgs

    sfe = get_run_sfe(run_dir)
    M_star = mCloud * sfe

    t_cross = find_rcloud_crossing(t, R2, rCloud, run_name=run_dir.name)
    if t_cross is None:
        # R2 never reached rCloud — run hit stop_t (or another non-breakout
        # exit). tau_TOT becomes a lower limit set by the run length.
        tau_TOT = float(t[-1])
        breakout = False
    else:
        tau_TOT = t_cross
        breakout = True

    # Raw simulation-end reason is recorded for traceability only; breakout
    # status is derived from the actual R2=rCloud crossing above.
    raw_reason = parse_raw_reason(run_dir)

    return {
        "run_name":   run_dir.name,
        "mCloud":     mCloud,
        "sfe":        sfe,
        "nCore_cgs":  nCore_cgs,
        "M_star":     M_star,
        "tau_TOT":    tau_TOT,
        "breakout":   breakout,
        "end_reason": raw_reason,
    }


def collect_all(sweep_dir: Path) -> list[dict]:
    sim_files = find_all_simulations(sweep_dir)
    rows = []
    for i, f in enumerate(sim_files, 1):
        print(f"[pedrini_tau] ({i}/{len(sim_files)}) {f.parent.name}")
        rows.append(collect_run(f.parent))
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "run_name", "mCloud_Msun", "sfe", "nCore_cm-3", "M_star_Msun",
        "tau_TOT_Myr", "breakout_flag", "end_reason",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in rows:
            w.writerow([
                r["run_name"], r["mCloud"], r["sfe"], r["nCore_cgs"],
                r["M_star"], r["tau_TOT"],
                r["breakout"], r["end_reason"],
            ])


def load_pedrini_csv(source: str | Path):
    """Pedrini+2026 reference data.

    `source` is either the literal string ``"mock"`` (use the embedded
    MOCK_PEDRINI_CSV) or a path to a CSV file with columns
    log_Mstar, tau_TOT, tau_TOT_err. Errors are interpreted as 1-sigma
    symmetric in linear (Myr) space.
    """
    import pandas as pd
    if str(source) == "mock":
        return pd.read_csv(io.StringIO(MOCK_PEDRINI_CSV))
    return pd.read_csv(source)


def _marker_size(mCloud: float, m_min: float, m_max: float) -> float:
    """Marker size (in points) scaled linearly with log10(mCloud).

    Maps [log10(m_min), log10(m_max)] -> [MS_MIN, MS_MAX]. Falls back to
    the midpoint size if all runs share a single mCloud.
    """
    MS_MIN, MS_MAX = 4.0, 14.0
    lo, hi = np.log10(m_min), np.log10(m_max)
    if hi <= lo:
        return 0.5 * (MS_MIN + MS_MAX)
    frac = (np.log10(mCloud) - lo) / (hi - lo)
    return MS_MIN + frac * (MS_MAX - MS_MIN)


def _sfe_color_map(rows: list[dict]) -> dict[float, str]:
    """Map each unique sfe (ascending) to a Wong palette colour.

    Cycles the palette if a sweep has more sfe values than colours, which
    isn't expected for the standard pedrini sweeps but keeps the helper
    robust.
    """
    unique_sfe = sorted({float(r["sfe"]) for r in rows})
    return {s: WONG[i % len(WONG)] for i, s in enumerate(unique_sfe)}


def _half_dex_ticks(m_min: float, m_max: float) -> list[float]:
    """mCloud values at 0.5-dex steps (1e3, 3.16e3, 1e4, ...) inside [m_min, m_max].

    Returns log10 values. Falls back to a single endpoint when the data range
    is narrower than one half-dex (so the legend always has at least one
    swatch).
    """
    log_lo, log_hi = np.log10(m_min), np.log10(m_max)
    start = np.ceil(2 * log_lo) / 2.0
    end   = np.floor(2 * log_hi) / 2.0
    ticks = list(np.arange(start, end + 1e-9, 0.5))
    if not ticks:
        ticks = [log_lo]
    return ticks


def _fmt_half_dex(log_m: float) -> str:
    """Pretty-print a half-dex value: 10^k for log_m == k, 3*10^k for k+0.5.

    Off-grid values (the single-mCloud fallback) fall back to one-digit
    scientific notation so the label still reflects the actual mass.
    """
    exp = int(np.floor(log_m + 1e-9))
    frac = log_m - exp
    if frac < 0.1:
        return fr"10^{{{exp}}}"
    if abs(frac - 0.5) < 0.1:
        return fr"3{{\times}}10^{{{exp}}}"
    return fr"{10 ** log_m:.1e}"


def make_plot(rows: list[dict], pedrini_df, out_pdf: Path) -> None:
    plt.style.use(str(STYLE_PATH))

    masses = [r["mCloud"] for r in rows]
    m_min, m_max = min(masses), max(masses)

    sfe_colors = _sfe_color_map(rows)
    REF_COLOR  = "k"       # Pedrini+2026 reference data

    fig, ax = plt.subplots()

    for r in rows:
        x = np.log10(r["M_star"])
        ms = _marker_size(r["mCloud"], m_min, m_max)
        color = sfe_colors[float(r["sfe"])]
        marker = "^" if not r["breakout"] else "o"
        ax.plot(x, r["tau_TOT"], marker=marker,
                mfc=color, mec=color,
                linestyle="none", markersize=ms)

    if pedrini_df is not None:
        ax.errorbar(pedrini_df["log_Mstar"], pedrini_df["tau_TOT"],
                    yerr=pedrini_df["tau_TOT_err"],
                    marker="o", mfc=REF_COLOR, mec=REF_COLOR, ecolor=REF_COLOR,
                    linestyle="none", markersize=7, capsize=2)

    ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
    ax.set_ylabel(r"$\tau_{\rm TOT}\,[\mathrm{Myr}]$")

    # Legend: shape entries (breakout vs lower-limit) use a fixed mid size,
    # mCloud size-tier swatches sit on a 0.5-dex grid (1e3, 3e3, 1e4, ...)
    # inside the data range, and sfe entries get one swatch per unique value.
    mid_ms = 0.5 * (_marker_size(m_min, m_min, m_max)
                    + _marker_size(m_max, m_min, m_max))

    handles = [
        Line2D([], [], marker="o", linestyle="none",
               mfc="0.4", mec="0.4", markersize=mid_ms,
               label="breakout"),
        Line2D([], [], marker="^", linestyle="none",
               mfc="0.4", mec="0.4", markersize=mid_ms,
               label="lower limit (no breakout)"),
    ]
    for sfe, color in sfe_colors.items():
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              mfc=color, mec=color, markersize=mid_ms,
                              label=fr"$\mathrm{{sfe}}={sfe:g}$"))
    for log_m in _half_dex_ticks(m_min, m_max):
        m = 10 ** log_m
        ms = _marker_size(m, m_min, m_max)
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              mfc="0.4", mec="0.4", markersize=ms,
                              label=fr"$M_{{\rm cloud}}={_fmt_half_dex(log_m)}"
                                    r"\,M_\odot$"))
    if pedrini_df is not None:
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              mfc=REF_COLOR, mec=REF_COLOR, markersize=mid_ms,
                              label="Pedrini+2026"))
    ax.legend(handles=handles, loc="best")

    fig.savefig(out_pdf)
    print(f"Saved: {out_pdf}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sweep_dir", required=True, type=Path,
                    help="Sweep output directory (contains per-run subdirs).")
    ap.add_argument("--pedrini_csv", type=str, default=None,
                    help="Optional Pedrini+2026 CSV path "
                         "(log_Mstar, tau_TOT, tau_TOT_err; errors are "
                         "1-sigma symmetric in Myr). "
                         "Pass 'mock' to use the digitised reference data "
                         "embedded in this script.")
    args = ap.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    if not sweep_dir.is_dir():
        ap.error(f"--sweep_dir not found: {sweep_dir}")

    rows = collect_all(sweep_dir)
    if not rows:
        ap.error(f"No simulations found under {sweep_dir}")

    fig_dir = FIG_DIR / sweep_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = fig_dir / "pedrini_emergence_timescales_summary.csv"
    write_summary_csv(rows, csv_path)
    print(f"Wrote summary: {csv_path} ({len(rows)} rows)")

    if args.pedrini_csv is None:
        pedrini_df = None
    elif args.pedrini_csv == "mock":
        pedrini_df = load_pedrini_csv("mock")
    else:
        csv_in = Path(args.pedrini_csv).resolve()
        if not csv_in.is_file():
            ap.error(f"--pedrini_csv not found: {csv_in}")
        pedrini_df = load_pedrini_csv(csv_in)

    pdf_path = fig_dir / "pedrini_emergence_timescales.pdf"
    make_plot(rows, pedrini_df, pdf_path)


if __name__ == "__main__":
    main()
