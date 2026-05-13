#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pedrini+2026 emergence-timescale comparison.

For each run in a TRINITY sweep, compute:
  - tau_TOT: time at which R2 first crosses rCloud (cluster emergence),
             linearly interpolated between the two straddling snapshots.
             For runs that never reach rCloud, tau_TOT = t_max (lower limit).
  - tau_PDR (optional, enable with `--show_tau_pdr`): cumulative time during
             which (is_phiDepleted AND NOT isDissolved), integrated only
             over t in [0, tau_TOT] using the left-rectangle rule (snapshots
             are saved BEFORE ODE integration, so each snapshot's flag
             value applies for its upcoming segment).  The dissolved-shell
             exclusion is a defensive guard — see cumulative_phi_time.

Recollapsing runs (any v2<0 in the trajectory) are excluded from the
plot since they never emerge in Pedrini's sense.  They remain in the
CSV with a `recollapse_flag` column for traceability.

Output (under <FIG_DIR>/<sweep_dir.name>/):
  - pedrini_emergence_timescales.pdf
  - pedrini_emergence_timescales_summary.csv

Usage
-----
Run from the project root:

    python src/_plots/pedrini_emergence_timescales.py \
        --sweep_dir outputs/pedrini_sweep_grid

Add `--show_tau_pdr` to also compute and plot tau_PDR.  Use
`--colourbar discrete` to swap the default viridis colourbar for a
Wong-palette band-per-sfe colourbar (useful for narrow sweeps).

Merge mode
----------
Pass `--merge_dir <other_sweep>` to overlay a second sweep whose runs
differ from `--sweep_dir` only in their density-profile settings (e.g.
densPL vs densBE, or two densPL_alpha values). Runs are paired by
(mCloud, sfe, nCore); unpaired runs are listed and skipped. The
`--sweep_dir` markers are drawn at alpha=0.3 (70%% transparent) and the
`--merge_dir` markers at alpha=0.8 (20%% transparent), so overlapping
points are still distinguishable. Output is
`pedrini_emergence_timescales_merge.pdf` under the `--sweep_dir` fig dir.

The Pedrini+2026 overlay is optional.  Pass `--pedrini_csv mock` to
use the hand-digitised reference data embedded in this script:

    python src/_plots/pedrini_emergence_timescales.py \
        --sweep_dir outputs/pedrini_sweep_grid \
        --pedrini_csv mock

Or pass a real CSV path:

    python src/_plots/pedrini_emergence_timescales.py \
        --sweep_dir outputs/pedrini_sweep_grid \
        --pedrini_csv path/to/pedrini2026.csv

The CSV needs columns log_Mstar, tau_TOT, tau_TOT_err; if `--show_tau_pdr`
is also set, columns tau_PDR, tau_PDR_err are required as well (all errors
1-sigma symmetric in Myr).

Stdout
------
While running, one progress line per simulation is printed:

    [pedrini_tau] (i/N) <run_name>

A one-line notice for each run that breaks out:

    [pedrini_tau] <run_name>: rCloud crossing interpolated at t=... Myr
        (snapshots i=k-1/k, R2=...->... pc, t=...->... Myr)

Runs without that notice did not break out; their tau_TOT is a lower
limit (t_max), plotted as an open circle with a dashed edge in the figure.

If any runs recollapsed (v2<0 anywhere), a notice listing the dropped
run names is printed after the CSV is written.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
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
# isn't available. Activate with `--pedrini_csv mock`. The tau_PDR columns
# are only used when `--show_tau_pdr` is also set.
MOCK_PEDRINI_CSV = """\
log_Mstar,tau_TOT,tau_TOT_err,tau_PDR,tau_PDR_err
2.25,3.80,0.40,1.85,0.30
2.35,3.90,0.40,2.10,0.30
2.45,4.20,0.40,2.60,0.30
2.55,4.90,0.40,3.20,0.30
2.65,5.20,0.40,3.40,0.30
2.75,5.80,0.40,3.70,0.30
2.85,6.70,0.40,4.50,0.30
2.95,6.90,0.40,4.80,0.30
3.05,7.20,0.40,5.00,0.30
3.15,7.40,0.40,5.00,0.30
3.25,7.50,0.40,5.00,0.30
3.35,7.50,0.40,4.90,0.30
3.45,7.40,0.40,4.80,0.30
3.55,7.00,0.40,4.40,0.30
3.72,6.00,0.50,4.10,0.40
3.92,5.95,0.50,4.30,0.40
4.62,4.90,0.50,3.60,0.40
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


def cumulative_phi_time(t: np.ndarray, phi: np.ndarray, diss: np.ndarray,
                        t_end: float) -> float:
    """Cumulative duration of (is_phiDepleted AND NOT isDissolved) over [t[0], t_end].

    Left-rectangle rule: the snapshot at t[k] reflects the shell structure
    used for the segment t[k] -> t[k+1] (snapshots saved before ODE integration,
    see trinity_reader.py:93-99), so mask[k] applies for the whole segment.

    Dissolved-shell guard: shell_structure_modified.py:412 force-sets
    is_phiDepleted=True whenever isDissolved=True, but a dissolved shell has
    dispersed into the ISM — no neutral exterior, no PDR. Under the current
    code flow only the final reconciliation snap carries isDissolved=True
    (set post-save in phases 1c/2 termination blocks), and its mask is
    never indexed by this loop; this is therefore a defensive guard against
    future changes that might land a dissolved snapshot mid-run.
    """
    if len(t) < 2:
        return 0.0
    mask = np.asarray(phi, dtype=bool) & ~np.asarray(diss, dtype=bool)
    total = 0.0
    for k in range(len(t) - 1):
        a = t[k]
        if a >= t_end:
            break
        b = min(t[k + 1], t_end)
        if mask[k]:
            total += b - a
    return total


def collect_run(run_dir: Path, show_tau_pdr: bool = False) -> dict:
    data_path = run_dir / "dictionary.jsonl"
    if not data_path.exists():
        data_path = run_dir / "dictionary.json"
    out = TrinityOutput.open(data_path)

    t   = np.asarray(out.get("t_now"),  dtype=float)
    R2  = np.asarray(out.get("R2"),     dtype=float)
    v2  = np.asarray(out.get("v2"),     dtype=float)

    # Recollapse detection: any negative velocity over the trajectory.
    # SimulationEndCode.SHELL_COLLAPSED is the terminal flag, but a run
    # in progress can turn around without ever crossing coll_r, so v2<0
    # is the broader (and physically correct) signal that the shell will
    # not emerge in Pedrini's sense.
    recollapse = bool(np.any(v2 < 0))

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

    row: dict = {
        "run_name":   run_dir.name,
        "mCloud":     mCloud,
        "sfe":        sfe,
        "nCore_cgs":  nCore_cgs,
        "M_star":     M_star,
        "tau_TOT":    tau_TOT,
        "breakout":   breakout,
        "recollapse": recollapse,
        "end_reason": raw_reason,
    }
    if show_tau_pdr:
        phi = np.asarray(
            [bool(x) for x in out.get("is_phiDepleted", as_array=False)]
        )
        diss = np.asarray(
            [bool(x) for x in out.get("isDissolved", as_array=False)]
        )
        row["tau_PDR"] = cumulative_phi_time(t, phi, diss, tau_TOT)
    return row


def collect_all(sweep_dir: Path, show_tau_pdr: bool = False) -> list[dict]:
    sim_files = find_all_simulations(sweep_dir)
    rows = []
    for i, f in enumerate(sim_files, 1):
        print(f"[pedrini_tau] ({i}/{len(sim_files)}) {f.parent.name}")
        rows.append(collect_run(f.parent, show_tau_pdr=show_tau_pdr))
    return rows


# ---------------------------------------------------------------------------
# Merge-mode helpers
# ---------------------------------------------------------------------------

def _round_sig(x: float, n: int = 6) -> float:
    """Round to n significant figures so float keys can be matched reliably."""
    if x == 0:
        return 0.0
    return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def _match_key(row: dict) -> tuple:
    """Identity used to pair runs across two sweep folders.

    (mCloud, sfe, nCore_cgs) covers the plot's x-axis, colour and the
    underlying density input — the params the user actually sweeps. Anything
    else (dens_profile, densPL_alpha, densBE_Omega, ...) is allowed to differ,
    which is the whole point of the merge mode.
    """
    return (_round_sig(row["mCloud"]),
            _round_sig(row["sfe"]),
            _round_sig(row["nCore_cgs"]))


def pair_runs(rows_a: list[dict], rows_b: list[dict]):
    """Return (matched_a, matched_b, unmatched_a, unmatched_b).

    matched_a[i] and matched_b[i] share the same _match_key. Unmatched rows
    are kept around so main() can announce them; they aren't plotted.
    """
    keys_a = {_match_key(r): r for r in rows_a}
    keys_b = {_match_key(r): r for r in rows_b}
    common = set(keys_a) & set(keys_b)
    matched_a = [keys_a[k] for k in sorted(common)]
    matched_b = [keys_b[k] for k in sorted(common)]
    unmatched_a = [r for r in rows_a if _match_key(r) not in common]
    unmatched_b = [r for r in rows_b if _match_key(r) not in common]
    return matched_a, matched_b, unmatched_a, unmatched_b


def folder_dens_profile_label(sweep_dir: Path) -> str:
    """Construct a legend label from the first .param's density-profile settings.

    Reads dens_profile (densPL / densBE) and the relevant shape knob
    (densPL_alpha or densBE_Omega) from any one run in the sweep. Falls back
    to the sweep directory name when no .param can be parsed.
    """
    try:
        first_param = next(sweep_dir.rglob("*.param"))
    except StopIteration:
        return sweep_dir.name
    params = parse_param_file(first_param)
    profile = params.get("dens_profile")
    if profile == "densPL":
        return fr"densPL ($\alpha={params.get('densPL_alpha', '?')}$)"
    if profile == "densBE":
        return fr"densBE ($\Omega={params.get('densBE_Omega', '?')}$)"
    return profile or sweep_dir.name


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_summary_csv(rows: list[dict], out_path: Path,
                      show_tau_pdr: bool = False) -> None:
    fieldnames = [
        "run_name", "mCloud_Msun", "sfe", "nCore_cm-3", "M_star_Msun",
        "tau_TOT_Myr",
    ]
    if show_tau_pdr:
        fieldnames.append("tau_PDR_Myr")
    fieldnames += ["breakout_flag", "recollapse_flag", "end_reason"]

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in rows:
            row = [
                r["run_name"], r["mCloud"], r["sfe"], r["nCore_cgs"],
                r["M_star"], r["tau_TOT"],
            ]
            if show_tau_pdr:
                row.append(r["tau_PDR"])
            row += [r["breakout"], r["recollapse"], r["end_reason"]]
            w.writerow(row)


def load_pedrini_csv(source: str | Path):
    """Pedrini+2026 reference data.

    `source` is either the literal string ``"mock"`` (use the embedded
    MOCK_PEDRINI_CSV) or a path to a CSV file with columns
    log_Mstar, tau_TOT, tau_TOT_err (and optionally tau_PDR, tau_PDR_err
    when `--show_tau_pdr` is set). Errors are interpreted as 1-sigma
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


def _sfe_color_mapping(rows: list[dict], style: str):
    """Build (cmap, norm, lookup) for sfe colouring.

    style="continuous": viridis with LogNorm across the data range.  When the
        sweep has a single unique sfe, LogNorm collapses, so fall back to a
        ±10% window so the colorbar still renders (and matches the single
        plotted colour).
    style="discrete": Wong palette as a ListedColormap, one band per unique
        sfe, indexed 0..N-1 via a BoundaryNorm.
    `lookup(sfe)` returns the matplotlib colour to use when plotting a row.
    """
    unique_sfe = sorted({float(r["sfe"]) for r in rows})
    if style == "discrete":
        cmap = ListedColormap([WONG[i % len(WONG)] for i in range(len(unique_sfe))])
        norm = BoundaryNorm(np.arange(len(unique_sfe) + 1) - 0.5, cmap.N)
        idx = {s: i for i, s in enumerate(unique_sfe)}
        lookup = lambda sfe: cmap(norm(idx[float(sfe)]))
        return cmap, norm, lookup, unique_sfe
    if style == "continuous":
        cmap = plt.cm.viridis
        if len(unique_sfe) > 1:
            norm = LogNorm(vmin=unique_sfe[0], vmax=unique_sfe[-1])
        else:
            v = unique_sfe[0]
            norm = LogNorm(vmin=v * 0.9, vmax=v * 1.1)
        lookup = lambda sfe: cmap(norm(float(sfe)))
        return cmap, norm, lookup, unique_sfe
    raise ValueError(f"unknown colourbar style: {style!r}")


def _size_legend_ticks(m_min: float, m_max: float) -> list[float]:
    """Smallest and largest log10(mCloud) values for the size legend.

    Both snapped to the nearest half-dex.  Single-mCloud sweeps fall back
    to that one value (formatted via _fmt_log_m).
    """
    log_lo, log_hi = np.log10(m_min), np.log10(m_max)
    if log_hi <= log_lo:
        return [log_lo]
    smallest = round(log_lo * 2) / 2.0
    biggest  = round(log_hi * 2) / 2.0
    return sorted({smallest, biggest})


def _fmt_log_m(log_m: float) -> str:
    """Format a log10(mCloud) value as 10^k (integer k) or 10^k.5 (half-dex).

    Off-grid values (single-mCloud fallback) print the raw log to one decimal,
    so the label still reflects the actual mass.
    """
    if abs(log_m - round(log_m)) < 1e-9:
        return fr"10^{{{int(round(log_m))}}}"
    return fr"10^{{{log_m:.1f}}}"


def make_plot(rows: list[dict], pedrini_df, out_pdf: Path,
              show_tau_pdr: bool = False,
              colourbar: str = "continuous") -> None:
    plt.style.use(str(STYLE_PATH))

    # Defensive: recollapse runs should already have been filtered out by
    # main(), but guard here too so make_plot is safe to call directly.
    rows = [r for r in rows if not r.get("recollapse")]
    if not rows:
        print("[pedrini_tau] No non-recollapsing runs to plot.")
        return

    masses = [r["mCloud"] for r in rows]
    m_min, m_max = min(masses), max(masses)

    sfe_cmap, sfe_norm, sfe_color, unique_sfe = _sfe_color_mapping(rows, colourbar)
    REF_COLOR  = "k"       # Pedrini+2026 reference data

    fig, ax = plt.subplots()

    # tau_TOT marker recipe (recollapse already dropped):
    #   - breakout:        filled circle, solid edge
    #   - lower limit:     open circle,   dashed edge  (still going at stop_t)
    # tau_PDR markers echo the same edge-style convention with a square so
    # the TOT/PDR axes stay visually independent.
    LW_SOLID  = 1.0
    LW_DASHED = 1.2
    for r in rows:
        x = np.log10(r["M_star"])
        ms = _marker_size(r["mCloud"], m_min, m_max)
        s_area = ms ** 2  # scatter `s` is area in pts^2; plot `markersize` is diameter in pts
        color = sfe_color(r["sfe"])
        breakout = r["breakout"]
        alpha = r.get("_alpha", 1.0)  # merge mode pre-tags rows with an alpha
        if breakout:
            ax.scatter([x], [r["tau_TOT"]], s=s_area, c=[color],
                       marker="o", edgecolors="k", linewidths=LW_SOLID,
                       alpha=alpha)
            if show_tau_pdr:
                ax.scatter([x], [r["tau_PDR"]], s=s_area,
                           facecolors="none", edgecolors=[color],
                           marker="s", linewidths=LW_SOLID, alpha=alpha)
        else:
            ax.scatter([x], [r["tau_TOT"]], s=s_area,
                       facecolors="none", edgecolors=[color],
                       marker="o", linestyles="--", linewidths=LW_DASHED,
                       alpha=alpha)
            if show_tau_pdr:
                ax.scatter([x], [r["tau_PDR"]], s=s_area,
                           facecolors="none", edgecolors=[color],
                           marker="s", linestyles="--", linewidths=LW_DASHED,
                           alpha=alpha)

    if pedrini_df is not None:
        ax.errorbar(pedrini_df["log_Mstar"], pedrini_df["tau_TOT"],
                    yerr=pedrini_df["tau_TOT_err"],
                    marker="o", mfc=REF_COLOR, mec=REF_COLOR, ecolor=REF_COLOR,
                    linestyle="none", markersize=7, capsize=2)
        if show_tau_pdr and "tau_PDR" in pedrini_df.columns:
            ax.errorbar(pedrini_df["log_Mstar"], pedrini_df["tau_PDR"],
                        yerr=pedrini_df["tau_PDR_err"],
                        marker="s", mfc="none", mec=REF_COLOR, ecolor=REF_COLOR,
                        linestyle="none", markersize=7, capsize=2)

    ax.set_xlabel(r"$\log_{10}\!\left(M_\star\right)$ [$M_\odot$]")
    if show_tau_pdr:
        ax.set_ylabel(r"$\tau$ [Myr]")
    else:
        ax.set_ylabel(r"$\tau_{\rm disp}$ [Myr]")

    # Pad axis limits so size-inflated markers don't touch the spines.
    # y always anchors at 0; top adds 15% headroom over the largest plotted
    # value (including Pedrini errorbars when overlaid).  x pads 7% of range
    # on each side, with a 0.2-dex floor for single-mCloud sweeps.
    all_tau = [r["tau_TOT"] for r in rows]
    if show_tau_pdr:
        all_tau += [r["tau_PDR"] for r in rows]
    if pedrini_df is not None:
        all_tau += list(pedrini_df["tau_TOT"] + pedrini_df["tau_TOT_err"])
        if show_tau_pdr and "tau_PDR" in pedrini_df.columns:
            all_tau += list(pedrini_df["tau_PDR"] + pedrini_df["tau_PDR_err"])
    y_top = max(all_tau) if all_tau else 1.0
    ax.set_ylim(0.0, y_top * 1.15)

    all_x = [np.log10(r["M_star"]) for r in rows]
    if pedrini_df is not None:
        all_x += list(pedrini_df["log_Mstar"])
    x_lo, x_hi = min(all_x), max(all_x)
    x_pad = max(0.2, (x_hi - x_lo) * 0.07)
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)

    # sfe colourbar on the right of the axes (replaces the per-sfe legend
    # swatches).  Continuous mode uses viridis with LogNorm; discrete mode
    # uses one Wong-palette band per unique sfe value, ticks centred on
    # each band.  In continuous mode the colourbar ticks are anchored at
    # the actual sweep sfe values so each plotted point is identifiable.
    sfe_mappable = plt.cm.ScalarMappable(cmap=sfe_cmap, norm=sfe_norm)
    sfe_mappable.set_array([])
    if colourbar == "discrete":
        # All bands ticked — each is a distinct sweep value worth labelling.
        cbar = fig.colorbar(
            sfe_mappable, ax=ax, location="right",
            fraction=0.04, pad=0.01,
            ticks=np.arange(len(unique_sfe)),
        )
        cbar.set_ticklabels([f"{s:g}" for s in unique_sfe])
    else:
        # Continuous mode: just label the endpoints to keep the axis clean.
        endpoints = [unique_sfe[0], unique_sfe[-1]]
        cbar = fig.colorbar(
            sfe_mappable, ax=ax, location="right",
            fraction=0.04, pad=0.01,
            ticks=endpoints,
        )
        cbar.set_ticklabels([f"{v:g}" for v in endpoints])
    cbar.set_label("sfe")

    # Remaining legend: shape entries (breakout / lower-limit / tau_PDR)
    # and size entries (smallest + largest mCloud only).
    mid_ms = 0.5 * (_marker_size(m_min, m_min, m_max)
                    + _marker_size(m_max, m_min, m_max))

    # Show the breakout/no-breakout shape legend only when both variants
    # are actually present in the data.  In single-variant sweeps the shape
    # carries no information and the entry would just be noise.
    has_breakout    = any(r["breakout"]     for r in rows)
    has_no_breakout = any(not r["breakout"] for r in rows)
    mixed_breakout  = has_breakout and has_no_breakout

    def _lower_limit_proxy(label: str):
        """Open-circle proxy for the 'lower limit' legend entry.

        Plotted points use scatter with linestyles='--' for a dashed edge,
        but a PathCollection-as-legend-handle picks up colormap state in
        practice and renders the swatch in the wrong colour.  A plain
        Line2D open circle still conveys 'lower limit' next to the filled
        breakout marker; the dashed edge on the actual data points carries
        the rest of the signal.
        """
        return Line2D([], [], marker="o", linestyle="none",
                      mfc="none", mec="0.4",
                      markersize=mid_ms, markeredgewidth=LW_DASHED,
                      label=label)

    handles: list = []
    if show_tau_pdr:
        # TOT/PDR distinction is always meaningful in tau_PDR mode.  Edge
        # style encodes breakout-vs-lower-limit; shape encodes TOT-vs-PDR.
        handles += [
            Line2D([], [], marker="o", linestyle="none",
                   mfc="0.4", mec="k", markersize=mid_ms,
                   label=r"$\tau_{\rm disp}$"),
            Line2D([], [], marker="s", linestyle="none",
                   mfc="none", mec="0.4", markersize=mid_ms,
                   label=r"$\tau_{\rm PDR}$"),
        ]
        if mixed_breakout:
            handles.append(_lower_limit_proxy("lower limit (stop_t)"))
    elif mixed_breakout:
        handles += [
            Line2D([], [], marker="o", linestyle="none",
                   mfc="0.4", mec="k", markersize=mid_ms,
                   label="breakout"),
            _lower_limit_proxy("lower limit (stop_t)"),
        ]
    for log_m in _size_legend_ticks(m_min, m_max):
        m = 10 ** log_m
        ms = _marker_size(m, m_min, m_max)
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              mfc="0.4", mec="k", markersize=ms,
                              label=fr"$M_{{\rm cloud}}={_fmt_log_m(log_m)}"
                                    r"\,M_\odot$"))
    if pedrini_df is not None:
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              mfc=REF_COLOR, mec=REF_COLOR, markersize=mid_ms,
                              label="Pedrini+2026"))

    # Merge mode: one swatch per folder, alpha matches the data points so the
    # reader can map faded/solid markers in the plot back to the dens-profile
    # label.  Preserves the order in which the folders were tagged in main().
    seen_groups: list[tuple[str, float]] = []
    for r in rows:
        g = r.get("_group")
        if g is None:
            continue
        entry = (g, r.get("_alpha", 1.0))
        if entry not in seen_groups:
            seen_groups.append(entry)
    for label, alpha in seen_groups:
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              mfc="0.4", mec="k", markersize=mid_ms,
                              alpha=alpha, label=label))
    # Park the legend above the axes in a two-column layout.  The
    # colourbar sits on the right, so leaving the strip above the axes
    # for the legend keeps the data area uncluttered.
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, fontsize="small",
              handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)

    fig.savefig(out_pdf, bbox_inches="tight")
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
    ap.add_argument("--merge_dir", type=Path, default=None,
                    help="Optional second sweep directory.  When set, runs in "
                         "the two folders are paired by (mCloud, sfe, nCore); "
                         "the matched pairs are drawn together with --sweep_dir "
                         "at alpha=0.3 (70%% transparent) and --merge_dir at "
                         "alpha=0.8 (20%% transparent).  Output file is "
                         "pedrini_emergence_timescales_merge.pdf.")
    ap.add_argument("--pedrini_csv", type=str, default=None,
                    help="Optional Pedrini+2026 CSV path "
                         "(log_Mstar, tau_TOT, tau_TOT_err, and optionally "
                         "tau_PDR, tau_PDR_err when --show_tau_pdr is set; "
                         "errors are 1-sigma symmetric in Myr). "
                         "Pass 'mock' to use the digitised reference data "
                         "embedded in this script.")
    ap.add_argument("--show_tau_pdr", action="store_true",
                    help="Also compute, write, and plot tau_PDR (cumulative "
                         "is_phiDepleted time integrated over [0, tau_TOT]). "
                         "Off by default.")
    ap.add_argument("--colourbar", choices=["continuous", "discrete"],
                    default="continuous",
                    help="sfe colourbar style.  'continuous' (default): "
                         "viridis with LogNorm, ticks anchored at the "
                         "actual sweep sfe values.  'discrete': Wong "
                         "palette with one band per unique sfe value.")
    args = ap.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    if not sweep_dir.is_dir():
        ap.error(f"--sweep_dir not found: {sweep_dir}")

    rows = collect_all(sweep_dir, show_tau_pdr=args.show_tau_pdr)
    if not rows:
        ap.error(f"No simulations found under {sweep_dir}")

    fig_dir = FIG_DIR / sweep_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    # In merge mode the per-sweep CSVs use folder-suffixed names so they
    # don't overwrite the single-folder run's CSV in the same fig_dir.
    if args.merge_dir is None:
        csv_path = fig_dir / "pedrini_emergence_timescales_summary.csv"
    else:
        csv_path = fig_dir / f"pedrini_emergence_timescales_summary_{sweep_dir.name}.csv"
    # CSV keeps every row (with recollapse_flag) for the audit trail.
    write_summary_csv(rows, csv_path, show_tau_pdr=args.show_tau_pdr)
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

    if args.merge_dir is not None:
        merge_dir = args.merge_dir.resolve()
        if not merge_dir.is_dir():
            ap.error(f"--merge_dir not found: {merge_dir}")
        if merge_dir == sweep_dir:
            print("[pedrini_tau] --merge_dir matches --sweep_dir; "
                  "merge mode will pair every run with itself.")

        rows_b = collect_all(merge_dir, show_tau_pdr=args.show_tau_pdr)
        if not rows_b:
            ap.error(f"No simulations found under {merge_dir}")

        csv_path_b = fig_dir / f"pedrini_emergence_timescales_summary_{merge_dir.name}.csv"
        write_summary_csv(rows_b, csv_path_b, show_tau_pdr=args.show_tau_pdr)
        print(f"Wrote summary: {csv_path_b} ({len(rows_b)} rows)")

        # Recollapse runs lack a Pedrini-relevant tau_TOT, so drop them
        # BEFORE pairing — otherwise we'd reject a perfectly good run in
        # folder A only because its partner in folder B recollapsed.
        rows_a_ok = [r for r in rows  if not r.get("recollapse")]
        rows_b_ok = [r for r in rows_b if not r.get("recollapse")]
        n_recol_a = len(rows)   - len(rows_a_ok)
        n_recol_b = len(rows_b) - len(rows_b_ok)
        if n_recol_a or n_recol_b:
            print(f"[pedrini_tau] Dropped recollapsing runs before pairing: "
                  f"{n_recol_a} from {sweep_dir.name}, "
                  f"{n_recol_b} from {merge_dir.name}.")

        matched_a, matched_b, unmatched_a, unmatched_b = pair_runs(
            rows_a_ok, rows_b_ok
        )
        if unmatched_a or unmatched_b:
            print(f"[pedrini_tau] Unpaired runs (no (mCloud, sfe, nCore) match):")
            for r in unmatched_a:
                print(f"  - {sweep_dir.name}/{r['run_name']}")
            for r in unmatched_b:
                print(f"  - {merge_dir.name}/{r['run_name']}")
        if not matched_a:
            ap.error("No paired runs found between the two folders — "
                     "check that they share (mCloud, sfe, nCore) values.")

        # alpha values follow the user's spec: 70% transparent for folder1,
        # 20% transparent for folder2.  Labels come from the density-profile
        # settings in each folder's .param.
        label_a = folder_dens_profile_label(sweep_dir)
        label_b = folder_dens_profile_label(merge_dir)
        for r in matched_a:
            r["_alpha"] = 0.3
            r["_group"] = label_a
        for r in matched_b:
            r["_alpha"] = 0.8
            r["_group"] = label_b
        plot_rows = matched_a + matched_b
        pdf_path = fig_dir / "pedrini_emergence_timescales_merge.pdf"
        print(f"[pedrini_tau] Plotting {len(matched_a)} matched pair(s): "
              f"{label_a} (α=0.3) vs {label_b} (α=0.8).")
        make_plot(plot_rows, pedrini_df, pdf_path,
                  show_tau_pdr=args.show_tau_pdr, colourbar=args.colourbar)
        return

    # Recollapse runs don't have a Pedrini-relevant tau_TOT (their shell
    # turned around before emergence); drop them from the plot and announce
    # which ones went.
    recollapsed = [r for r in rows if r.get("recollapse")]
    if recollapsed:
        print(f"[pedrini_tau] Dropped {len(recollapsed)} recollapsing run(s) "
              f"from the plot (v2<0 seen in trajectory):")
        for r in recollapsed:
            print(f"  - {r['run_name']}")
    plot_rows = [r for r in rows if not r.get("recollapse")]
    if not plot_rows:
        print("[pedrini_tau] All runs recollapsed — nothing to plot.")
        return

    pdf_path = fig_dir / "pedrini_emergence_timescales.pdf"
    make_plot(plot_rows, pedrini_df, pdf_path, show_tau_pdr=args.show_tau_pdr,
              colourbar=args.colourbar)


if __name__ == "__main__":
    main()
