#!/usr/bin/env python3
"""theta5s analysis — the f_A (cooling_boost_fA) all-9-config validation (Phase 4).

Reads runs/data/theta5s_summary.csv (harvested from the 81-arm HPC matrix; no sims here) and
answers the Phase-4 question of SOURCE_TERM_DESIGN.md §3:

  does a single f_A fire the cooling_balance trigger across the 7 FIREABLE configs (the
  multiplier's gold standard was window [4, 4.5] at 7/7), while the two controls (small_1e6
  route-a, fail_repro PdV) pass UNCHANGED? "works on ALL configs" is per-CLASS, not "all fire".

Deliverables:
  1. data/theta5s_fire_map.csv  — per (config, f_A) outcome + theta_max; whole-band-f_A verdict
     over the 7 FIREABLE configs (controls excluded).
  2. theta5s_fire_map.png       — the outcome matrix (read off: is there a whole-band f_A?).
  3. theta5s_theta_rise.png     — theta_max vs f_A per config.
  4. data/theta5s_collapse_law.csv — the source-edition collapse-law fit f_fire = A*(0.95/theta0)^p.
     REGISTERED PREDICTION (SOURCE_TERM_DESIGN §3 Phase 4): p_source ~ 1/0.30 ~ 3.3 (vs the
     multiplier's 1.82); per-config screen exponents vary 0.19-0.30 so the fit may be looser
     than the multiplier's rms 0.064 dex — report the rms either way. If the LIVE p is <<2 or
     >>4.5 the screen exponent did not survive coupling: STOP and write it up (Phase-6 tree row),
     do NOT tune around it.

The (iii) dMdt-suppression fidelity measurement (theta5s_dmdt_suppression.csv, El-Badry Eq 47
trend) needs the RAW dictionary.jsonl per arm (bubble_dMdt(t)), not the summary — run it against
$WS/outputs/theta5s/* on HPC (or downloaded arms) with runs/harvest_dmdt_suppression.py.

REPRODUCE (after ./sync_theta5s.sh down drops the summary):
    python docs/dev/transition/pdv-trigger/data/make_theta5s_analysis.py
"""

import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _PDV)
from _stamp import stamp  # noqa: E402
from _trinity_style import use_trinity_style  # noqa: E402

use_trinity_style()
import matplotlib.pyplot as plt  # noqa: E402

# the implicit-phase handoff streak cap (run_energy_implicit_phase.NO_ROOT_HANDOFF_STREAK):
# a non-fired momentum run with n_impl exactly here is the condensation handoff. Phase 1 found
# f_A has NO reachable condensation edge (no dMdt<=0 even at fA=512), so CONDENSE is NOT expected
# for f_A — a DRAIN or NOFIRE is the non-fire fate. A CONDENSE here would be worth a second look.
HANDOFF_STREAK = 50

ORDER = [
    "fail_repro", "small_1e6", "pl2_steep", "be_sphere", "large_diffuse_lowsfe",
    "midrange_pl0", "small_dense_highsfe", "simple_cluster", "normal_n1e3",
]  # by theta0, outliers first (same as theta5k figures)
THETA0 = {
    "simple_cluster": 0.676, "small_dense_highsfe": 0.717, "midrange_pl0": 0.636,
    "large_diffuse_lowsfe": 0.535, "be_sphere": 0.529, "pl2_steep": 0.511,
    "small_1e6": 0.297, "fail_repro": 0.003, "normal_n1e3": 1.047,
}
CONTROLS = ("fail_repro", "small_1e6")               # per-CLASS acceptance: NOT expected to fire
FIREABLE = [c for c in ORDER if c not in CONTROLS]    # the 7 the whole-band question is about
SHORT = {"large_diffuse_lowsfe": "large_diffuse", "small_dense_highsfe": "small_dense"}

OUTCOLOR = {"FIRED": "#0072B2", "CONDENSE": "#CC79A7", "DRAIN": "#E69F00", "NOFIRE": "#009E73"}
OUTMARK = {"FIRED": "o", "CONDENSE": "d", "DRAIN": "v", "NOFIRE": "s"}
OUTLABEL = {
    "FIRED": "fires cooling_balance ($\\theta\\geq0.95$)",
    "CONDENSE": "condensation handoff (NOT expected for f_A — Phase 1)",
    "DRAIN": "momentum/dissolve WITHOUT firing",
    "NOFIRE": "stays energy-driven (healthy to 5 Myr)",
}


def f_of_mode(mode):
    if mode == "none":
        return 1.0
    return float(mode[2:].replace("p", ".")) if mode.startswith("fa") else None


def read_summary(path):
    with open(path) as fh:
        return list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))


def classify(r):
    try:
        th = float(r["theta_max"])
    except (ValueError, TypeError):
        th = float("nan")
    if r["fired_cooling_balance"] == "True":
        return "FIRED", th
    if r["reached_momentum"] == "True":
        if int(r["n_impl"]) == HANDOFF_STREAK:
            return "CONDENSE", th
        return "DRAIN", th
    return "NOFIRE", th


def load():
    rows = read_summary(os.path.join(_PDV, "runs", "data", "theta5s_summary.csv"))
    cells = {}
    for r in rows:
        cfg, mode = r["run_name"].rsplit("__", 1)
        f = f_of_mode(mode)
        if f is not None:
            cells[(cfg, f)] = (*classify(r), r)
    return cells


def write_fire_map(cells):
    fs = sorted({f for _, f in cells})
    whole = [f for f in fs if all(cells.get((c, f), ("?",))[0] == "FIRED" for c in FIREABLE)]
    n_cond = sum(1 for (o, _, _) in cells.values() if o == "CONDENSE")
    # control check: neither control should FIRE at any f_A
    ctrl_fires = [(c, f) for (c, f), (o, _, _) in cells.items() if c in CONTROLS and o == "FIRED"]
    path = os.path.join(_HERE, "theta5s_fire_map.csv")
    with open(path, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        whole_txt = (str(whole) if whole else
                     "NONE — no single f_A fires all 7 fireable configs (multiplier window "
                     "[4,4.5] did 7/7; Phase-6 decision tree)")
        fh.write(
            "# f_A matrix (stop_t=5, theta_max from dictionary accepted rows), single-knob. "
            f"Whole-band f_A (all 7 FIREABLE FIRED): {whole_txt}. "
            f"CONDENSE arms: {n_cond} (Phase 1 predicts ~0 for f_A). "
            f"Control fires (SHOULD be empty; a fire here is a BUG, not a pass): {ctrl_fires}. "
            "Do NOT count CONDENSE/DRAIN as theta transitions.\n")
        w = csv.writer(fh)
        w.writerow(["config"] + [f"fa{f:g}" for f in fs])
        for cfg in ORDER:
            row = [cfg]
            for f in fs:
                out, th, _ = cells.get((cfg, f), (None, None, None))
                row.append(f"{out}:{th:.3f}" if out and th is not None and math.isfinite(th)
                           else (out or ""))
            w.writerow(row)
    print(f"wrote {path}  (whole-band f_A: {whole or 'NONE'}; control fires: {ctrl_fires or 'none'})")
    return fs, whole


def write_collapse_law(cells, fs):
    """Fit log(f_fire) = log(A) + p*log(0.95/theta0) over the FIREABLE configs that fired.
    f_fire = smallest f_A with a FIRED outcome. Registered prediction: p ~ 3.3."""
    pts = []  # (config, theta0, f_fire)
    for cfg in FIREABLE:
        fired_fs = sorted(f for f in fs if cells.get((cfg, f), ("?",))[0] == "FIRED")
        if fired_fs and cfg in THETA0 and THETA0[cfg] < 0.95:
            pts.append((cfg, THETA0[cfg], fired_fs[0]))
    path = os.path.join(_HERE, "theta5s_collapse_law.csv")
    with open(path, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        if len(pts) < 2:
            fh.write("# too few fired configs for a fit (need >=2 with theta0<0.95).\n")
            w = csv.writer(fh); w.writerow(["config", "theta0", "f_fire"])
            w.writerows([(c, t, f) for c, t, f in pts])
            print(f"wrote {path}  (only {len(pts)} fired configs — no fit)")
            return None
    # least-squares fit y = a + p*x, x = log10(0.95/theta0), y = log10(f_fire)
    xs = [math.log10(0.95 / t) for _, t, _ in pts]
    ys = [math.log10(f) for _, _, f in pts]
    n = len(xs); sx = sum(xs); sy = sum(ys); sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    p = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - p * sx) / n
    resid = [ys[i] - (a + p * xs[i]) for i in range(n)]
    rms = math.sqrt(sum(r * r for r in resid) / n)
    A = 10 ** a
    with open(path, "a", newline="") as fh:
        fh.write(f"# fit f_fire = A*(0.95/theta0)^p : A={A:.3f}, p={p:.3f}, rms={rms:.4f} dex "
                 f"(n={n}). REGISTERED PREDICTION p_source ~ 3.3 (multiplier was 1.82; rms 0.064). "
                 f"{'CONSISTENT' if 2 <= p <= 4.5 else 'OUT OF RANGE -> STOP + write up (Phase-6)'}.\n")
        w = csv.writer(fh); w.writerow(["config", "theta0", "f_fire", "resid_dex"])
        for i, (c, t, f) in enumerate(pts):
            w.writerow([c, t, f, f"{resid[i]:.4f}"])
    print(f"wrote {path}  (p={p:.3f}, A={A:.3f}, rms={rms:.4f} dex, n={n}; predicted p~3.3)")
    return dict(A=A, p=p, rms=rms, n=n, pts=pts)


def fig_fire_map(cells, fs):
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ys = {cfg: i for i, cfg in enumerate(ORDER)}
    for (cfg, f), (out, th, _) in cells.items():
        y = ys[cfg]
        ax.plot(f, y, marker=OUTMARK[out], color=OUTCOLOR[out], ms=9,
                mfc=OUTCOLOR[out] if out == "FIRED" else "white", mew=1.6, zorder=4)
        if out == "FIRED" and th is not None and math.isfinite(th):
            ax.annotate(f"{th:.2f}", (f, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=6.5, color="0.35")
    ax.axhspan(-0.6, 1.5, color="0.9", zorder=0)  # shade the two controls
    ax.text(1.0, -0.45, "controls (must NOT fire)", fontsize=7, color="0.45")
    ax.set_xscale("log"); ax.set_xticks(fs); ax.set_xticklabels([f"{f:g}" for f in fs], fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator()); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([SHORT.get(c, c) for c in ORDER])
    ax.set_ylim(-0.6, len(ORDER) - 0.1); ax.set_xlim(0.9, 40)
    ax.set_xlabel("$f_A$ (cooling_boost_fA)")
    handles = [plt.Line2D([], [], marker=OUTMARK[k], lw=0, ms=8, color=OUTCOLOR[k],
                          mfc=OUTCOLOR[k] if k == "FIRED" else "white", mew=1.5, label=OUTLABEL[k])
               for k in OUTLABEL]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.10), frameon=False)
    ax.set_title("theta5s fire map: outcome per (config, $f_A$)\n"
                 "does a single $f_A$ fire the 7 fireable configs while the controls stay cold?")
    fig.savefig(os.path.join(_PDV, "theta5s_fire_map.png"), bbox_inches="tight")
    plt.close(fig)


def fig_theta_rise(cells, fs):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "0.4"]
    for i, cfg in enumerate(FIREABLE):
        pts = sorted((f, cells[(cfg, f)][1], cells[(cfg, f)][0]) for f in fs if (cfg, f) in cells)
        pts = [(f, th, o) for f, th, o in pts if th is not None and math.isfinite(th)]
        if not pts:
            continue
        xs = [p[0] for p in pts]; ths = [p[1] for p in pts]
        ax.plot(xs, ths, "-", color=palette[i % len(palette)], lw=1.3, alpha=0.75, zorder=3)
        for f, th, out in pts:
            ax.plot(f, th, marker=OUTMARK[out], color=palette[i % len(palette)], ms=6.5,
                    mfc=palette[i % len(palette)] if out == "FIRED" else "white", mew=1.3, zorder=4)
        ax.annotate(SHORT.get(cfg, cfg), (xs[-1], ths[-1]), textcoords="offset points",
                    xytext=(7, 0), fontsize=7.5, color=palette[i % len(palette)], va="center")
    ax.axhline(0.95, color="0.3", lw=1.0, ls="--")
    ax.text(1.0, 0.965, "trigger $\\theta=0.95$", fontsize=8, color="0.3", ha="left")
    ax.set_xscale("log"); ax.set_xticks(fs); ax.set_xticklabels([f"{f:g}" for f in fs], fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator()); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(0.9, 42); ax.set_xlabel("$f_A$")
    ax.set_ylabel("$\\theta_{\\max}$ (5 Myr, dictionary rows)")
    ax.set_title("theta5s: $\\theta_{\\max}$ vs $f_A$ — filled = fired\n"
                 "sub-linear rise (screen: $\\theta\\propto f_A^{0.3}$); laggards fire near the grid top")
    fig.savefig(os.path.join(_PDV, "theta5s_theta_rise.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    cells = load()
    fs, whole = write_fire_map(cells)
    write_collapse_law(cells, fs)
    fig_fire_map(cells, fs)
    fig_theta_rise(cells, fs)
    counts = {}
    for out, _, _ in cells.values():
        counts[out] = counts.get(out, 0) + 1
    print("outcome counts:", counts)
    print("wrote theta5s_fire_map.png, theta5s_theta_rise.png ->", _PDV)


if __name__ == "__main__":
    main()
