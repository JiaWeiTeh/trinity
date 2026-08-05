#!/usr/bin/env python3
"""Figures for the phase-1a fix decision dossier.

Reads ONLY committed CSVs in ../data (no runs) and writes PNGs into ../figures.
These are the plots behind the maintainer decision recorded in PLAN.md §4/§8:
does the early-phase shift on published configs get accepted, or does the
pre-registered bar get re-sited?

    python docs/dev/phase1a-init/harness/make_decision_figures.py

Per docs/dev/CLAUDE.md: Agg backend, text.usetex=False, dpi~140.
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")

# Palette shared with the dossier page so figures and page read as one system.
INK = "#161b22"
MUTED = "#6b7684"
GRID = "#d8dbe0"
STOCK = "#b4553a"     # the artifact / pre-fix arm
FIXED = "#2b7a8c"     # the converged / post-fix arm
OBS = "#3f7d52"       # observation
WARN = "#c98a2b"

plt.rcParams.update({
    "text.usetex": False,
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

MYR2YR = 1e6


def load(stem):
    with open(os.path.join(DATA, f"{stem}.csv")) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def series(stem, key="R2"):
    rows = load(stem)
    return ([float(r["t_now"]) * MYR2YR for r in rows], [float(r[key]) for r in rows])


def interp(xs, ys, x):
    if x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i] >= x:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return None


def save(fig, name):
    fig.tight_layout()
    path = os.path.join(FIGS, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.normpath(path)}")


# ---------------------------------------------------------------- fingerprint
def fig_fingerprint():
    """Segment-1 exit velocity is the same number across four decades of mass."""
    masses, vexit = [], []
    for m, stem in ((3e3, "mass_3e3"), (3e4, "mass_3e4"), (3e5, "mass_3e5"), (3e6, "mass_3e6")):
        rows = load(stem)
        masses.append(m)
        vexit.append(float(rows[1]["v2_kms"]))
    probe = float(load("m43_probe")[1]["v2_kms"])
    ablated = float(load("m43_noapprox")[1]["v2_kms"])

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.axhline(722.82, color=STOCK, lw=1, ls="--", alpha=0.7)
    ax.plot(masses, vexit, "o-", color=STOCK, lw=1.6, ms=6, label="stock: segment-1 exit velocity")
    ax.plot([300], [probe], "D", color=STOCK, ms=7, label="stock: M43 probe (mCloud=300)")
    ax.plot([300], [ablated], "s", color=MUTED, ms=6,
            label="override deleted, stock segments (2429 km/s)")
    ax.set_xscale("log")
    ax.set_xlabel("cloud mass  $M_\\mathrm{cloud}$  [$M_\\odot$]")
    ax.set_ylabel("velocity leaving segment 1  [km/s]")
    ax.set_ylim(0, 2800)
    ax.annotate("722.82 km/s — identical for every run\n"
                "$v_\\mathrm{exit}=v_0-10^8\\times$SEGMENT_DURATION",
                xy=(1e4, 722.82), xytext=(1.4e3, 1180), color=STOCK, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=STOCK, lw=0.9))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("The fingerprint: an arithmetic result, not a physical one", loc="left",
                 fontsize=10.5, color=INK, pad=8)
    save(fig, "decision_fingerprint.png")


# ------------------------------------------------------------------ M43 truth
def fig_m43():
    ts, rs = series("m43_probe")
    tf, rf = series("g2_m43_prod")
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(ts, rs, color=STOCK, lw=1.8, label="stock (fixed 30-yr segments + override)")
    ax.plot(tf, rf, color=FIXED, lw=1.8, label="fixed (age-scaled segments, no override)")
    # observation box
    ax.add_patch(plt.Rectangle((17000, 0.142), 4000, 0.164 - 0.142,
                               facecolor=OBS, alpha=0.25, edgecolor=OBS, lw=1.2, zorder=5))
    ax.annotate("M43 observed\n0.153 pc at 1.7–2.1e4 yr", xy=(19000, 0.145),
                xytext=(2.2e4, 0.021), color=OBS, fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=OBS, lw=0.9))
    ax.axhline(0.153, color=OBS, lw=0.8, ls=":", alpha=0.8)
    for t, col in ((620, STOCK), (13501, FIXED)):
        ax.plot([t], [0.153], "o", color=col, ms=6, zorder=6)
    ax.annotate("", xy=(660, 0.153), xytext=(12700, 0.153),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0))
    ax.text(2900, 0.166, "crossed 21.8x too early", color=MUTED, fontsize=8.5, ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5, 3e4); ax.set_ylim(2e-3, 0.6)
    ax.set_xlabel("time since star formation  [yr]")
    ax.set_ylabel("shell radius  $R_2$  [pc]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("M43: the fix moves the model onto the observation", loc="left",
                 fontsize=10.5, color=INK, pad=8)
    save(fig, "decision_m43.png")


# ------------------------------------------------------------- the decision
def fig_decision():
    """dR2/R2 (fixed vs stock) at matched t, the number the decision rests on."""
    pairs = [("simple_cluster  (nCore=1e3)", "g2_1myr_simple_stock", "g2_1myr_simple_fixed", FIXED),
             ("GMC control  (nCore=1e3, uniform)", "gmc_control", "g2_gmc_fixed_full", INK),
             ("f1edge_lowdens  (nCore=1e2)", "g2_lowdens_stock", "g2_lowdens_fixed", OBS),
             ("f1edge_hidens  (nCore=1e6)", "g2_longhidens_stock", "g2_longhidens_fixed", STOCK)]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.axhspan(-1, 1, color=MUTED, alpha=0.13, zorder=0)
    ax.text(2.2e6, -3.1, "pre-registered bar\n|ΔR₂| < 1%", color=MUTED, fontsize=8.5,
            ha="right", va="top")
    ax.axhline(0, color=MUTED, lw=0.8)
    for label, s_stem, f_stem, col in pairs:
        ts, rs = series(s_stem)
        tf, rf = series(f_stem)
        # ends at the latest time BOTH arms cover — the "or the end of the run"
        # clause of the adopted bar (PLAN.md §4)
        first, last = max(ts[0], tf[0]), min(ts[-1], tf[-1])
        grid = [t for t in (1e2, 3e2, 1e3, 3e3, 5e3, 8e3, 1e4, 1.5e4, 2e4, 3e4, 5e4, 8e4,
                            1.2e5, 3e5, 5e5, 1e6, 2e6) if first <= t <= last]
        if grid and grid[-1] < last:
            grid.append(last)
        d = [(t, 100 * (interp(tf, rf, t) - interp(ts, rs, t)) / interp(ts, rs, t)) for t in grid]
        ax.plot([p[0] for p in d], [p[1] for p in d], "o-", color=col, lw=1.7, ms=4.5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("time since star formation  [yr]")
    ax.set_ylabel("ΔR₂ / R₂   fixed vs stock  [%]")
    ax.set_ylim(-32, 6)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("The early transient changes — and then the trajectories reconverge",
                 loc="left", fontsize=10.5, color=INK, pad=8)
    save(fig, "decision_shift.png")


# ------------------------------------------------------------------ G3 slope
def fig_slope():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for stem, col, lab in (("gmc_control", STOCK, "stock"), ("g2_gmc_prod", FIXED, "fixed")):
        t, r = series(stem)
        pts = []
        for i in range(1, len(t) - 1):
            if min(t[i - 1], r[i - 1], t[i + 1], r[i + 1]) <= 0:
                continue
            s = (math.log(r[i + 1]) - math.log(r[i - 1])) / (math.log(t[i + 1]) - math.log(t[i - 1]))
            pts.append((t[i], s))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.5, label=lab, alpha=0.9)
    ax.axhline(0.6, color=INK, lw=1.1, ls="--")
    ax.text(2.5e4, 0.615, "Weaver energy-driven law:  $R\\propto t^{3/5}$", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 2e5); ax.set_ylim(0.15, 0.8)
    ax.set_xlabel("time since star formation  [yr]")
    ax.set_ylabel("local slope  $d\\ln R_2/d\\ln t$")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Independent check: the fix is on the similarity solution from the first decade",
                 loc="left", fontsize=10.5, color=INK, pad=8)
    save(fig, "decision_slope.png")


# ------------------------------------------------------------ eps convergence
def fig_eps():
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    eps, vals = [0.3, 0.1, 0.03], []
    for stem in ("eps0.3_m43", "g2_m43_prod", "eps0.03_m43"):
        t, r = series(stem)
        vals.append(interp(t, r, 21000))
    ax.plot(eps, vals, "o-", color=FIXED, lw=1.7, ms=7)
    for e, v in zip(eps, vals):
        ax.annotate(f"{v:.5f} pc", xy=(e, v), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("segment fraction  phase1a_segFrac   (refining →)")
    ax.set_ylabel("$R_2$ at the observed age  [pc]")
    ax.set_ylim(0.1955, 0.1968)
    ax.set_title("The answer is converged: refining 10x moves it 0.11%", loc="left",
                 fontsize=10.5, color=INK, pad=8)
    save(fig, "decision_eps.png")


# ------------------------------------------------------------------- E8b ramp
def fig_e8b():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for act, abl, col, lab in (("g2_gmc_prod", "e8b_gmc_noramp", STOCK, "GMC control"),
                               ("g2_m43_prod", "e8b_m43_noramp", FIXED, "M43 probe")):
        ta, ra = series(act)
        tb, rb = series(abl)
        grid = [t for t in (10, 20, 30, 50, 70, 100, 200, 300, 1e3, 3e3, 1e4, 2.1e4, 3e4, 8e4)
                if t <= min(ta[-1], tb[-1]) and t >= max(ta[0], tb[0])]
        d = [(t, 100 * (interp(tb, rb, t) - interp(ta, ra, t)) / interp(ta, ra, t)) for t in grid]
        ax.plot([p[0] for p in d], [p[1] for p in d], "o-", color=col, lw=1.7, ms=4.5, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhspan(-0.1, 0.1, color=MUTED, alpha=0.15)
    ax.text(2.2e4, 0.35, "0.1% noise floor", fontsize=8.5, color=MUTED)
    ax.plot([0.26], [0], "X", color=WARN, ms=12, zorder=6)
    ax.annotate("f1edge_hidens stalls here:\n4 snapshots in 90 min of wall clock,\n"
                "so there is no trajectory to draw",
                xy=(0.3, -0.15), xytext=(0.62, -4.6), color=WARN, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=WARN, lw=0.9))
    ax.set_xscale("log")
    ax.set_xlabel("time since star formation  [yr]")
    ax.set_ylabel("ΔR₂ / R₂   ramp removed  [%]")
    ax.set_ylim(-6.8, 1.6)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("E8b: removing the R1 ramp — small where it runs, fatal where it doesn't",
                 loc="left", fontsize=10.5, color=INK, pad=8)
    save(fig, "decision_e8b.png")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_fingerprint()
    fig_m43()
    fig_decision()
    fig_slope()
    fig_eps()
    fig_e8b()
