#!/usr/bin/env python3
"""Figures for `pdvtrigger_report.html (was phase6_brief.html, consolidated 2026-07-28)` — the corrected f_A vs f_mix head-to-head (FINDINGS §18–§20).

Seven panels, all read ONLY from committed artifacts (no sims, seconds to run):
  1. phase6_fig1_correction.png     the artifact vs the fix: raw vs effective fm dose-response
  2. phase6_fig2_headtohead.png     Θ_cum dose-response, both knobs, 3 clean benches + L21b band
  3. phase6_fig3_uniformity.png     band-entry dose vs density — the decision metric (5.39x vs 2.96x)
  4. phase6_fig4_stale.png          Θ_cum split into frozen-no-root vs solved rows (§18's new finding)
  5. phase6_fig5_windcap.png        the 3-Myr wind-only cap on production arms (§20 gap c)
  6. phase6_fig6_slope.png          Phase-5 metric 2's slope half vs the L21b -0.5 expectation
  7. phase6_fig7_mechanism.png      bench3 fm8: why Θ_cum=4.635 is a frozen-solver artifact

Plus the brief's DISPLAY EQUATIONS, rendered to self-contained SVG (phase6_eq*.svg).

    Why SVG and not MathJax: a CDN <script> cannot be verified to render (this container's proxy
    403s cdn.jsdelivr.net) and would leave the brief showing raw TeX source to any offline reader.
    matplotlib's mathtext is already this workstream's LaTeX-free renderer (_trinity_style.py), it
    is offline and deterministic, and `svg.fonttype='path'` embeds the glyphs as outlines so the
    result needs no fonts either. The three constructs mathtext lacks (underbrace, texttt, lor) are
    written around rather than worked around; the TeX source of each equation is kept in the HTML
    `alt` attribute so it stays searchable and accessible.

Sources: data/bench6_analysis.csv, data/bench5_analysis.csv, data/bench_stale_segments.csv,
runs/data/bench6_traj/. Regenerate everything:

    python docs/dev/transition/pdv-trigger/data/make_phase6_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PDV = HERE.parent
RDATA = PDV / "runs" / "data"

sys.path.insert(0, str(HERE))
from _trinity_style import use_trinity_style  # noqa: E402
from make_bench5_analysis import _fnum, _read_csv  # noqa: E402

use_trinity_style()

L21B = (0.90, 0.99)
CLEAN = ["bench3_m1e5_r5", "bench2_m1e5_r10", "bench1_m5e4_r20"]
NBAR = {"bench3_m1e5_r5": 5520.0, "bench2_m1e5_r10": 690.0, "bench1_m5e4_r20": 43.1}
LBL = {
    "bench3_m1e5_r5": r"bench3  $\bar{n}_H=5520$",
    "bench2_m1e5_r10": r"bench2  $\bar{n}_H=690$",
    "bench1_m5e4_r20": r"bench1  $\bar{n}_H=43$",
}
CBENCH = {"bench3_m1e5_r5": "#2a9d3f", "bench2_m1e5_r10": "#1f6feb", "bench1_m5e4_r20": "#6a4c93"}
C_FA, C_FM = "#8a1c1c", "#e07a00"


def _b6():
    return _read_csv(HERE / "bench6_analysis.csv")


def _series(rows, bench, knob, col="theta_cum"):
    """(dose, value) for the diag arms of one (bench, knob); fm inherits the dose-1 __none arm."""
    pts = [
        (float(r["dose"]), _fnum(r[col]))
        for r in rows
        if r["bench"] == bench and r["knob"] == knob and r["arm"] == "diag" and r[col]
    ]
    if knob == "fmix":
        pts += [
            (1.0, _fnum(r[col]))
            for r in rows
            if r["bench"] == bench
            and r["knob"] == "fA"
            and r["arm"] == "diag"
            and float(r["dose"]) == 1
            and r[col]
        ]
    return sorted(p for p in pts if p[1] is not None)


def _band(ax):
    ax.axhspan(*L21B, color="0.55", alpha=0.18, zorder=0)


def _logx(ax, ticks):
    """Plain integer labels on a log dose axis (and no leaked 6x10^0 minor labels)."""
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())


def _plot_clipped(ax, xs, ys, top, color, marker, **kw):
    """Plot a series, but stop the line at `top` and flag any off-scale point with an arrow.

    bench3's f_mix=8 point is Theta_cum=4.635 — real, but a frozen-solver artifact (Fig 7). Letting
    the line shoot off the panel hides the in-band structure that the figure is about.
    """
    keep = [(x, y) for x, y in zip(xs, ys) if y <= top]
    ax.plot([x for x, _ in keep], [y for _, y in keep], marker + "-", color=color, **kw)
    for x, y in zip(xs, ys):
        if y > top:
            ax.plot(
                [x],
                [top * 0.985],
                marker,
                color=color,
                clip_on=False,
                ms=kw.get("ms", 5),
                markerfacecolor="white",
            )
            # label BELOW the clipped marker (arrow points up, off the panel) — above it would
            # collide with the axes title on the multi-panel figures.
            ax.annotate(
                f"{y:.2f}",
                xy=(x, top * 0.985),
                xytext=(0, -16),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0),
            )


def fig1_correction():
    """The artifact and the fix, side by side — this is the whole reason §15j was re-opened."""
    rows = _b6()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, col, title in (
        (
            axes[0],
            "theta_cum_raw_superseded",
            r"SUPERSEDED: $\int (L_{\rm cool}+L_{\rm leak})\,dt$" "\n(raw — omits the boost)",
        ),
        (
            axes[1],
            "theta_cum",
            r"CORRECTED: $\int \theta\,L_{\rm mech}\,dt$" "\n(effective — what the run drains)",
        ),
    ):
        _band(ax)
        for b in CLEAN:
            s = _series(rows, b, "fmix", col)
            _plot_clipped(
                ax,
                [d for d, _ in s],
                [v for _, v in s],
                1.15,
                CBENCH[b],
                "o",
                ms=5,
                lw=1.6,
                label=LBL[b] if col == "theta_cum" else None,
            )
        _logx(ax, [1, 2, 3, 4, 8])
        ax.set_xlabel(r"$f_{\rm mix}$ dose")
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"$\Theta_{\rm cum}$")
    axes[0].set_ylim(0, 1.15)
    axes[0].annotate(
        "falls with dose\n(the artifact)",
        xy=(4, 0.22),
        xytext=(1.7, 0.62),
        fontsize=9,
        color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )
    axes[1].annotate(
        "rises monotonically",
        xy=(3, 0.77),
        xytext=(1.1, 0.99),
        fontsize=9,
        color="#2a9d3f",
        arrowprops=dict(arrowstyle="->", color="#2a9d3f", lw=1.2),
    )
    axes[1].legend(loc="lower right", fontsize=8.5)
    axes[0].text(1.05, 0.915, "L21b band", fontsize=8, color="0.35")
    fig.suptitle(
        r"Fig 1 — the $f_{\rm mix}$ $\Theta_{\rm cum}$ metric artifact (FINDINGS §17) and its fix (§18)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "phase6_fig1_correction.png")


def fig2_headtohead():
    """Both knobs, same axes, same band — the corrected head-to-head."""
    rows = _b6()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharey=True)
    for ax, b in zip(axes, CLEAN):
        _band(ax)
        for knob, c, m in (("fA", C_FA, "o"), ("fmix", C_FM, "s")):
            s = _series(rows, b, knob)
            _plot_clipped(
                ax,
                [d for d, _ in s],
                [v for _, v in s],
                1.25,
                c,
                m,
                ms=4.5,
                lw=1.5,
                label=r"$f_A$" if knob == "fA" else r"$f_{\rm mix}$",
            )
        ax.set_xscale("log")
        ax.set_xlabel("dose")
        ax.set_title(LBL[b], fontsize=10.5)
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 1.25)
    axes[0].set_ylabel(r"$\Theta_{\rm cum}$")
    axes[0].legend(loc="upper left", fontsize=9.5)
    axes[2].text(1.3, 0.945, "L21b band [0.90, 0.99]", fontsize=8, color="0.35")
    fig.suptitle(
        r"Fig 2 — corrected dose–response: $f_{\rm mix}$ needs a far SMALLER dose than $f_A$ "
        "to approach the band",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "phase6_fig2_headtohead.png")


def fig3_uniformity():
    """The tree's own decision metric: how much the calibrated dose moves across density."""
    fa = {"bench3_m1e5_r5": 13.9, "bench2_m1e5_r10": 53.5, "bench1_m5e4_r20": 74.8}
    fm = {"bench3_m1e5_r5": 4.0, "bench2_m1e5_r10": 8.16, "bench1_m5e4_r20": 11.9}
    fm_measured = {"bench3_m1e5_r5"}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    n = [NBAR[b] for b in CLEAN]
    ax.plot(
        n,
        [fa[b] for b in CLEAN],
        "o-",
        color=C_FA,
        ms=8,
        lw=1.8,
        label=r"$f_A$ — all MEASURED in-grid, spread $5.39\times$",
    )
    ax.plot(
        n,
        [fm[b] for b in CLEAN],
        "s--",
        color=C_FM,
        ms=8,
        lw=1.8,
        markerfacecolor="white",
        label=r"$f_{\rm mix}$ — 1/3 measured, spread $2.96\times$",
    )
    for b in CLEAN:
        if b in fm_measured:
            ax.plot(NBAR[b], fm[b], "s", color=C_FM, ms=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"cloud mean density $\bar{n}_H$  [cm$^{-3}$]")
    ax.set_ylabel("band-entry dose")
    ax.grid(alpha=0.2, which="both")
    ax.legend(loc="upper right", fontsize=9)
    ax.annotate(
        "",
        xy=(33.0, 74.8),
        xytext=(33.0, 13.9),
        arrowprops=dict(arrowstyle="<->", color=C_FA, lw=1.4),
    )
    ax.text(34.5, 30, r"$5.39\times$", color=C_FA, fontsize=10)
    ax.annotate(
        "",
        xy=(33.0, 11.9),
        xytext=(33.0, 4.0),
        arrowprops=dict(arrowstyle="<->", color=C_FM, lw=1.4),
    )
    ax.text(34.5, 6.5, r"$2.96\times$", color=C_FM, fontsize=10)
    ax.set_xlim(28, 1.1e4)
    ax.text(
        0.015,
        0.04,
        r"open squares = extrapolated past the $f_{\rm mix}\leq 8$ grid (ESTIMATE)",
        transform=ax.transAxes,
        fontsize=8.5,
        color="0.35",
    )
    ax.set_title(
        "Fig 3 — dose-uniformity across density: smaller spread = better single constant",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "phase6_fig3_uniformity.png")


def fig4_stale():
    """§18's new finding: how much of Θ_cum is frozen no-root rows — worse for f_A."""
    rows = _read_csv(HERE / "bench_stale_segments.csv")
    want = [
        ("bench3_m1e5_r5", "fA", 16.0, r"bench3  $f_A$=16"),
        ("bench2_m1e5_r10", "fA", 64.0, r"bench2  $f_A$=64"),
        ("bench1_m5e4_r20", "fA", 64.0, r"bench1  $f_A$=64"),
        ("bench3_m1e5_r5", "fmix", 4.0, r"bench3  $f_{\rm mix}$=4"),
        ("bench2_m1e5_r10", "fmix", 8.0, r"bench2  $f_{\rm mix}$=8"),
        ("bench1_m5e4_r20", "fmix", 8.0, r"bench1  $f_{\rm mix}$=8"),
    ]
    idx = {(r["bench"], r["knob"], float(r["dose"]), r["arm"]): r for r in rows}
    labels, solved, stale = [], [], []
    for b, k, d, lab in want:
        r = idx.get((b, k, d, "diag"))
        if not r:
            continue
        labels.append(lab)
        solved.append(_fnum(r["theta_cum_from_solved"]) or 0.0)
        stale.append(_fnum(r["theta_cum_from_stale"]) or 0.0)
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    x = range(len(labels))
    ax.bar(x, solved, 0.62, label="from SOLVED rows", color="#2a9d3f")
    ax.bar(
        x,
        stale,
        0.62,
        bottom=solved,
        label="from FROZEN no-root rows",
        color="#c0392b",
        hatch="//",
        edgecolor="white",
    )
    _band(ax)
    for i, (s, t) in enumerate(zip(solved, stale)):
        ax.text(
            i, s + t + 0.02, f"{100 * t / (s + t):.0f}%", ha="center", fontsize=9, color="#c0392b"
        )
    ax.axvline(2.5, color="0.4", lw=1.1, ls=":")
    ax.text(1.0, 1.10, r"$f_A$ arms", ha="center", fontsize=10, color=C_FA)
    ax.text(4.0, 1.10, r"$f_{\rm mix}$ arms", ha="center", fontsize=10, color=C_FM)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\Theta_{\rm cum}$ contribution")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    ax.set_title(
        "Fig 4 — band-setting arms: the frozen share is LARGER on the $f_A$ side", fontsize=11
    )
    fig.tight_layout()
    _save(fig, "phase6_fig4_stale.png")


def fig5_windcap():
    """§20 gap (c): the spec's wind-only window vs what was actually integrated."""
    rows = [
        r
        for r in _read_csv(HERE / "bench5_analysis.csv")
        if r["theta_cum_prefire"]
        and r["theta_cum_wind_only"]
        and r["theta_cum_prefire"] != r["theta_cum_wind_only"]
    ]
    rows.sort(key=lambda r: _fnum(r["theta_cum_prefire"]) or 0)
    full = [_fnum(r["theta_cum_prefire"]) for r in rows]
    wind = [_fnum(r["theta_cum_wind_only"]) for r in rows]
    labels = [
        r["run_name"]
        .replace("_m5e4_r20", "")
        .replace("_m1e5_r10", "")
        .replace("_m1e5_r5", "")
        .replace("_m1e5_r2p5", "")
        .replace("__", " ")
        for r in rows
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    y = range(len(rows))
    for i, (f, w) in enumerate(zip(full, wind)):
        ax.plot([f, w], [i, i], "-", color="0.6", lw=1.2, zorder=1)
    ax.scatter(
        full,
        list(y),
        s=34,
        color="#1f6feb",
        label="integrated to stop_t = 5 Myr (as published)",
        zorder=2,
    )
    ax.scatter(
        wind,
        list(y),
        s=34,
        color="#e07a00",
        marker="D",
        label=r"wind-only window $t \leq 3$ Myr (the spec)",
        zorder=2,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$\Theta_{\rm cum}$")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.2, axis="x")
    ax.set_title(
        "Fig 5 — the 3-Myr cap moves 17/60 arms by 4.3–33.1% (all never-fired production arms)",
        fontsize=10.5,
    )
    fig.tight_layout()
    _save(fig, "phase6_fig5_windcap.png")


def fig6_slope():
    """Phase-5 metric 2's self-contained half — passes the band, misses the L21b power law."""
    rows = [
        r
        for r in _read_csv(HERE / "bench5_analysis.csv")
        if r["arm"] == "diag" and r["slope_1mtheta"]
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhspan(-1.0, 0.0, color="#2a9d3f", alpha=0.10, zorder=0)
    ax.axhline(-0.5, color="k", ls="--", lw=1.3, zorder=1)
    ax.text(17, -0.47, r"L21b: $1-\Theta \propto t^{-1/2}$", fontsize=9)
    for b in CLEAN:
        s = sorted((int(r["f_A"]), _fnum(r["slope_1mtheta"])) for r in rows if r["bench"] == b)
        ax.plot(
            [d for d, _ in s], [v for _, v in s], "o-", color=CBENCH[b], ms=5, lw=1.6, label=LBL[b]
        )
    _logx(ax, [1, 4, 6, 8, 12, 16])
    ax.set_xlabel(r"$f_A$ dose")
    ax.set_ylabel(r"fitted $d\log_{10}(1-\theta)\,/\,d\log_{10} t$")
    ax.set_ylim(-1.1, 0.08)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)
    ax.text(1.05, -0.93, "pass band [-1, 0]", fontsize=8.5, color="#2a7d3f")
    ax.set_title("Fig 6 — metric 2 PASSES its band but decays 3–8x too slowly vs L21b", fontsize=11)
    fig.tight_layout()
    _save(fig, "phase6_fig6_slope.png")


def fig7_mechanism():
    """Why bench3 fm8's Θ_cum = 4.635 is a solver artifact: θ held frozen across no-root spans."""
    rows = _read_csv(RDATA / "bench6_traj" / "bench3_m1e5_r5__fm8_diag.csv")
    t = [_fnum(r["t_now"]) for r in rows]
    th = [_fnum(r["theta"]) for r in rows]
    stale = [i for i in range(1, len(rows)) if rows[i]["Lcool"] == rows[i - 1]["Lcool"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.plot(t, th, "-", color="0.35", lw=1.3, zorder=2)
    for i in stale:
        ax.axvspan(t[i - 1], t[i], color="#c0392b", alpha=0.16, lw=0, zorder=0)
    ax.plot(
        [t[i] for i in stale],
        [th[i] for i in stale],
        "o",
        color="#c0392b",
        ms=3.4,
        zorder=3,
        label=f"frozen no-root rows ({len(stale)}/{len(rows)} = 71%)",
    )
    solved = [i for i in range(len(rows)) if i not in stale]
    ax.plot(
        [t[i] for i in solved],
        [th[i] for i in solved],
        "o",
        color="#2a9d3f",
        ms=3.4,
        zorder=3,
        label="solved rows",
    )
    ax.axhline(1.0, color="k", ls=":", lw=1.1)
    ax.set_yscale("log")
    ax.text(0.0036, 1.12, r"$\theta = 1$ (loss = mechanical input)", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t$  [Myr]")
    ax.set_ylabel(r"$\theta = L_{\rm loss}/L_{\rm mech}$")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_title(
        r"Fig 7 — bench3 $f_{\rm mix}$=8: $\Theta_{\rm cum}=4.635$ is a frozen-solver artifact",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "phase6_fig7_mechanism.png")


# The brief's display equations. Keys become phase6_eq<key>.svg; values are mathtext (the HTML
# carries the same TeX in each <img alt=...>). Rendered at 15pt to sit with 15px Georgia body text.
EQUATIONS = {
    "cool_split": r"$L_{\rm cool} \; = \; L_1 \; + \; L_2 \; + \; L_3$",
    "theta_def": r"$\theta \; = \; \frac{L_{\rm loss}}{L_{\rm mech}}\,, \qquad "
    r"\mathrm{fire\ when}\ \ \frac{L_{\rm gain}-L_{\rm loss}}{L_{\rm gain}} < 0.05 "
    r"\ \ \Leftrightarrow \ \ \theta \gtrsim 0.95$",
    "layout": r"$R_1 \;\; [\,L_1\!:\ \mathrm{hot,\ CIE}\,] \;\rightarrow\; "
    r"[\,L_2\!:\ \mathrm{conduction\ zone}\,] \;\rightarrow\; "
    r"[\,L_3\!:\ \mathrm{sliver\ to}\ 10^4\,] \;\rightarrow\; "
    r"\mathbf{R_2}\ \mathrm{(CD \leftrightarrow shell)}$",
    "fa": r"$\mathrm{in\ the\ ODE:}\quad \frac{du}{dt} \;\longrightarrow\; "
    r"f_A \cdot \left(\frac{du}{dt}\right)_{\rm rad} \quad \mathrm{only\ where}\ "
    r"T < 10^{5.5}\,\mathrm{K}$"
    "\n"
    r"$L_{\rm cool} = L_1 + f_A\,(L_2+L_3)\,, \qquad "
    r"L_{\rm loss} = L_1 + f_A\,(L_2+L_3) + L_{\rm leak}$",
    "fmix": r"$\mathrm{structure\ solved\ FIRST,\ unboosted} \;\rightarrow\; L_{\rm cool}\,, "
    r"\qquad \mathrm{then} \qquad L_{\rm eff} \; = \; L_{\rm leak} + f_{\rm mix} \cdot L_{\rm cool}$",
    "energy": r"$\frac{dE_b}{dt} \; = \; L_{\rm mech} - L_{\rm loss} - P\,\frac{dV}{dt}\,, "
    r"\qquad \Theta_{\rm cum} \; \equiv \; "
    r"\frac{\int_W L_{\rm loss}\,dt}{\int_W L_{\rm mech}\,dt}$",
    "numerators": r"$\Theta^{\rm raw}_{\rm cum} = "
    r"\frac{\int (L_{\rm cool}+L_{\rm leak})\,dt}{\int L_{\rm mech}\,dt}"
    r"\ \ \mathrm{(superseded:\ drops\ the}\ f_{\rm mix}\ \mathrm{boost)}$"
    "\n"
    r"$\Theta_{\rm cum} = "
    r"\frac{\int \theta\,L_{\rm mech}\,dt}{\int L_{\rm mech}\,dt}"
    r"\ \ \mathrm{(corrected:\ the\ effective\ loss)}$",
    "extrap": r"$p = \frac{\ln(\Theta_1/\Theta_0)}{\ln(d_1/d_0)}\,, \qquad "
    r"d_{\rm band} = d_1 \left(\frac{0.90}{\Theta_1}\right)^{1/p}$",
    "stale_split": r"$\Theta_{\rm cum} \; = \; "
    r"\frac{\int_{\rm solved}\theta\,L_{\rm mech}\,dt}{\int L_{\rm mech}\,dt}"
    r"\ \ \mathrm{(physics)} \quad + \quad "
    r"\frac{\int_{\rm frozen}\theta_{\rm held}\,L_{\rm mech}\,dt}{\int L_{\rm mech}\,dt}"
    r"\ \ \mathrm{(solver\ state)}$",
    "slope": r"$1-\Theta \propto t^{-1/2} \;\; \Longrightarrow \;\; "
    r"\frac{d\log_{10}(1-\theta)}{d\log_{10} t} = -0.5\,, \qquad "
    r"\mathrm{pass\ band}\ [-1,\,0]$",
    "doubleboost": r"$L^{\rm fallback}_{\rm loss} \; = \; f_{\rm mix} \cdot "
    r"\left(f_{\rm mix}\,L_{\rm cool}\right) \; = \; f_{\rm mix}^{2}\,L_{\rm cool}"
    r"\qquad \mathrm{(already\ boosted\ once)}$",
    "fired": r"$\mathtt{fired} \; = \; \mathtt{meta\_fired} \ \ \mathrm{OR} \ \ "
    r"\left(\mathtt{reached\_momentum} \ \ \mathrm{AND} \ \ \theta_{\max} \geq 0.95\right)$",
    "fkappa": r"$\kappa_{\rm eff} = f_\kappa \cdot C_{\rm th}\,T^{5/2}\quad \mathrm{(3\ sites{:}\ \dot{M}\ seed,\ Eq44\ ICs,\ ODE\ RHS)}\,, \qquad \dot{M}_{\rm seed} \propto f_\kappa^{\,2/7}\ \uparrow$",
    "elbadry": r"$\theta_{\rm EB}(n) = \frac{A_{\rm mix}\sqrt{\lambda\delta v\; n}}{\frac{11}{5} + A_{\rm mix}\sqrt{\lambda\delta v\; n}}\,, \qquad A_{\rm mix}{=}3.5\,,\ \lambda\delta v{=}3\ \mathrm{pc\,km/s}\,,\ n_{\rm fire}(\theta{=}0.95) \approx 48\ \mathrm{cm^{-3}}$",
    "eq47": r"$\dot{m} = \dot{m}_0\,\frac{(1-\theta)^{37/35}}{\theta^{2/7}}\qquad \mathrm{(El\!-\!Badry\ Eq\ 47{:}\ evaporation\ FALLS\ as\ cooling\ rises)}$",
    "thetatarget": r"$L_{\rm loss} = \max\left(L_{\rm cool}+L_{\rm leak},\ \theta_t\,L_{\rm mech}\right) \qquad \mathrm{(single{-}count\ top{-}up{:}\ inert\ where\ resolved\ loss\ exceeds\ target)}$",
    "fk_site1": r"$\mathrm{site\ 1\ (:304,\ Weaver\ Eq\ 33)\!:}\quad \dot{M}_{\rm seed} = \frac{12}{75}\,\xi^{5/2}\,\frac{4\pi R_2^{3}}{t}\,\frac{\mu_{\rm ion}}{k_B}\left(\frac{t\,f_\kappa C_{\rm th}}{R_2^{2}}\right)^{2/7} P_b^{5/7}\ \ \Rightarrow\ \ \dot{M}\propto f_\kappa^{2/7}$",
    "fk_site2": r"$\mathrm{site\ 2\ (:398,\ Weaver\ Eq\ 44)\!:}\quad dR_2 = \frac{T_{\rm init}^{5/2}}{\mathcal{C}\,\dot{M}/(4\pi R_2^{2})}\,, \qquad \mathcal{C} = \frac{25}{4}\frac{k_B}{\mu_{\rm ion}\,f_\kappa C_{\rm th}}\ \ \Rightarrow\ \ dR_2 \propto f_\kappa$",
    "fk_site3": r"$\mathrm{site\ 3\ (:441,\ Weaver\ Eqs\ 42\!-\!43)\!:}\quad \frac{d^{2}T}{dr^{2}} = \frac{P_b}{f_\kappa C_{\rm th}T^{5/2}}\left[\frac{\beta + \frac{5}{2}\delta}{t} + \frac{5}{2}(v-v_t)\frac{1}{T}\frac{dT}{dr} - \frac{\dot{u}}{P_b}\right] - \frac{5}{2T}\left(\frac{dT}{dr}\right)^{2} - \frac{2}{r}\frac{dT}{dr}$",
    "fa_site1": r"$\mathrm{site\ 1\ (:435\!-\!437,\ in\ the\ ODE\ RHS)\!:}\quad \dot{u} \longrightarrow f_A\,\dot{u} \quad \mathrm{if}\ T < 10^{5.5}\,\mathrm{K}\qquad \mathrm{(enters}\ d^{2}T/dr^{2}\ \mathrm{above)}$",
    "fa_site2": r"$\mathrm{site\ 2\ (:845\!-\!848,\ on\ the\ integrals)\!:}\quad L_2 \longrightarrow f_A L_2\,, \quad L_3 \longrightarrow f_A L_3\,, \quad L_1\ \mathrm{and}\ L_{\rm leak}\ \mathrm{untouched}$",
    "integrand": r"$L_1 = \int \chi_e\,n^{2}\Lambda_{\rm CIE}(T)\,4\pi r^{2}\,dr \quad (\mathrm{:746})\,, \qquad L_{2,3} = \int \dot{u}_{\rm net}(n,T,\phi)\,4\pi r^{2}\,dr \quad (\mathrm{:793,\ :835})$",
    "ndens": r"$n(r) = \frac{P_b}{(\mu_{\rm conv}/\mu_{\rm ion})\,k_B\,T(r)} \quad (\mathrm{:673}) \qquad \Rightarrow \qquad n \propto 1/T \ \ \mathrm{at\ near\!-\!uniform}\ P_b$",
    "sc0": r"$\mathrm{C1\ El\!-\!Badry:}\quad f_A = \theta_{\rm EB}(\lambda\delta v,\, n_{\rm amb})"
    r"\cdot \frac{L_{\rm mech}}{L_2+L_3}$"
    "\n"
    r"$\mathrm{C2\ Lancaster\ Eq\ 11:}\quad f_A = \alpha_A "
    r"\left(\frac{R_2}{\ell_{\rm cool}}\right)^{d}, \quad "
    r"\ell_{\rm cool} = \frac{(v_t\,t_{\rm cool})^2}{L}$"
    "\n"
    r"$\mathrm{C3\ fitted\ baseline:}\quad f_A = 315\,\bar{n}^{-0.335}$",
}


def render_equations():
    """Each display equation -> a self-contained SVG (glyphs as paths, transparent background)."""
    matplotlib.rcParams["svg.fonttype"] = "path"
    for key, tex in EQUATIONS.items():
        # ONE text call: matplotlib parses each newline-separated segment as its own mathtext run
        # and lays them out with `linespacing`. Separate fig.text calls at hand-computed offsets do
        # not survive bbox_inches="tight" on a zero-size figure — they collapse onto one line.
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(
            0.0, 0.0, tex, fontsize=15, color="#1a1a1a", ha="left", va="baseline", linespacing=3.4
        )
        out = PDV / f"phase6_eq_{key}.svg"
        fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.06, transparent=True)
        plt.close(fig)
        print(f"wrote {out}")


def _save(fig, name):
    out = PDV / name
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render_equations()
    fig1_correction()
    fig2_headtohead()
    fig3_uniformity()
    fig4_stale()
    fig5_windcap()
    fig6_slope()
    fig7_mechanism()
