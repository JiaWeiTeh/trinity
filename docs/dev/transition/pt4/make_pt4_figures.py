#!/usr/bin/env python3
"""pt4 findings figures (H1, H2 [, H3 when its CSVs land]).

Pure reads of the committed CSVs in this folder and ../cleanroom/data/ — no sims
re-run. Saves PDF+PNG into docs/dev/transition/pt4/figures/.

    python docs/dev/transition/pt4/make_pt4_figures.py

H1 (Lcool is correct; surge-then-collapse; the divergence is the beta-clamp):
  h1_lloss_surge_collapse  - Lloss(t), all 6 hybr configs, log-log, peak marked.
  h1_beta_clamp_divergence - ratio(t) + cool_beta(t), hybr vs legacy (simple_cluster).
  h1_ratio_min_stats       - bar chart of ratio_min per config, hybr vs legacy vs 0.05.
H2 (rCloud is not a fail; cooling is set by LOCAL density, not rCloud proximity):
  h2_ratio_vs_rcloud       - cooling ratio vs R2/rCloud, all configs, edge crossing marked.
  h2_matched_r2            - the decisive experiment: ratio vs absolute R2, baseline vs
                             bigcloud overlay (identical in-cloud despite 5.2x rCloud).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

HERE = Path(__file__).resolve().parent
CLEAN = HERE.parent / "cleanroom" / "data"
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False

WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
THRESH = 0.05
OUT = HERE / "figures"

# authoritative rCloud [pc] per config (h2_rcloud_edge.csv / H2_rcloud_audit.md §1)
RCLOUD = {
    "simple_cluster": 1.690, "pl2_steep": 21.355, "small_dense_highsfe": 0.3255,
    "midrange_pl0": 8.530, "large_diffuse_lowsfe": 88.053, "be_sphere": 15.501,
}
CONFIGS = ["simple_cluster", "small_dense_highsfe", "midrange_pl0",
           "pl2_steep", "be_sphere", "large_diffuse_lowsfe"]


def _f(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def load(path, cols, drop_momentum=True):
    """Return dict of column-lists over rows with finite t_now and the requested cols."""
    out = {c: [] for c in ["t_now", *cols]}
    if not Path(path).exists():
        return None
    for row in csv.DictReader(open(path)):
        if drop_momentum and row.get("phase") == "momentum":
            continue
        tn = _f(row, "t_now")
        vals = {c: _f(row, c) for c in cols}
        if tn > 0 and all(v == v for v in vals.values()):  # all finite
            out["t_now"].append(tn)
            for c in cols:
                out[c].append(vals[c])
    return out if out["t_now"] else None


def ratio_of(d):
    return [(g - l) / g for g, l in zip(d["bubble_Lgain"], d["bubble_Lloss"]) if g > 0]


def _ratio_xy(d, xkey):
    x, r = [], []
    for xx, g, l in zip(d[xkey], d["bubble_Lgain"], d["bubble_Lloss"]):
        if g > 0:
            x.append(xx); r.append((g - l) / g)
    return x, r


# ---------------------------------------------------------------- H1
def fig_h1_lloss():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, name in enumerate(CONFIGS):
        d = load(CLEAN / f"c0_{name}_h0.csv", ["bubble_Lloss"])
        if not d:
            continue
        c = WONG[i % len(WONG)]
        ax.plot(d["t_now"], d["bubble_Lloss"], color=c, lw=1.3, label=name)
        j = max(range(len(d["bubble_Lloss"])), key=lambda k: d["bubble_Lloss"][k])
        ax.scatter([d["t_now"][j]], [d["bubble_Lloss"][j]], color=c, s=28,
                   zorder=5, edgecolor="0.2", linewidth=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("time  [Myr]")
    ax.set_ylabel(r"$L_{\rm cool}=L_{\rm loss}$  [code units]")
    ax.set_title("H1: $L_{\\rm cool}$ surges ~2x (dot = peak) then collapses 4-9x "
                 "— emission-measure turnover, not a bug", fontsize=9.5)
    ax.legend(fontsize=7, ncol=2, loc="lower center", framealpha=0.9)
    _save(fig, "h1_lloss_surge_collapse")


def fig_h1_beta_clamp(name="simple_cluster"):
    h = load(CLEAN / f"c0_{name}_h0.csv", ["bubble_Lgain", "bubble_Lloss", "cool_beta"])
    g = load(CLEAN / f"c0_{name}_legacy.csv", ["bubble_Lgain", "bubble_Lloss", "cool_beta"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1]})
    for d, lab, col in ((g, "legacy (beta clamped [0,1])", "#0072B2"),
                        (h, "hybr (beta free)", "#D55E00")):
        if not d:
            continue
        x, r = _ratio_xy(d, "t_now")
        a1.plot(x, r, color=col, lw=1.5, label=lab)
        a2.plot(d["t_now"], d["cool_beta"], color=col, lw=1.5, label=lab)
    a1.axhline(THRESH, ls="--", lw=1.2, color="0.5")
    a1.text(0.99, THRESH + 0.02, "transition threshold 0.05",
            transform=a1.get_yaxis_transform(), ha="right", fontsize=8, color="0.45")
    a1.set_ylabel(r"cooling ratio $(L_{\rm gain}-L_{\rm loss})/L_{\rm gain}$")
    a1.set_ylim(-0.1, 1.0); a1.legend(fontsize=8, loc="center right")
    a1.set_title(f"H1: legacy crosses 0.05 only because beta is clamped — "
                 f"hybr's free beta -> +4 under-cools  [{name}]", fontsize=9.5)
    a2.axhspan(0, 1, color="0.85", alpha=0.6, zorder=0)
    a2.text(0.012, 0.9, "legacy clamp band [0,1]", fontsize=7.5, color="0.4")
    a2.axhline(0, ls=":", lw=0.8, color="0.6")
    a2.set_xscale("log"); a2.set_ylabel(r"$\beta = -t\,\partial_t\ln P_b$ (cool_beta)")
    a2.set_xlabel("time  [Myr]")
    _save(fig, "h1_beta_clamp_divergence")


def fig_h1_stats():
    rows = list(csv.DictReader(open(HERE / "H1_lcool_direction_summary.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[r["tag"]] = r
    names = [n for n in CONFIGS if n in by]
    import numpy as np
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    hy = [float(by[n]["h0"]["ratio_min"]) for n in names]
    lg = [float(by[n]["legacy"]["ratio_min"]) if "legacy" in by[n] else float("nan")
          for n in names]
    ax.bar(x - w / 2, hy, w, color="#D55E00", label="hybr (production)")
    ax.bar(x + w / 2, lg, w, color="#0072B2", label="legacy (clamped)")
    ax.axhline(THRESH, ls="--", lw=1.4, color="0.3")
    ax.text(len(names) - 0.5, THRESH + 0.02, "0.05 trigger", ha="right",
            fontsize=8, color="0.3")
    ax.axhline(0, lw=0.8, color="0.6")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cooling-ratio minimum  (fires if < 0.05)")
    ax.set_title("H1 statistics: hybr floors 0.28-0.49 (0/6 fire); "
                 "legacy dips <=0 (5/6 fire) — purely the beta-clamp", fontsize=9.5)
    ax.legend(fontsize=8)
    _save(fig, "h1_ratio_min_stats")


# ---------------------------------------------------------------- H2
def fig_h2_vs_rcloud():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, name in enumerate(CONFIGS):
        d = load(CLEAN / f"c0_{name}_h0.csv", ["bubble_Lgain", "bubble_Lloss", "R2"])
        if not d or name not in RCLOUD:
            continue
        c = WONG[i % len(WONG)]
        xr = [r2 / RCLOUD[name] for r2 in d["R2"]]
        x, r = [], []
        for rr, g, l in zip(xr, d["bubble_Lgain"], d["bubble_Lloss"]):
            if g > 0:
                x.append(rr); r.append((g - l) / g)
        ax.plot(x, r, color=c, lw=1.3, label=name)
    ax.axvline(1.0, ls="-", lw=1.4, color="0.3")
    ax.text(1.04, 0.92, "R2 = rCloud\n(blowout: clean phase switch)", fontsize=8, color="0.3")
    ax.axhline(THRESH, ls="--", lw=1.2, color="#D55E00")
    ax.text(2e-2, THRESH + 0.02, "0.05 trigger", fontsize=8, color="#D55E00")
    ax.set_xscale("log"); ax.set_xlabel(r"$R_2 / r_{\rm Cloud}$")
    ax.set_ylabel(r"cooling ratio $(L_{\rm gain}-L_{\rm loss})/L_{\rm gain}$")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("H2: ratio bottoms at the cloud edge (~0.3-0.5) then RECOVERS past it "
                 "— blowout never drives it to 0.05", fontsize=9.3)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)
    _save(fig, "h2_ratio_vs_rcloud")


def fig_h2_matched():
    b = load(HERE / "h2_sc_baseline.csv", ["bubble_Lgain", "bubble_Lloss", "R2"])
    g = load(HERE / "h2_sc_bigcloud.csv", ["bubble_Lgain", "bubble_Lloss", "R2"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # baseline drawn as a THICK underlay so the perfect overlap reads as two curves
    for d, lab, col, lw, ms in ((b, "baseline (rCloud=1.69 pc)", "#0072B2", 5.0, 0),
                                (g, "bigcloud (rCloud=8.83 pc, 5.2x)", "#D55E00", 1.6, 2.6)):
        if not d:
            continue
        x, r = _ratio_xy(d, "R2")
        ax.plot(x, r, color=col, lw=lw, marker="o", ms=ms, label=lab,
                alpha=0.55 if ms == 0 else 1.0)
    ax.axvline(0.894, ls=":", lw=1.2, color="0.4")
    ax.annotate("matched R2 = 0.894 pc\nratio = 0.4845 in BOTH\n"
                "(R2/rCloud = 0.53 vs 0.10)", xy=(0.894, 0.4845),
                xytext=(0.93, 0.30), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
    ax.axhline(THRESH, ls="--", lw=1.2, color="0.5")
    ax.set_xlabel(r"shell radius $R_2$  [pc]  (absolute, in-cloud)")
    ax.set_ylabel(r"cooling ratio $(L_{\rm gain}-L_{\rm loss})/L_{\rm gain}$")
    ax.set_title("H2 decisive test: at matched absolute R2 the cooling state is IDENTICAL "
                 "— set by local density, not rCloud", fontsize=9.0)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "h2_matched_r2")


def fig_h2_dipgradient():
    """Why pl2_steep bottoms deep inside while flat configs bottom at the edge:
    the cooling-ratio minimum sits at the first steep density gradient the shell
    meets — the cloud edge for flat (alpha=0) clouds, but rCore for the alpha=-2
    profile. Top: actual production ambient density profile n(r) with the dip
    location marked. Bottom: the clean correlation (dip location vs in-cloud decline)."""
    import numpy as np
    try:
        from trinity._input.read_param import read_param
        from trinity.phase0_init.get_InitCloudProp import get_InitCloudProp
        from trinity.cloud_properties.density_profile import get_density_profile
        import trinity._functions.unit_conversions as cvt
    except Exception as e:  # pragma: no cover
        print(f"skip h2_dip_vs_density_gradient (trinity import failed: {e})")
        return
    cfg = HERE.parent / "cleanroom" / "configs"
    cross = {r["config"]: r for r in csv.DictReader(open(HERE / "h2_crossing_summary.csv"))}
    edge = {r["config"]: r for r in csv.DictReader(open(HERE / "h2_rcloud_edge.csv"))}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 7.6))
    for i, name in enumerate(CONFIGS):
        c = WONG[i % len(WONG)]
        params = read_param(str(cfg / f"{name}.param"))
        props = get_InitCloudProp(params)
        rC = props.rCloud
        xr = np.logspace(-2, np.log10(1.9), 260)        # r / rCloud
        n = np.array([get_density_profile(x * rC, params) * cvt.ndens_au2cgs for x in xr])
        a1.plot(xr, n, color=c, lw=1.3, label=name)
        loc = float(cross[name]["R2overRc_at_ratio_min"])
        n_at = get_density_profile(min(loc, 1.85) * rC, params) * cvt.ndens_au2cgs
        a1.scatter([loc], [n_at], color=c, s=55, zorder=6, edgecolor="0.15", linewidth=0.7)
    a1.axvline(1.0, ls=":", lw=1.2, color="0.4")
    a1.text(1.03, a1.get_ylim()[1] * 0.3 if False else 2e5, "cloud edge", rotation=90,
            fontsize=7.5, color="0.4", va="top")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel(r"$r / r_{\rm Cloud}$")
    a1.set_ylabel(r"ambient density $n(r)$  [cm$^{-3}$]")
    a1.set_title("Dot = where the cooling ratio bottoms. It sits at the density 'knee': "
                 "the EDGE for flat clouds, rCore for pl2_steep", fontsize=8.8)
    a1.legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.9)

    declines, locs, labels, cols = [], [], [], []
    for i, name in enumerate(CONFIGS):
        d = float(edge[name]["nCore_cgs"]) / float(edge[name]["nEdge_cgs"])
        declines.append(d)
        locs.append(float(cross[name]["R2overRc_at_ratio_min"]))
        labels.append(name); cols.append(WONG[i % len(WONG)])
    a2.scatter(declines, locs, c=cols, s=80, edgecolor="0.15", linewidth=0.7, zorder=5)
    # the 4 flat (alpha=0) configs all sit at decline=1.0 -> annotate as a group,
    # label the two gradient configs individually
    for d, l, name in zip(declines, locs, labels):
        if name in ("be_sphere", "pl2_steep"):
            a2.annotate(name, (d, l), textcoords="offset points", xytext=(9, -2), fontsize=8)
    a2.annotate("4 flat ($\\alpha$=0) configs\n(simple_cluster, small_dense,\n"
                "midrange, large_diffuse)", (1.0, 1.13), textcoords="offset points",
                xytext=(14, 6), fontsize=7.5, va="center",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    a2.axhline(1.0, ls=":", lw=1.2, color="0.4")
    a2.text(60, 1.04, "ratio bottoms AT the cloud edge", fontsize=7.5, color="0.4", ha="right")
    a2.set_xscale("log")
    a2.set_xlabel(r"in-cloud density decline  $n_{\rm Core}/n_{\rm Edge}$  (profile steepness)")
    a2.set_ylabel(r"$R_2/r_{\rm Cloud}$ at the cooling-ratio minimum")
    a2.set_title("Correlation: the steeper the in-cloud density, the deeper inside the dip sits",
                 fontsize=8.8)
    a2.set_ylim(-0.05, 1.35)
    _save(fig, "h2_dip_vs_density_gradient")


def fig_h3_ebfloor():
    """H3: flooring Eb>0 is a clean no-op where Eb never collapses, and does NOT
    rescue the 5e9 collapse configs (the solver just grinds). Left: fail_repro Eb(t)
    V0 (collapses through 0 -> ENERGY_COLLAPSED) vs EBFLOOR (pinned >0 but R2 stuck,
    grind). Right: simple_cluster no-op (V0 and EBFLOOR exactly overlaid)."""
    T = HERE / "traj"
    fr0 = load(T / "h3_traj_fail_repro_V0.csv", ["Eb", "R2"], drop_momentum=False)
    frf = load(T / "h3_traj_fail_repro_EBFLOOR.csv", ["Eb", "R2"], drop_momentum=False)
    sc0 = load(T / "h3_traj_simple_cluster_V0.csv", ["Eb"], drop_momentum=False)
    scf = load(T / "h3_traj_simple_cluster_EBFLOOR.csv", ["Eb"], drop_momentum=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    if fr0:
        a1.plot(fr0["t_now"], fr0["Eb"], color="#0072B2", lw=1.8, label="V0 baseline")
    if frf:
        a1.plot(frf["t_now"], frf["Eb"], color="#D55E00", lw=1.8, ls="--",
                label="EBFLOOR (solver grinds at collapse)")
    a1.axhline(0, lw=0.9, color="0.5")
    a1.text(a1.get_xlim()[0], 0, " Eb=0 -> ENERGY_COLLAPSED", fontsize=7.5, color="0.4", va="bottom")
    a1.set_xlabel("time  [Myr]"); a1.set_ylabel(r"$E_b$  [code units]")
    a1.set_title("fail_repro (5e9): flooring Eb avoids the collapse stop but the\n"
                 "implicit solver then GRINDS (R2 stuck) — Eb-floor does not rescue", fontsize=8.8)
    a1.legend(fontsize=8, loc="upper right")
    if sc0:
        a2.plot(sc0["t_now"], sc0["Eb"], color="#0072B2", lw=4.2, alpha=0.55, label="V0 baseline")
    if scf:
        a2.plot(scf["t_now"], scf["Eb"], color="#D55E00", lw=1.5, label="EBFLOOR")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("time  [Myr]"); a2.set_ylabel(r"$E_b$  [code units]")
    a2.set_title("simple_cluster (control): EBFLOOR overlays V0 exactly\n"
                 "max rel|dEb| = 0  -> bit-identical no-op (Eb never collapses)", fontsize=8.8)
    a2.legend(fontsize=8, loc="lower right")
    _save(fig, "h3_ebfloor_noop_and_grind")


def fig_h4_control():
    """H4 control vs collapse (the requested controlled setting): PdV/Lmech(t) for
    simple_cluster (control, stays <1 -> cap never bites, byte-identical) vs the 5e9
    clouds (cross 1 -> Eb collapses). One control parameter sorts the regimes."""
    T = HERE / "traj"
    series = [("simple_cluster", "h4_traj_simple_cluster_V0.csv", "#009E73", "simple_cluster (control, doesn't collapse)"),
              ("fail_repro", "h4_traj_fail_repro_V0.csv", "#D55E00", "fail_repro (5e9, collapses)"),
              ("fail_helix", "h4_traj_fail_helix_V0.csv", "#0072B2", "fail_helix (5e9, collapses)")]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name, fn, col, lab in series:
        d = load(T / fn, ["pdv_over_lmech"], drop_momentum=False)
        if not d:
            continue
        ax.plot(d["t_now"], d["pdv_over_lmech"], color=col, lw=1.7, label=lab)
    ax.axhline(1.0, ls="--", lw=1.4, color="0.3")
    ax.text(0.02, 1.04, "PdV = Lmech (break-even): above -> Eb collapses",
            transform=ax.get_yaxis_transform(), fontsize=8, color="0.3")
    ax.set_xscale("log")
    # fail_repro's ratio spikes hugely negative AT Eb->0 (numerical artifact of the
    # collapse); clip to the physically meaningful band so the crossing of 1 is visible
    ax.set_ylim(-0.1, 3.0)
    ax.set_xlabel("time  [Myr]")
    ax.set_ylabel(r"$\mathrm{PdV}/L_{\rm mech} = 4\pi R_2^2 P_b v_2 / L_{\rm mech}$")
    ax.set_title("H4 control vs collapse: the cap is inert on simple_cluster (PdV/Lmech<1) "
                 "but the 5e9 clouds cross 1", fontsize=8.8)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "h4_control_vs_collapse")


def fig_r1_firing_preview():
    """R1 shadow PREVIEW (from committed data; offline criteria == the in-code shadow,
    proven by the byte-identical gate). Per config: where R1 would hand off
    (blowout R2>rCloud = star; Eb-peak = diamond) vs the CURRENT trigger, which never
    fires (the bubble stays energy-driven the whole bar -> the grey duration)."""
    import numpy as np
    rows = []  # (name, t0, t1, blowout_t, ebpeak_t, color)
    for i, name in enumerate(CONFIGS):  # 6 normal, from c0 data
        d = load(CLEAN / f"c0_{name}_h0.csv", ["R2", "Eb"], drop_momentum=False)
        if not d or name not in RCLOUD:
            continue
        t = d["t_now"]; R2 = d["R2"]; Eb = d["Eb"]; rc = RCLOUD[name]
        bt = next((t[k] for k in range(len(R2)) if R2[k] > rc), None)
        imax = max(range(len(Eb)), key=lambda k: Eb[k])
        et = t[imax] if imax < len(Eb) - 3 else None  # interior peak only
        rows.append((name, t[0], t[-1], bt, et, WONG[i % len(WONG)]))
    for j, name in enumerate(["fail_repro", "fail_helix"]):  # heavy, from h4 traj
        d = load(HERE / "traj" / f"h4_traj_{name}_V0.csv", ["R2", "Eb"], drop_momentum=False)
        if not d:
            continue
        t = d["t_now"]; Eb = d["Eb"]
        imax = max(range(len(Eb)), key=lambda k: Eb[k])
        et = t[imax] if imax < len(Eb) - 1 else t[imax]
        rows.append((name + " (5e9)", t[0], t[-1], None, et, "0.35"))

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for y, (name, t0, t1, bt, et, c) in enumerate(rows):
        ax.plot([t0, t1], [y, y], color=c, lw=5, alpha=0.35, solid_capstyle="round")
        if bt is not None:
            ax.scatter([bt], [y], marker="*", s=220, color=c, edgecolor="0.15",
                       linewidth=0.7, zorder=6)
        if et is not None:
            ax.scatter([et], [y], marker="D", s=70, color=c, edgecolor="0.15",
                       linewidth=0.7, zorder=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("time  [Myr]")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title("R1 shadow: where the transition WOULD hand off\n"
                 "star = blowout (R2>rCloud) · diamond = Eb-peak · grey bar = how long the CURRENT "
                 "trigger keeps it energy-driven (never fires)", fontsize=8.6)
    leg = [mlines.Line2D([], [], marker="*", color="0.3", ls="", ms=13, label="blowout (R1) — normal clouds"),
           mlines.Line2D([], [], marker="D", color="0.3", ls="", ms=7, label="Eb-peak (R1) — heavy clouds"),
           mlines.Line2D([], [], color="0.5", lw=5, alpha=0.4, label="energy-driven duration (current: never transitions)")]
    ax.legend(handles=leg, fontsize=7.5, loc="upper left", framealpha=0.95)
    _save(fig, "r1_firing_preview")


def _save(fig, stem):
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/{stem}.(pdf,png)")


def main():
    fig_h1_lloss()
    fig_h1_beta_clamp()
    fig_h1_stats()
    fig_h2_vs_rcloud()
    fig_h2_matched()
    fig_h2_dipgradient()
    fig_h3_ebfloor()
    fig_h4_control()
    print("pt4 H1-H4 figures done.")


if __name__ == "__main__":
    main()
