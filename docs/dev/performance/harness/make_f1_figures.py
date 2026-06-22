#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the F1 plan figures from COMMITTED data (no sim re-runs).

Reads ../data/{bubble_resample_*.csv, f1edge_{orig,f1}_trajectories.csv} and writes
../figs/*.png. Demonstrates: the idea + per-call result, full-run equivalence, the
matched-t verdict, and the validation workflow.

    python docs/dev/performance/harness/make_f1_figures.py
"""
import csv
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
plt.style.use(str(ROOT / "paper" / "_lib" / "trinity.mplstyle"))
plt.rcParams.update({
    "text.usetex": False,
    "figure.dpi": 130,
    "savefig.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.constrained_layout.use": True,
})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figs")
os.makedirs(FIGS, exist_ok=True)
GATE = 3e-3


# ---------------- Figure 1: per-call speedup + accuracy ----------------
def load_percall():
    res = {}
    for f in sorted(glob.glob(os.path.join(DATA, "bubble_resample_*.csv"))):
        for row in csv.DictReader(open(f)):
            c, ph, v = row["config"], row["phase"], row["variant"]
            try:
                t, r = float(row["time_ms"]), abs(float(row["rel_dMdt"]))
            except ValueError:
                continue
            d = res.setdefault(c, {}).setdefault(ph, {}).setdefault(v, {"t": [], "r": []})
            d["t"].append(t); d["r"].append(r)
    return res


pc = load_percall()
configs = [c for c in ["mock_hybr", "probe_typical_hybr", "steep",
                       "dense_flat", "simple_cluster", "sfe0.6"] if c in pc]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(configs)); w = 0.38
for i, ph in enumerate(["energy", "implicit"]):
    sp = []
    for c in configs:
        d = pc[c].get(ph, {})
        bt = np.mean(d["baseline"]["t"]) if "baseline" in d else np.nan
        mt = np.mean(d["M500"]["t"]) if "M500" in d else np.nan
        sp.append(bt / mt if mt else np.nan)
    bars = ax1.bar(x + (i - 0.5) * w, sp, w, label=ph)
    for b, s in zip(bars, sp):
        ax1.text(b.get_x() + b.get_width() / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=7)
ax1.axhline(1.0, color="k", lw=0.8, ls=":")
ax1.set_xticks(x); ax1.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
ax1.set_ylabel("per-call speedup  (baseline 60k / F1 500-pt)")
ax1.set_title("(a) F1 per-bubble-call speedup"); ax1.legend()
worst = []
for c in configs:
    w_ = 0.0
    for ph in pc[c]:
        for v, d in pc[c][ph].items():
            if v != "baseline":
                w_ = max(w_, max(d["r"]))
    worst.append(w_)
ax2.bar(x, worst, color="tab:green")
ax2.axhline(GATE, color="r", ls="--", label="0.3% gate")
ax2.set_yscale("log"); ax2.set_ylim(1e-8, 1e-2)
ax2.set_xticks(x); ax2.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("worst per-call rel_dMdt"); ax2.set_title("(b) F1 per-call accuracy vs gate"); ax2.legend()
fig.suptitle("F1: 60k dense-output resample -> 500-pt coarse t_eval in the dMdt residual", fontsize=12)
fig.savefig(os.path.join(FIGS, "f1_percall.png")); plt.close(fig)


# ---------------- full-run trajectories ----------------
def load_traj(path):
    d = {}
    for row in csv.DictReader(open(path)):
        c = row["config"]
        try:
            rec = d.setdefault(c, {k: [] for k in ("t", "R2", "Eb", "rShell")})
            rec["t"].append(float(row["t_now"])); rec["R2"].append(float(row["R2"]))
            rec["Eb"].append(float(row["Eb"])); rec["rShell"].append(float(row["rShell"]))
        except (ValueError, TypeError):
            pass
    return d


orig = load_traj(os.path.join(DATA, "f1edge_orig_trajectories.csv"))
f1 = load_traj(os.path.join(DATA, "f1edge_f1_trajectories.csv"))
tcfg = [("f1cmp_simple", "simple_cluster"),
        ("f1edge_lowdens", "low-rho / hi-M / hi-sfe"),
        ("f1edge_hidens", "hi-rho / hi-M / lo-sfe (stiff)")]

# Figure 2: R2(t) & Eb(t) overlay
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for j, (c, label) in enumerate(tcfg):
    o, f = orig[c], f1[c]
    ot = np.array(o["t"]); fo = np.argsort(np.array(f["t"])); ft = np.array(f["t"])[fo]
    tmax = min(ot.max(), ft.max())
    for r, k in enumerate(["R2", "Eb"]):
        ax = axes[r][j]
        ax.plot(ot, o[k], "-", color="tab:blue", lw=2.4, label="original 60k")
        ax.plot(ft, np.array(f[k])[fo], "--", color="tab:orange", lw=1.4, label="F1 coarse")
        ax.axvline(tmax, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("t [Myr]"); ax.set_ylabel(k + (" [pc]" if k == "R2" else ""))
        if r == 0:
            ax.set_title(label, fontsize=9)
        if k == "Eb":
            ax.set_yscale("log")
        if j == 0 and r == 0:
            ax.legend(fontsize=8)
fig.suptitle("F1 full-run equivalence — R2(t) & Eb(t): original-60k vs F1-coarse overlaid "
             "(dotted line = common-t cap; F1 runs further as it is faster)", fontsize=10)
fig.savefig(os.path.join(FIGS, "f1_fullrun_overlay.png")); plt.close(fig)

# Figure 3: matched-t rel-diff (t normalised to common range so all 3 are comparable)
fig, ax = plt.subplots(figsize=(9, 5.5))
for c, label in tcfg:
    o, f = orig[c], f1[c]
    ot = np.array(o["t"]); fo = np.argsort(np.array(f["t"])); ft = np.array(f["t"])[fo]
    tmax = min(ot.max(), ft.max()); mask = ot <= tmax; tg = ot[mask]
    oR = np.array(o["R2"])[mask]; fR = np.interp(tg, ft, np.array(f["R2"])[fo])
    rel = np.abs(fR - oR) / np.maximum(np.abs(oR), 1e-300)
    ax.plot(tg / tmax, np.maximum(rel, 1e-12), lw=1.6, label=f"{label}  (worst {rel.max():.1e})")
ax.axhline(GATE, color="r", ls="--", lw=1.5, label="0.3% gate")
ax.set_yscale("log"); ax.set_ylim(1e-11, 1e-2)
ax.set_xlabel("fraction of common t-range"); ax.set_ylabel("matched-t  |R2_F1 - R2_orig| / R2_orig")
ax.set_title("F1 full-run accuracy: matched-t R2 rel-diff, ~500x inside the 0.3% gate")
ax.legend(fontsize=8, loc="upper left")
fig.savefig(os.path.join(FIGS, "f1_matched_reldiff.png")); plt.close(fig)


# ---------------- Figure 4: validation workflow ----------------
# Manually-positioned schematic: opt out of constrained_layout (bbox_inches crops it).
fig, ax = plt.subplots(figsize=(13, 4), layout="none"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
phases = [("P0", "capture + sweep\n6 configs", "G0 ok"),
          ("P1", "pick N = 500\n(npts-insensitive)", "G1 ok"),
          ("P2", "per-call equiv\nrel_dMdt 3e-6", "G2 ok\n(necessary,\nNOT sufficient)"),
          ("P3", "apply patch\n+ 538 unit tests", "ok"),
          ("P5", "FULL-RUN equiv\nmock + 3 edges\nworst ~6e-6", "DECIDER ok"),
          ("G3", "stress: 0 crash\n+ golden match", "ok -> SHIP")]
n = len(phases); bw = 0.85 / n
for i, (p, desc, gate) in enumerate(phases):
    x0 = 0.02 + i * (1.0 / n)
    ax.add_patch(plt.Rectangle((x0, 0.42), bw, 0.46, fc="#e8f5e9", ec="tab:green", lw=2))
    ax.text(x0 + bw / 2, 0.81, p, ha="center", fontweight="bold", fontsize=12)
    ax.text(x0 + bw / 2, 0.62, desc, ha="center", fontsize=7.3)
    ax.text(x0 + bw / 2, 0.30, gate, ha="center", fontsize=7.0, color="darkgreen")
    if i < n - 1:
        ax.annotate("", xy=(x0 + bw + 0.018, 0.65), xytext=(x0 + bw, 0.65),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="k"))
ax.text(0.5, 0.06,
        "Key lesson: the per-call gate (P2) is NECESSARY but NOT SUFFICIENT — "
        "only the full-run equivalence (P5) can clear a change to the residual.",
        ha="center", fontsize=8.5, style="italic", color="darkred")
fig.suptitle("F1 'drop the 60k resample' — validation workflow", fontsize=12)
fig.savefig(os.path.join(FIGS, "f1_workflow.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 5: variant tradeoff — WHY M500 ----------------
VORDER = ["baseline", "M2000", "M1000", "M500", "M200", "Mnodes"]
VNPTS = {"baseline": "~60k", "M2000": "2000", "M1000": "1000",
         "M500": "500", "M200": "200", "Mnodes": "adaptive"}
agg = {v: {"t": [], "r": []} for v in VORDER}
for c in pc:
    for ph in pc[c]:
        for v, d in pc[c][ph].items():
            if v in agg:
                agg[v]["t"] += d["t"]; agg[v]["r"] += d["r"]
mean_t = {v: (np.mean(agg[v]["t"]) if agg[v]["t"] else np.nan) for v in VORDER}
worst_r = {v: (max(agg[v]["r"]) if agg[v]["r"] else 0.0) for v in VORDER}

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))
xv = np.arange(len(VORDER))
tcol = ["tab:gray" if v == "baseline" else ("tab:green" if v == "M500" else "tab:blue")
        for v in VORDER]
ba = axa.bar(xv, [mean_t[v] for v in VORDER], color=tcol)
for b, v in zip(ba, VORDER):
    sp = mean_t["baseline"] / mean_t[v]
    axa.text(b.get_x() + b.get_width() / 2, mean_t[v] + 8,
             f"{mean_t[v]:.0f} ms\n{sp:.2f}x", ha="center", fontsize=7.5)
axa.set_xticks(xv); axa.set_xticklabels([f"{v}\n({VNPTS[v]})" for v in VORDER], fontsize=8)
axa.set_ylabel("mean per-call time [ms] (all configs)")
axa.set_title("(a) speed: 60k baseline vs coarse options (≈flat across npts)")

cv = VORDER[1:]  # skip baseline (it's the 0 reference)
rcol = ["tab:green" if v == "M500" else ("tab:purple" if v == "Mnodes" else "tab:blue") for v in cv]
bb = axb.bar(np.arange(len(cv)), [worst_r[v] for v in cv], color=rcol)
for b, v in zip(bb, cv):
    axb.text(b.get_x() + b.get_width() / 2, worst_r[v] * 1.25, f"{worst_r[v]:.1e}",
             ha="center", fontsize=7.5)
axb.axhline(GATE, color="r", ls="--", label="0.3% gate")
axb.set_yscale("log"); axb.set_ylim(1e-7, 1e-2)
axb.set_xticks(np.arange(len(cv))); axb.set_xticklabels([f"{v}\n({VNPTS[v]})" for v in cv], fontsize=8)
axb.set_ylabel("worst rel_dMdt (all configs)")
axb.set_title("(b) accuracy: npts-INSENSITIVE in [200,2000]")
axb.legend(loc="upper right")
fig.suptitle("Why 500: every coarse option beats the 60k baseline ~1.5x (speed ~flat across npts) and is "
             "npts-insensitive (all << gate).\nM500 (green) gives a GUARANTEED fixed-resolution "
             "min_T/monotonic check; Mnodes (purple) uses only LSODA's variable adaptive nodes — fine "
             "here, less predictable on an unseen config.", fontsize=9.5)
fig.savefig(os.path.join(FIGS, "f1_variant_tradeoff.png")); plt.close(fig)

print("wrote:", ", ".join(sorted(os.listdir(FIGS))))