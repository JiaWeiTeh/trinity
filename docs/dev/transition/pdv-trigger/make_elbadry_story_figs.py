#!/usr/bin/env python3
"""Figures for ELBADRY_THETA_STORY.html — document the El-Badry theta closed form:
what it is, what we impose, what we check, and the physics behind it.

All inputs are committed CSVs in data/ (no sim run here). Writes story_elbadry_f*.png next to this script.
Run:  python docs/dev/transition/pdv-trigger/make_elbadry_story_figs.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False  # dev fig — never LaTeX (plot_base guard mirrors this)
import matplotlib.pyplot as plt
import numpy as np

# Figures land directly in the workstream folder (the `fig/` subdir is gitignored as scratch).
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = HERE

A_MIX = 3.5          # El-Badry+2019 Eq 37 fit
LDV = 3.0            # lambda*delta_v (pc km/s), calibrated
THETA_MAX = 0.99
THETA_FIRE = 0.95    # cooling_balance fires at theta >= 0.95  ((Lgain-Lloss)/Lgain < 0.05)


def theta_elbadry(n, ldv=LDV, a_mix=A_MIX):
    """El-Badry+2019 Eq 37/38 closed form. n in cm^-3, ldv in pc km/s."""
    X = a_mix * np.sqrt(ldv * n)
    return X / (11.0 / 5.0 + X)


def n_fire(ldv=LDV, a_mix=A_MIX, thr=THETA_FIRE):
    """Density at which theta crosses the firing threshold: solve theta(n)=thr."""
    # thr = X/(11/5+X) -> X = thr*(11/5)/(1-thr); X = a_mix*sqrt(ldv*n)
    X = thr * (11.0 / 5.0) / (1.0 - thr)
    return (X / a_mix) ** 2 / ldv


def load_csv(name):
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ shared config table
shadow = load_csv("shadow_te_fate.csv")
CFG = {}
for r in shadow:
    CFG[r["config"]] = dict(n=float(r["nCore_cm3"]), m=float(r["mCloud_Msun"]),
                            theta=float(r["theta_max"]), fate=r["fate"])

FATE_COLOR = {"SHELL_COLLAPSED": "#c0392b", "shell_collapsed": "#c0392b",
              "stopping_time": "#2f9e44", "energy_collapsed": "#7b4fb0",
              "CRASHED_EARLY": "#888888", "large_radius": "#2f9e44",
              "velocity_runaway": "#e8842a"}


# ================================================================== FIG 1: the closed form
def fig1():
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    n = np.logspace(-1, 6.5, 400)
    for ldv, c, ls in [(1, "#9bb7d4", "--"), (3, "#2c6fb3", "-"), (10, "#1b3a5c", ":")]:
        ax.plot(n, np.minimum(theta_elbadry(n, ldv), THETA_MAX), ls, color=c, lw=2.2,
                label=fr"$\lambda\delta v={ldv}$ pc·km/s" + (" (used)" if ldv == 3 else ""))
    ax.axhline(THETA_FIRE, color="#c0392b", lw=1.3, ls="-.")
    ax.text(0.12, THETA_FIRE + 0.006, r"firing threshold  $\theta=0.95$", color="#c0392b", fontsize=10)
    ax.axhline(THETA_MAX, color="#555", lw=1.0, ls=":")
    ax.text(0.12, THETA_MAX + 0.004, r"cap  $\theta_{\max}=0.99$", color="#555", fontsize=9)
    nf = n_fire()
    ax.axvline(nf, color="#c0392b", lw=1.0, ls=":")
    ax.text(nf * 1.15, 0.30, fr"$n_{{\rm fire}}\approx{nf:.0f}$ cm$^{{-3}}$", color="#c0392b", fontsize=10)
    # overlay the configs (lambda dv = 3)
    for name, d in CFG.items():
        th = min(theta_elbadry(d["n"]), THETA_MAX)
        ax.scatter([d["n"]], [th], s=46, color=FATE_COLOR.get(d["fate"], "#333"),
                   edgecolor="k", lw=0.5, zorder=5)
        ax.annotate(name, (d["n"], th), fontsize=7.3, xytext=(4, -9),
                    textcoords="offset points", color="#333")
    ax.set_xscale("log")
    ax.set_xlabel(r"local density at the shell  $n$  (cm$^{-3}$)")
    ax.set_ylabel(r"$\theta = L_{\rm cool}/L_{\rm mech}$  (imposed)")
    ax.set_title("Fig 1 — El-Badry closed form: "
                 r"$\theta(n)=\dfrac{A_{\rm mix}\sqrt{\lambda\delta v\,n}}{11/5+A_{\rm mix}\sqrt{\lambda\delta v\,n}}$"
                 f"   ($A_{{\\rm mix}}={A_MIX}$)", fontsize=11.5)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "story_elbadry_f1_closedform.png"), dpi=130)
    plt.close(fig)


# ================================================================== FIG 2: what we impose
def fig2():
    names = ["diffuse_probe", "small_1e6", "fail_repro", "large_diffuse_lowsfe",
             "be_sphere", "midrange_pl0", "pl2_steep", "simple_cluster"]
    names = [n for n in names if n in CFG]
    th = [min(theta_elbadry(CFG[n]["n"]), THETA_MAX) for n in names]
    native = 0.66  # native radiative theta peak for compact/dense clouds (FINDINGS sec6a)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(names))
    bars = ax.bar(x, th, color=[FATE_COLOR.get(CFG[n]["fate"], "#333") for n in names],
                  edgecolor="k", lw=0.5)
    ax.axhline(THETA_FIRE, color="#c0392b", lw=1.3, ls="-.")
    ax.text(len(names) - 0.5, THETA_FIRE + 0.006, "fires  θ≥0.95", color="#c0392b",
            fontsize=9.5, ha="right")
    ax.axhspan(0, native, color="#7b4fb0", alpha=0.08)
    ax.axhline(native, color="#7b4fb0", lw=1.1, ls="--")
    ax.text(0.0, native - 0.055, "native radiative θ peak ≈ 0.66\n(stock cooling_balance never fires these)",
            color="#7b4fb0", fontsize=8.6)
    for xi, t in zip(x, th):
        ax.text(xi, t + 0.008, f"{t:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={CFG[n]['n']:g})" for n in names], fontsize=7.8, rotation=12)
    ax.set_ylabel(r"$\theta$")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig 2 — What we IMPOSE:  effective $L_{\\rm loss}=\\max(L_{\\rm cool}{+}L_{\\rm leak},\\ "
                 "\\theta\\,L_{\\rm mech})$\n"
                 "El-Badry θ (bars) always beats native radiative θ  →  resolved-wins = 0/N everywhere",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "story_elbadry_f2_impose.png"), dpi=130)
    plt.close(fig)


# ================================================================== FIG 3: what we check
def fig3():
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    th = np.linspace(0.80, 1.0, 200)
    ax.plot(th, 1 - th, "-", color="#2c6fb3", lw=2.4)
    ax.axhline(0.05, color="#c0392b", lw=1.3, ls="-.")
    ax.axvline(THETA_FIRE, color="#c0392b", lw=1.0, ls=":")
    ax.fill_between(th, 0, 1 - th, where=(th >= THETA_FIRE), color="#2f9e44", alpha=0.14)
    ax.text(0.955, 0.14, "FIRES\n(energy→momentum)", color="#1b7a3a", fontsize=10, ha="left")
    ax.text(0.905, 0.075, r"$(L_{\rm gain}-L_{\rm loss})/L_{\rm gain} < 0.05$", color="#c0392b", fontsize=10)
    ax.text(0.808, 0.165, "energy-driven\n(no fire)", color="#555", fontsize=10)
    # configs
    for name, d in CFG.items():
        t = min(theta_elbadry(d["n"]), THETA_MAX)
        ax.scatter([t], [1 - t], s=42, color=FATE_COLOR.get(d["fate"], "#333"),
                   edgecolor="k", lw=0.4, zorder=5)
    ax.set_xlabel(r"imposed $\theta = L_{\rm loss}/L_{\rm mech}$")
    ax.set_ylabel(r"net energy fraction  $(L_{\rm gain}-L_{\rm loss})/L_{\rm gain} = 1-\theta$")
    ax.set_title("Fig 3 — What we CHECK: the cooling_balance trigger  ≡  θ ≥ 0.95", fontsize=11.5)
    ax.set_xlim(0.80, 1.0)
    ax.set_ylim(0, 0.20)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "story_elbadry_f3_check.png"), dpi=130)
    plt.close(fig)


# ================================================================== FIG 4: Stage-A fate map
def fig4():
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    seen = set()
    for name, d in CFG.items():
        c = FATE_COLOR.get(d["fate"], "#333")
        lab = d["fate"] if d["fate"] not in seen else None
        seen.add(d["fate"])
        ax.scatter([d["n"]], [d["m"]], s=150 + 260 * (d["theta"] - 0.85),
                   color=c, edgecolor="k", lw=0.7, alpha=0.9, label=lab)
        ax.annotate(name, (d["n"], d["m"]), fontsize=7.5, xytext=(6, 4),
                    textcoords="offset points")
    ax.axvline(n_fire(), color="#c0392b", lw=1.0, ls=":")
    ax.text(n_fire() * 1.2, 3e4, "n_fire≈48", color="#c0392b", fontsize=9, rotation=90)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"core density $n_{\rm core}$ (cm$^{-3}$)")
    ax.set_ylabel(r"cloud mass $M_{\rm cloud}$ (M$_\odot$)")
    ax.set_title("Fig 4 — Stage-A fate map (θ imposed, PRE-PR#715 code): "
                 "dense→recollapse, diffuse→survive", fontsize=10.8)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95, title="fate (marker size ∝ θ)")
    ax.grid(alpha=0.22, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "story_elbadry_f4_fatemap.png"), dpi=130)
    plt.close(fig)


# ================================================================== FIG 5: PR#715 reversal
def fig5():
    rows = load_csv("newcode_default_vs_theta.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.8))
    labels, R2, V2, cols = [], [], [], []
    for r in rows:
        tag = "default" if r["path"] == "default" else "θ imposed"
        labels.append(f"{r['config']}\n{tag}")
        R2.append(float(r["end_R2_pc"]))
        V2.append(float(r["end_v2_pcMyr"]))
        cols.append("#2f9e44" if r["path"] == "default" else "#c0392b")
    x = np.arange(len(labels))
    a1.bar(x, R2, color=cols, edgecolor="k", lw=0.5)
    a1.set_yscale("log"); a1.set_ylabel("final R2 (pc)")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8)
    a1.set_title("final radius", fontsize=11)
    for xi, v in zip(x, R2):
        a1.text(xi, v * 1.1, f"{v:g}", ha="center", fontsize=8)
    a2.bar(x, V2, color=cols, edgecolor="k", lw=0.5)
    a2.axhline(0, color="k", lw=0.8)
    a2.axhline(-500, color="#e8842a", lw=1.0, ls="--")
    a2.text(len(x) - 0.5, -470, "velocity_runaway cap −500", color="#e8842a", fontsize=8, ha="right")
    a2.set_ylabel("final v2 (pc/Myr)")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8)
    a2.set_title("final velocity  (+out / −in)", fontsize=11)
    fig.suptitle("Fig 5 — On merged PR#715 code: DEFAULT expands (green); imposing El-Badry θ recollapses (red)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "story_elbadry_f5_reversal.png"), dpi=130)
    plt.close(fig)


# ================================================================== FIG 6: regime error (PdV vs radiative)
def fig6():
    rows = load_csv("live_pdv_decomp.csv")
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    names = [r["config"] for r in rows]
    pdv = [float(r["PdV_over_Lmech_atEbpeak"]) for r in rows]
    rad = [float(r["Lbub_over_Lmech_atEbpeak"]) for r in rows]
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, pdv, w, label="PdV / L_mech  (inertial loading)", color="#c0392b", edgecolor="k", lw=0.5)
    ax.bar(x + w / 2, rad, w, label="L_bub(radiative) / L_mech", color="#2c6fb3", edgecolor="k", lw=0.5)
    ax.axhline(1.0, color="#555", lw=0.9, ls=":")
    for xi, (p, rr) in enumerate(zip(pdv, rad)):
        dom = "PdV-dominated" if p > rr else "radiative-dom."
        ax.text(xi, max(p, rr) + 0.03, dom, ha="center", fontsize=8.5,
                color="#c0392b" if p > rr else "#2c6fb3")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.2, rotation=8)
    ax.set_ylabel("fraction of L_mech at the Eb-peak")
    ax.set_title("Fig 6 — Why imposing a RADIATIVE θ is a regime error for massive clouds\n"
                 "(live decomposition, main's data/live_pdv_decomp.csv)", fontsize=10.6)
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "story_elbadry_f6_regime.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("wrote 6 figures to", FIG)
    print("n_fire(lambda_dv=3) =", round(n_fire(), 2), "cm^-3")
