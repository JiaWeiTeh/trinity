#!/usr/bin/env python3
"""f_kappa(n_H) FUNCTIONAL FORM — a closed-form calibration target, composed from
   (verified literature target) + (measured TRINITY baseline) + (measured leverage).

THE ASK (this session): instead of waiting for the 819-combo HPC sweep to FIT f_κ(n_H)
cold, give a functional form f_κ(n_H) we can USE now, grounded in literature / other
quantities. The sweep then CONFIRMS/REFINES it rather than discovering it.

THE COMPOSITION (three separable pieces — each independently checkable):

  (1) TARGET  θ*  — the obs/3D loss fraction we calibrate toward.
        Lancaster, Ostriker, Kim & Kim 2021 (ApJ 914, 89 = arXiv:2104.07691; Paper II
        914, 90 = 2104.07722): 1-Θ ~ 0.1-0.01, i.e. Θ ≈ 0.9-0.99, and CRUCIALLY
        "generic ... over more than three orders of magnitude in density" (abstract,
        verbatim) -> θ* is ~FLAT in n_H over the GMC range. We adopt θ* = 0.90 (and
        report 0.95). NOT a rising El-Badry sqrt(n) curve: El-Badry+2019 (MNRAS 490,
        1961; arXiv:1902.09547) is an ambient n~0.1-10 SUPERbubble paper whose theta(n)
        must not be pushed to GMC density; its specific psi/theta algebra is UNVERIFIED
        here (every arXiv/ADS/journal host 403s in-container). So the verified, GMC-
        regime-appropriate anchor is the Lancaster PLATEAU.

  (2) BASELINE  θ0(n_H)  — TRINITY's EMERGENT loss fraction at f_κ=1 (no boost), which
        RISES with density. Measured at blowout for 6 reference configs (fmix_table.csv:
        0.25 @ 1e2 -> 0.70 @ 1e6). We fit logit(θ0) = a + b*log10(n_H). The density
        structure of f_κ comes from THIS rising baseline under a FLAT target -> this is
        what breaks the "flat target == 0.95 trigger" degeneracy flagged in FINDINGS §2a.

  (3) LEVERAGE  q  — how θ responds to f_κ. Measured from the full-run grid
        (kappa_blowout_calibration.csv: f_κ=1,2,4 on compact/mid/diffuse). The docs'
        single power law θ ∝ f_κ^0.63 is UNSTABLE: the measured raw exponent p falls
        0.42(diffuse)->0.21(mid)->0.15(compact) as θ0 rises — a SATURATION artifact (θ
        can't exceed 1), which is exactly why kappa_calibration_estimate.csv is
        "optimistic". The fix is to fit leverage in ODDS (logit) space, which IS bounded
        by θ->1:   logit(θ(f_κ)) = logit(θ0) + q*ln(f_κ).  Measured q is far more
        stable (~0.5 diffuse .. ~0.7 mid) though it still rises with density (the
        de-conflation the 819-sweep resolves).

INVERSION (the functional form):
        logit θ(f_κ; n_H) = logit θ0(n_H) + q*ln f_κ
  =>    f_κ(n_H) = exp{ [ logit(θ*) - logit(θ0(n_H)) ] / q }
  and since logit θ0(n_H) is linear in log10 n_H, this is APPROXIMATELY a power law
        f_κ(n_H) ≈ A * n_H^(-s)   (s reported below; s STEEPENS if q rises with n_H).

PHYSICAL BRACKET (honest ceiling): a real Spitzer-conduction boost saturates. Cowie &
McKee 1977: q_sat = 5 φ_s ρ c_s^3 (φ_s≈0.3), so the saturated effective conductivity
∝ n_H^(+1) — the f_κ CEILING RISES with density, OPPOSITE to the required f_κ (which
FALLS). Where required > ceiling (the diffuse end, where required f_κ is tens-hundreds)
a pure conduction boost is unphysical: that regime needs El-Badry's temperature-
INDEPENDENT turbulent-mixing diffusivity κ_mix, not f_κ·Spitzer. Magnetic suppression
(Narayan & Medvedev 2001 f~0.2; ISM ~0.1) makes the same point: f_κ>1 is a PROXY for
turbulent mixing, not literal extra Spitzer conduction.

REPRODUCE (from repo root; reads only committed CSVs, runs NO sims):
    python docs/dev/transition/pdv-trigger/data/make_fkappa_functional_form.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/fkappa_functional_form.csv
    docs/dev/transition/pdv-trigger/fkappa_functional_form.png
"""

import csv
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)

TARGETS = (0.90, 0.95)          # Lancaster plateau (primary 0.90; 0.95 == shipped trigger)


def _logit(p):
    return math.log(p / (1.0 - p))


def _inv_logit(x):
    return 1.0 / (1.0 + math.exp(-x))


def _read_baselines():
    """6 anchors: nCore -> θ0 at blowout (f_κ=1), from fmix_table.csv."""
    rows = list(csv.DictReader(open(os.path.join(_HERE, "fmix_table.csv"))))
    # nCore is not in fmix_table; join from kappa_calibration_estimate.csv (same configs).
    ncore = {r["config"]: float(r["nCore"])
             for r in csv.DictReader(open(os.path.join(_HERE, "kappa_calibration_estimate.csv")))}
    out = []
    for r in rows:
        cfg = r["config"]
        out.append((cfg, ncore[cfg], float(r["Lcool_over_Lmech_at_blowout"])))
    return sorted(out, key=lambda t: t[1])


def _measure_leverage():
    """Per-config odds-space leverage q AND raw power-law p, from the full-run grid.

    logit(θ) = logit(θ0) + q*ln(f_κ)      (bounded, stable)
    ln(θ)    = ln(θ0)    + p*ln(f_κ)      (the docs' form — unstable as θ->1)
    Only θ<0.99 points are usable in logit space (θ>=1 -> odds blows up / fires).
    """
    rows = list(csv.DictReader(open(os.path.join(_HERE, "kappa_blowout_calibration.csv"))))
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append((float(r["f_kappa"]),
                                                    float(r["theta_blowout"])))
    res = {}
    for cfg, pts in by_cfg.items():
        pts = sorted(pts)
        good = [(fk, th) for fk, th in pts if 0.0 < th < 0.99]
        lnfk = np.array([math.log(fk) for fk, _ in good])
        q = float(np.polyfit(lnfk, [_logit(th) for _, th in good], 1)[0]) if len(good) >= 2 else float("nan")
        p = float(np.polyfit(lnfk, [math.log(th) for _, th in good], 1)[0]) if len(good) >= 2 else float("nan")
        res[cfg] = {"q": q, "p_raw": p, "n_used": len(good),
                    "theta0": dict(pts).get(1.0)}
    return res


def main():
    base = _read_baselines()
    lev = _measure_leverage()

    # --- fit the baseline θ0(n_H): logit(θ0) = a + b*log10(n_H) ---
    x = np.array([math.log10(n) for _, n, _ in base])
    y = np.array([_logit(th) for _, _, th in base])
    b, a = np.polyfit(x, y, 1)          # slope, intercept
    resid = y - (a + b * x)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    print(f"baseline fit: logit(θ0) = {a:+.3f} {b:+.3f}*log10(n_H)   (RMS in logit = {rms:.3f})")

    # --- leverage summary ---
    qs = [v["q"] for v in lev.values() if v["q"] == v["q"]]
    q_med = float(np.median(qs))
    print("leverage (odds-space q vs raw power p):")
    for cfg, v in sorted(lev.items()):
        print(f"  {cfg:8s} θ0={v['theta0']:.3f}  q={v['q']:.3f}  p_raw={v['p_raw']:.3f}  (n={v['n_used']})")
    print(f"  -> median q = {q_med:.3f}  (use as the single-value leverage; q RISES with n_H -> sweep refines)")

    def fkappa(n_H, theta_star, q):
        l0 = a + b * math.log10(n_H)
        return math.exp((_logit(theta_star) - l0) / q)

    # --- the functional form on a grid + fit f_κ ≈ A * n_H^(-s) ---
    grid = np.logspace(2, 6, 25)
    rows_out = []
    for tgt in TARGETS:
        fk_grid = np.array([fkappa(n, tgt, q_med) for n in grid])
        s, lnA = -np.polyfit(np.log(grid), np.log(fk_grid), 1)[0], np.polyfit(np.log(grid), np.log(fk_grid), 1)[1]
        A = math.exp(lnA)
        print(f"θ*={tgt}: f_κ(n_H) ≈ {A:.3g} * n_H^(-{s:.3f})   "
              f"[f_κ(1e2)={fkappa(1e2,tgt,q_med):.1f}, f_κ(1e4)={fkappa(1e4,tgt,q_med):.2f}, "
              f"f_κ(1e6)={fkappa(1e6,tgt,q_med):.2f}]")

    # per-anchor table (csv): both targets, single-q and per-config-q where available
    cfg_q = {cfg: v["q"] for cfg, v in lev.items()}
    # map the 6 fmix configs onto the nearest measured-leverage config by density tier
    tier_q = {1e2: cfg_q.get("diffuse"), 1e4: cfg_q.get("mid"),
              1e5: cfg_q.get("compact"), 1e6: cfg_q.get("compact")}
    for cfg, n_H, th0 in base:
        rec = {"config": cfg, "nCore": f"{n_H:.0f}", "theta0": round(th0, 4)}
        for tgt in TARGETS:
            rec[f"fkappa_q{int(tgt*100)}_medq"] = round(fkappa(n_H, tgt, q_med), 3)
            tq = tier_q.get(n_H)
            rec[f"fkappa_q{int(tgt*100)}_tierq"] = round(fkappa(n_H, tgt, tq), 3) if tq else ""
        rows_out.append(rec)

    csv_path = os.path.join(_HERE, "fkappa_functional_form.csv")
    cols = ["config", "nCore", "theta0",
            "fkappa_q90_medq", "fkappa_q90_tierq", "fkappa_q95_medq", "fkappa_q95_tierq"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    # append the fitted parameters as a trailer comment block (machine + human readable)
    with open(csv_path, "a") as fh:
        fh.write(f"# fit: logit(theta0) = {a:+.4f} {b:+.4f}*log10(nH); RMS_logit={rms:.4f}\n")
        fh.write("# leverage q (odds-space): " +
                 ", ".join(f"{c}={v['q']:.3f}" for c, v in sorted(lev.items())) +
                 f"; median={q_med:.3f}\n")
        fh.write("# form: f_kappa(nH) = exp([logit(theta*) - (a + b*log10 nH)] / q)\n")
        fh.write("# target theta* = 0.90 (Lancaster 2021 plateau, density-independent); 0.95 == shipped trigger\n")
    print(f"wrote {csv_path}")

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from _trinity_style import COLORS, use_trinity_style
            use_trinity_style()
        except Exception:
            COLORS = {"diffuse": "#ff7f0e", "mid": "#9467bd", "compact": "#1f77b4", "dense": "#8c564b"}
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # LEFT: leverage in odds space vs raw — why the docs' power law is wrong
    fks = np.array([1, 2, 4])
    cal = {}
    for r in csv.DictReader(open(os.path.join(_HERE, "kappa_blowout_calibration.csv"))):
        cal.setdefault(r["config"], {})[float(r["f_kappa"])] = float(r["theta_blowout"])
    for cfg in ("diffuse", "mid", "compact"):
        th = np.array([cal[cfg][f] for f in fks])
        c = COLORS.get(cfg, "0.3")
        axL.plot(fks, th, "o-", color=c, lw=1.8, ms=6, label=f"{cfg} (θ0={th[0]:.2f})")
    axL.axhline(0.90, ls="--", color="#2ca02c", lw=1.2)
    axL.text(1.05, 0.905, "Lancaster θ*≈0.90", fontsize=8, color="#2ca02c")
    axL.set_xscale("log", base=2)
    axL.set_xlabel(r"$f_\kappa$ (cooling_boost_kappa)")
    axL.set_ylabel(r"emergent $\theta = L_{\rm cool}/L_{\rm mech}$ at blowout")
    axL.set_ylim(0, 1.05)
    axL.set_title("Measured leverage: θ saturates toward 1\n(odds-space q stable; raw power p is not)",
                  fontsize=10, fontweight="bold")
    axL.legend(fontsize=8.5, loc="lower right")

    # MIDDLE: the baseline fit θ0(n_H)
    nn = np.logspace(2, 6, 50)
    axM.plot(nn, [_inv_logit(a + b * math.log10(n)) for n in nn], "-", color="#1f77b4", lw=2,
             label=r"fit: logit$\,\theta_0=a+b\log_{10}n_H$")
    for cfg, n_H, th0 in base:
        axM.plot(n_H, th0, "o", color="#1f77b4", ms=7)
        axM.annotate(cfg.replace("_", " "), (n_H, th0), fontsize=6.5,
                     xytext=(0, 6), textcoords="offset points", ha="center", color="0.4")
    axM.axhspan(0.90, 0.99, color="#2ca02c", alpha=0.10, label="Lancaster target θ* (flat)")
    axM.set_xscale("log")
    axM.set_xlabel(r"$n_{\rm Core}$ [cm$^{-3}$]")
    axM.set_ylabel(r"baseline $\theta_0$ at $f_\kappa=1$")
    axM.set_ylim(0, 1.02)
    axM.set_title("TRINITY baseline RISES with density;\ntarget is FLAT -> f_κ(n_H) falls",
                  fontsize=10, fontweight="bold")
    axM.legend(fontsize=8.5, loc="lower right")

    # RIGHT: the functional form f_κ(n_H) + power-law fit + saturation ceiling direction
    for tgt, col in zip(TARGETS, ("#2ca02c", "#d62728")):
        fk = np.array([fkappa(n, tgt, q_med) for n in nn])
        s = -np.polyfit(np.log(nn), np.log(fk), 1)[0]
        axR.plot(nn, fk, "-", color=col, lw=2, label=fr"$\theta^*$={tgt}:  $f_\kappa\propto n_H^{{-{s:.2f}}}$")
    # the old optimistic estimate for contrast
    est = {float(r["nCore"]): float(r["fkappa_for_theta95"])
           for r in csv.DictReader(open(os.path.join(_HERE, "kappa_calibration_estimate.csv")))}
    axR.plot(sorted(est), [est[k] for k in sorted(est)], "x:", color="0.55", lw=1.2, ms=7,
             label="old power-law estimate (θ*=0.95, optimistic)")
    axR.annotate("saturation ceiling rises\n"r"($\kappa_{\rm sat}\propto n_H^{+1}$): diffuse end"
                 "\nunreachable by Spitzer boost\n-> needs El-Badry κ_mix",
                 xy=(1.3e2, 30), fontsize=7.2, color="0.35")
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel(r"$n_{\rm Core}$ [cm$^{-3}$]")
    axR.set_ylabel(r"$f_\kappa(n_H)$ needed")
    axR.set_title("The functional form: f_κ(n_H) ≈ A·n_H^(-s)\n(logistic leverage, median q; sweep refines q(n_H))",
                  fontsize=10, fontweight="bold")
    axR.legend(fontsize=8, loc="upper right")

    fig.suptitle("f_κ(n_H) functional form — composed from Lancaster target + TRINITY baseline + measured logistic leverage",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(_PDV, "fkappa_functional_form.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
