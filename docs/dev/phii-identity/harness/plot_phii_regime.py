#!/usr/bin/env python3
"""Figures for the C3c confinement regime switch, from the reduced trajectory CSV.

Answers three questions the CSVs alone do not make legible:

  (a) WHAT confinement is        -- P_C3a against the confining pressure it is
                                    compared to, along a real trajectory.
  (b) WHERE the regimes fall     -- the ratio P_C3a/P_conf against its threshold,
                                    banded by phase.
  (c) WHAT ACTUALLY CHANGED      -- P_HII and P_drive, stock arm vs C3c arm.

P_C3a is recomputed here for EVERY row, including the rows where the scheme
suppresses it. That is the point: the delivered ``P_HII`` is 0.0 on the confined
branch, so the stored field cannot show you where the switch nearly flipped.

Constants come from trinity's own loader (read_param on the run's base param), so
this figure cannot drift from the code it documents.

Usage:
    python docs/dev/phii-identity/harness/plot_phii_regime.py \
        --csv docs/dev/phii-identity/data/b7_regime_trajectory.csv \
        --param docs/dev/transition/pdv-trigger/runs/params/bench5/bench3_m1e5_r5__none_diag.param \
        --out fig/phii_regime
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from trinity._input import read_param  # noqa: E402
from trinity._functions import unit_conversions as cvt  # noqa: E402

PHASE_ORDER = ["energy", "implicit", "transition", "momentum"]
PHASE_C = {"energy": "#4C72B0", "implicit": "#55A868",
           "transition": "#DD8452", "momentum": "#C44E52"}
C_CONF = "#8172B3"   # confining pressure
C_C3A = "#CCB974"    # the photoionised pressure candidate
C_STOCK = "#937860"  # stock arm


def load(csv_path):
    arms = defaultdict(list)
    with open(csv_path) as fh:
        # Skip the provenance header (C-6): those '#' lines precede the real header row.
        body = (ln for ln in fh if not ln.startswith("#"))
        for row in csv.DictReader(body):
            def f(k):
                v = row.get(k, "")
                if v in ("", "None", "nan"):
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None
            arms[row["arm"]].append({
                "t": f("t_now"), "R2": f("R2"), "phase": row["current_phase"],
                "Qi": f("Qi"), "fabs": f("shell_fAbsorbedIon"),
                "Pb": f("Pb"), "P_ram": f("P_ram"), "P_HII": f("P_HII"),
                "P_drive": f("P_drive"), "n_IF_Str": f("n_IF_Str"),
                "n_IF_Str_raw": f("n_IF_Str_raw"), "shell_n0": f("shell_n0"),
            })
    return arms


def p_c3a_factory(param_path):
    """Return P_C3a(R2, Qi, f_abs) in au, using the run's own loaded constants.

    Mirrors get_bubbleParams.get_phii_c3c exactly:
        n = sqrt(3 Qi f_abs / (4 pi chi_e alpha_B R2**3))
        P = (mu_convert/mu_ion_shell) * n * k_B * T
    """
    p = read_param.read_param(str(param_path))
    chi_e = p["chi_e_shell"].value
    alpha_B = p["caseB_alpha"].value
    k_B = p["k_B"].value
    T = p["TShell_ion"].value
    mu_ratio = p["mu_convert"].value / p["mu_ion_shell"].value

    def f(R2, Qi, fabs):
        if not R2 or not Qi or R2 <= 0 or Qi <= 0:
            return None
        fa = fabs if (fabs is not None and 0.0 <= fabs <= 1.0) else 1.0
        Qi_abs = Qi * fa
        denom = 4.0 * math.pi * chi_e * alpha_B * R2 ** 3
        if denom <= 0 or Qi_abs <= 0:
            return None
        return mu_ratio * math.sqrt(3.0 * Qi_abs / denom) * k_B * T

    f.consts = dict(chi_e=chi_e, alpha_B=alpha_B, k_B=k_B, T=T, mu_ratio=mu_ratio)
    return f


def phase_spans(rows):
    """Contiguous [t0, t1, phase] spans, for banding the time axis."""
    spans, cur, t0 = [], None, None
    for r in rows:
        if r["phase"] != cur:
            if cur is not None:
                spans.append((t0, r["t"], cur))
            cur, t0 = r["phase"], r["t"]
    if cur is not None:
        spans.append((t0, rows[-1]["t"], cur))
    return spans


def band_phases(ax, spans, alpha=0.10):
    for t0, t1, ph in spans:
        ax.axvspan(t0, t1, color=PHASE_C.get(ph, "0.5"), alpha=alpha, lw=0, zorder=0)


def mark_zeros(ax, rows, label):
    """Draw P_HII == 0 as an explicit stripe on the bottom axis.

    A log axis silently DROPS zeros, so the confined branch -- the whole point of
    the scheme -- would otherwise render as absence. Draw it as a mark instead.
    """
    seg = [r["t"] for r in rows if not r["P_HII"]]
    if not seg:
        return
    ax.plot(seg, [0.02] * len(seg), transform=ax.get_xaxis_transform(),
            color="k", lw=4.0, solid_capstyle="butt", alpha=0.9,
            clip_on=False, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--param", required=True)
    ap.add_argument("--out", required=True, help="output basename (no extension)")
    args = ap.parse_args()

    arms = load(args.csv)
    if "c3c" not in arms:
        raise SystemExit(f"csv has arms {list(arms)}; need at least 'c3c'")
    p_c3a = p_c3a_factory(args.param)
    K = cvt.Pb_au2_KcmInv  # au -> P/k_B [K cm^-3]

    c3c = arms["c3c"]
    stock = arms.get("stock")
    spans = phase_spans(c3c)

    # --- derived series on the C3c arm -----------------------------------
    for r in c3c:
        r["P_C3a"] = p_c3a(r["R2"], r["Qi"], r["fabs"])
        # The confining pressure the switch actually compares against is
        # params['Pb'] -- which IS the wind ram pressure in the momentum phase,
        # because run_momentum_phase assigns it so.
        r["P_conf"] = r["Pb"]
        r["ratio"] = (r["P_C3a"] / r["P_conf"]
                      if r["P_C3a"] and r["P_conf"] and r["P_conf"] > 0 else None)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), dpi=140)
    (axA, axB), (axC, axD) = axes

    # ================= (a) the mechanism ==================================
    band_phases(axA, spans)
    t = [r["t"] for r in c3c]
    axA.plot(t, [r["P_conf"] * K if r["P_conf"] else None for r in c3c],
             color=C_CONF, lw=2.0, label=r"$P_{\rm conf}$  (confining: $P_b$, $=P_{\rm ram}$ in momentum)")
    axA.plot(t, [r["P_C3a"] * K if r["P_C3a"] else None for r in c3c],
             color=C_C3A, lw=2.0, ls="--", label="$P_{\\rm C3a}$  (cavity Strömgren, always computed)")
    axA.plot(t, [(r["P_HII"] * K if r["P_HII"] else float("nan")) for r in c3c],
             color="k", lw=2.6, alpha=0.85, label=r"$P_{\rm HII}$ delivered (0 where confined)")
    mark_zeros(axA, c3c, r"$P_{\rm HII}=0$: confined, skin transmits $P_{\rm conf}$")
    axA.set_yscale("log")
    axA.set_ylabel(r"$P/k_B$  [K cm$^{-3}$]")
    axA.set_title("(a) confinement: the ionised gas is a skin while "
                  r"$P_{\rm C3a}\leq P_{\rm conf}$", fontsize=10)
    axA.legend(fontsize=7.5, loc="lower left", framealpha=0.92)

    # ================= (b) the regime switch ==============================
    band_phases(axB, spans)
    axB.axhline(1.0, color="k", lw=1.2, ls=":")
    rt = [(r["t"], r["ratio"]) for r in c3c if r["ratio"]]
    axB.plot([x for x, _ in rt], [y for _, y in rt], color="k", lw=1.8)
    axB.fill_between([x for x, _ in rt], 1.0, [y for _, y in rt],
                     where=[y >= 1.0 for _, y in rt], color=PHASE_C["momentum"],
                     alpha=0.35, lw=0, label="DRIVING: confinement fails")
    axB.fill_between([x for x, _ in rt], [y for _, y in rt], 1.0,
                     where=[y < 1.0 for _, y in rt], color=PHASE_C["energy"],
                     alpha=0.30, lw=0, label="CONFINED: skin transmits, $P_{\\rm HII}=0$")
    axB.set_yscale("log")
    axB.set_ylabel(r"$P_{\rm C3a}\,/\,P_{\rm conf}$")
    axB.set_title("(b) the regime switch, banded by phase", fontsize=10)
    axB.legend(fontsize=7.5, loc="upper left", framealpha=0.92)

    # ================= (c) what P_HII became ==============================
    band_phases(axC, spans)
    if stock:
        axC.plot([r["t"] for r in stock],
                 [r["P_HII"] * K if r["P_HII"] else None for r in stock],
                 color=C_STOCK, lw=2.2, label="stock $P_{\\rm HII}$ (capped Strömgren)")
        # Drawn in a contrasting colour ON TOP of the stock P_HII curve: the two
        # are equal to ~1e-16, so same-coloured lines would hide the very identity
        # this panel exists to show.
        axC.plot([r["t"] for r in stock],
                 [r["Pb"] * K if r["Pb"] else None for r in stock],
                 color="white", lw=1.1, ls=(0, (2, 3)), alpha=1.0,
                 label=r"stock $P_b$ — rides exactly on it: the relabelling")
    axC.plot(t, [(r["P_HII"] * K if r["P_HII"] else float("nan")) for r in c3c],
             color="k", lw=2.2, label=r"C3c $P_{\rm HII}$ (driving branch)")
    mark_zeros(axC, c3c, r"C3c $P_{\rm HII}=\,$0 exactly (confined branch)")
    axC.set_yscale("log")
    axC.set_xlabel("t [Myr]")
    axC.set_ylabel(r"$P_{\rm HII}/k_B$  [K cm$^{-3}$]")
    axC.set_title(r"(c) $P_{\rm HII}$: stock tracked $P_b$ exactly; C3c does not", fontsize=10)
    axC.legend(fontsize=7.5, loc="lower left", framealpha=0.92)

    # ================= (d) the pressure that drives =======================
    band_phases(axD, spans)
    if stock:
        axD.plot([r["t"] for r in stock],
                 [r["P_drive"] * K if r["P_drive"] else None for r in stock],
                 color=C_STOCK, lw=2.2, label="stock $P_{\\rm drive}$")
    axD.plot(t, [r["P_drive"] * K if r["P_drive"] else None for r in c3c],
             color="k", lw=2.2, label="C3c $P_{\\rm drive}$")
    axD.set_yscale("log")
    axD.set_xlabel("t [Myr]")
    axD.set_ylabel(r"$P_{\rm drive}/k_B$  [K cm$^{-3}$]")
    axD.set_title(r"(d) the pressure the shell actually feels", fontsize=10)
    axD.legend(fontsize=7.5, loc="lower left", framealpha=0.92)

    # phase legend across the bottom
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_C[p], alpha=0.35) for p in PHASE_ORDER]
    fig.legend(handles, PHASE_ORDER, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.png / .pdf")

    # ---- the numbers the figure is claiming, to stdout -------------------
    print("\nconfined fraction by phase (C3c arm, ratio < 1):")
    by = defaultdict(list)
    for r in c3c:
        if r["ratio"]:
            by[r["phase"]].append(r["ratio"])
    for ph in PHASE_ORDER:
        v = by.get(ph)
        if v:
            frac = sum(1 for x in v if x < 1.0) / len(v)
            print(f"  {ph:11s} n={len(v):4d}  confined={frac:6.1%}  "
                  f"ratio med={sorted(v)[len(v)//2]:.4g}")
    if stock:
        # The stock identity, re-checked from the table rather than assumed.
        d = [abs(r["P_HII"] - r["Pb"]) / r["Pb"]
             for r in stock if r["P_HII"] and r["Pb"] and r["Pb"] > 0]
        if d:
            print(f"\nstock |P_HII - Pb|/Pb over {len(d)} rows: "
                  f"max={max(d):.3e}  med={sorted(d)[len(d)//2]:.3e}")

        # Which WAY the drive moved, per phase. The sign is not uniform -- C3c
        # lowers the energy-phase drive (the dt_switchon ramp is no longer bypassed)
        # and raises it once confinement fails -- so a single global number would
        # average away the whole effect.
        sp = sorted((r["t"], r["P_drive"]) for r in stock if r["P_drive"])
        st, sv = [x for x, _ in sp], [y for _, y in sp]

        def interp(x):
            if x <= st[0]:
                return sv[0]
            if x >= st[-1]:
                return sv[-1]
            lo, hi = 0, len(st) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if st[mid] <= x:
                    lo = mid
                else:
                    hi = mid
            if st[hi] == st[lo]:
                return sv[lo]
            w = (x - st[lo]) / (st[hi] - st[lo])
            return sv[lo] * (1 - w) + sv[hi] * w

        print("\nP_drive ratio C3c/stock (stock interpolated onto the C3c t grid):")
        agg = defaultdict(list)
        for r in c3c:
            if r["P_drive"] and r["t"] is not None:
                s = interp(r["t"])
                if s > 0:
                    agg[r["phase"]].append(r["P_drive"] / s)
        for ph in PHASE_ORDER:
            v = sorted(agg.get(ph, []))
            if v:
                print(f"  {ph:11s} n={len(v):4d}  min={v[0]:.4g}  "
                      f"med={v[len(v)//2]:.4g}  max={v[-1]:.4g}")


if __name__ == "__main__":
    main()
