#!/usr/bin/env python3
"""New-findings diagnostic: what does the F0-ratio surge coincide with? (user Q, 2026-06-20)

Question (make no assumptions): the sudden surge in the cooling trigger ratio
r = (Lgain - Lloss)/Lgain -- does it coincide with feedback (Lmech), or with the
structure variables beta / delta / beta+delta crossing some value (e.g. the
negative-beta re-pressurisation / negative velocity structure)? Per config or all?

Measured on the committed data/c0_*_h0.csv (implicit rows only), per config:
  - corr(dr, dLmech), corr(dr, dbeta), corr(dr, ddelta) : step-to-step coincidence
  - mean ratio in beta<0 vs beta>=0 rows, and %(beta<0)
  - the epoch + (beta, delta) of the single largest up-jump in r
and a grouped-bar chart of the three correlations. Also writes the table to
data/surge_coincidence.csv so the numbers regenerate without re-running.

    python plot_surge.py docs/dev/transition/cleanroom/data/c0_*_h0.csv
"""
from __future__ import annotations

import csv
import glob
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            t = float(r["t_now"]); Lg = float(r["bubble_Lgain"]); Ll = float(r["bubble_Lloss"])
            Lm = float(r["Lmech_total"]); b = float(r["cool_beta"]); d = float(r["cool_delta"])
        except (ValueError, KeyError, TypeError):
            continue
        if Lg > 0:
            out.append((t, (Lg - Ll) / Lg, b, d, Lm))
    return out


def analyze(path):
    R = load(path)
    name = Path(path).name.replace("c0_", "").replace("_h0.csv", "")
    if len(R) < 5:
        return None
    t = [x[0] for x in R]; r = [x[1] for x in R]; b = [x[2] for x in R]
    d = [x[3] for x in R]; Lm = [x[4] for x in R]
    dr = [r[i] - r[i - 1] for i in range(1, len(R))]
    db = [b[i] - b[i - 1] for i in range(1, len(R))]
    dd = [d[i] - d[i - 1] for i in range(1, len(R))]
    dLm = [Lm[i] - Lm[i - 1] for i in range(1, len(R))]
    bneg = [r[i] for i in range(len(R)) if b[i] < 0]
    bpos = [r[i] for i in range(len(R)) if b[i] >= 0]
    jmax = max(range(len(dr)), key=lambda i: dr[i])  # index into dr (row jmax+1)
    return dict(
        name=name, n=len(R), pct_bneg=100 * len(bneg) / len(R),
        r_bneg=(sum(bneg) / len(bneg) if bneg else float("nan")),
        r_bpos=(sum(bpos) / len(bpos) if bpos else float("nan")),
        c_dLm=corr(dr, dLm), c_db=corr(dr, db), c_dd=corr(dr, dd),
        jump_t=t[jmax + 1], jump_dr=dr[jmax], jump_b=b[jmax + 1], jump_d=d[jmax + 1],
    )


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_h0.csv")))
    rows = [a for a in (analyze(p) for p in paths) if a]
    if not rows:
        sys.exit("no data")
    rows.sort(key=lambda a: a["name"])

    # persist the table (diagnostics-commit rule)
    cols = ["name", "n", "pct_bneg", "r_bneg", "r_bpos", "c_dLm", "c_db", "c_dd",
            "jump_t", "jump_dr", "jump_b", "jump_d"]
    with open(HERE / "data" / "surge_coincidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for a in rows:
            w.writerow({k: (f"{a[k]:.4f}" if isinstance(a[k], float) else a[k]) for k in cols})

    names = [a["name"] for a in rows]
    y = range(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    h = 0.26
    ax.barh([i + h for i in y], [a["c_dLm"] for a in rows], height=h, color="#0072B2",
            label=r"corr($\Delta r,\ \Delta L_{\rm mech}$)  (feedback)")
    ax.barh([i for i in y], [a["c_db"] for a in rows], height=h, color="#D55E00",
            label=r"corr($\Delta r,\ \Delta\beta$)  ($\beta$ drop $\Rightarrow$ re-pressurise)")
    ax.barh([i - h for i in y], [a["c_dd"] for a in rows], height=h, color="#009E73",
            label=r"corr($\Delta r,\ \Delta\delta$)")
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(-1, 1); ax.set_xlabel("step-to-step correlation with the cooling-ratio change  $\\Delta r$")
    ax.set_title("What the cooling-ratio surge coincides with: feedback up (+), $\\beta$ dropping (−)")
    ax.legend(fontsize=7.8, loc="upper left", framealpha=0.92)
    ax.text(0.5, -0.165, "all configs: $\\Delta r$ is positively correlated with feedback "
            "($\\Delta L_{\\rm mech}$) and negatively with $\\Delta\\beta$ — the surge co-moves with "
            "re-pressurisation,\nbut at no fixed $\\beta$ value (max-jump $\\beta$ spans −0.2 … +2.3; "
            "see data/surge_coincidence.csv)", transform=ax.transAxes, ha="center", va="top",
            fontsize=7.4, color="0.4")
    fig.tight_layout()

    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"surge_coincidence.{ext}", dpi=150)
    print(f"wrote {outdir}/surge_coincidence.(pdf,png) + data/surge_coincidence.csv ({len(rows)} configs)")
    print(f"{'config':22s} {'%b<0':>5s} {'r@b<0':>6s} {'r@b>=0':>6s} {'c(dr,dLm)':>9s} {'c(dr,db)':>8s} {'c(dr,dd)':>8s}")
    for a in rows:
        print(f"{a['name']:22s} {a['pct_bneg']:5.1f} {a['r_bneg']:6.3f} {a['r_bpos']:6.3f} "
              f"{a['c_dLm']:9.2f} {a['c_db']:8.2f} {a['c_dd']:8.2f}")


if __name__ == "__main__":
    main()
