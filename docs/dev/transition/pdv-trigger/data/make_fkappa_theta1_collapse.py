#!/usr/bin/env python3
r"""theta1-collapse of the 819-run f_kappa sweep — the law behind cooling_boost_kappa='auto'.

Reads ONLY the committed fkappa_nH_sweep.csv (one row per (mCloud, sfe, nCore) cell; column
f_kappa_fire_measured = smallest swept f_kappa that fired cooling_balance) and answers the two
questions the sweep was built for:

1. DE-CONFLATION VERDICT: is f_kappa_fire a clean function of n_H alone?  NO — at fixed nCore the
   spread across (mCloud, sfe) reaches ~32x (worst at nCore=3e3), so a single-variable f_kappa(n_H)
   production law is REFUTED. This is why the shipped 'auto' mode (trinity/_input/fkappa_auto.py)
   interpolates the full 3-axis measured grid instead of an n_H curve.

2. WHAT COLLAPSES IT: the starting deficit. Fitting the fired cells (f_fire > 1) with
       log10 f_kappa_fire = c + s * log10(0.95 / theta_fk1)
   collapses ALL cells onto one line (corr ~ 0.97, rms ~ 0.12 dex vs ~0.21 dex for the best
   input-space fit): f_kappa_fire ~ (0.95/theta1)^(1/p) with a UNIVERSAL leverage p = 1/s ~ 0.27
   (theta ~ f_kappa^0.27) — the pessimistic developed-epoch exponent of FINDINGS §6, not the
   optimistic 0.63 snapshot estimate.

REPRODUCE:
    python docs/dev/transition/pdv-trigger/data/make_fkappa_theta1_collapse.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/fkappa_theta1_collapse.csv  (per-cell prediction/residual + fit header)
    docs/dev/transition/pdv-trigger/fkappa_theta1_collapse.png
"""

import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_TRIGGER = 0.95


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(_HERE, "fkappa_nH_sweep.csv")))]
    cells = []
    for r in rows:
        try:
            f_fire = float(r["f_kappa_fire_measured"])
            th1 = float(r["theta_fk1"])
        except ValueError:
            continue  # censored cell (never fired <= 64): excluded from the fit
        if not (math.isfinite(f_fire) and math.isfinite(th1)):
            continue
        cells.append((float(r["mCloud"]), float(r["sfe"]), float(r["nCore"]), th1, f_fire))

    # fit on the fired-above-1 cells (f_fire == 1 is left-censored: it fires at the grid floor)
    fit = [(th1, f) for *_, th1, f in cells if f > 1.0]
    xs = [math.log10(_TRIGGER / th1) for th1, _ in fit]
    ys = [math.log10(f) for _, f in fit]
    n = len(fit)
    mx, my = sum(xs) / n, sum(ys) / n
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    c = my - s * mx
    resid = [y - (c + s * x) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(e * e for e in resid) / n)
    r_num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r_den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    corr = r_num / r_den
    print(
        f"log10 f_fire = {c:.3f} + {s:.3f} * log10(0.95/theta1)   "
        f"(n={n}, corr={corr:.3f}, rms={rms:.3f} dex, leverage p=1/s={1/s:.3f})"
    )

    out = os.path.join(_HERE, "fkappa_theta1_collapse.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                f"# log10(f_kappa_fire) = {c:.4f} + {s:.4f}*log10(0.95/theta_fk1); "
                f"n={n} fired cells; corr={corr:.4f}; rms={rms:.4f} dex; leverage p=1/s={1/s:.4f}"
            ]
        )
        w.writerow(
            [
                "mCloud",
                "sfe",
                "nCore",
                "theta_fk1",
                "f_kappa_fire_measured",
                "f_kappa_fire_predicted",
                "resid_dex",
            ]
        )
        for mC, sf, nC, th1, f in cells:
            pred = 10 ** (c + s * math.log10(_TRIGGER / th1))
            w.writerow(
                [mC, sf, nC, th1, f, round(pred, 3), round(math.log10(f) - math.log10(pred), 4)]
            )
    print(f"wrote {out}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        sys.path.insert(0, _HERE)
        from _trinity_style import use_trinity_style

        use_trinity_style()
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    sfes = sorted({sf for _, sf, *_ in cells})
    markers = dict(zip(sfes, ["o", "s", "^"]))
    for sf in sfes:
        sub = [(th1, f, nC) for _, s_, nC, th1, f in cells if s_ == sf]
        x = np.array([_TRIGGER / th1 for th1, _, _ in sub])
        y = np.array([f for _, f, _ in sub])
        nn = np.log10([nC for _, _, nC in sub])
        sc = ax.scatter(
            x,
            y,
            c=nn,
            cmap="plasma",
            vmin=2,
            vmax=5,
            marker=markers[sf],
            s=48,
            edgecolors="k",
            linewidths=0.4,
            label=rf"sfe $={sf:g}$",
        )
    xr = np.logspace(0, np.log10(_TRIGGER / 0.19), 50)
    ax.plot(
        xr,
        10**c * xr**s,
        "--",
        color="0.4",
        lw=1.4,
        label=rf"$f_\kappa^{{\rm fire}} \propto (0.95/\theta_1)^{{{s:.2f}}}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"starting deficit  $0.95/\theta_1$   ($\theta_1$ = loss fraction at $f_\kappa{=}1$)"
    )
    ax.set_ylabel(r"measured $f_\kappa$ to fire cooling\_balance")
    ax.grid(True, which="both", alpha=0.25)
    fig.colorbar(sc, ax=ax, label=r"$\log_{10} n_{\rm core}$ [cm$^{-3}$]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        r"all 57 fired cells collapse on the starting deficit"
        f"\n(corr {corr:.3f}, rms {rms:.2f} dex; universal leverage "
        rf"$\theta \propto f_\kappa^{{{1/s:.2f}}}$)",
        fontsize=10.5,
    )
    fig.tight_layout()
    png = os.path.join(_PDV, "fkappa_theta1_collapse.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
