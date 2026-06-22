#!/usr/bin/env python3
"""Frozen-feedback experiment (user Q, 2026-06-21): "what if we continuously inject
the SAME feedback? freeze it to its t=1 Myr value and run 6 Myr -- does the cooling
ratio finally drop to 0.05, or do the WR/SN surges only mask an otherwise-falling
ratio?"

The real runs stall: the cooling ratio (Lgain-Lloss)/Lgain plateaus ~0.5-0.85 and the
feedback surges (WR bump, SN onset ~3 Myr) reset it upward. This experiment removes the
surges entirely -- ALL stellar feedback is held CONSTANT at its t=1.0 Myr value for the
whole run (c0_consistency.py --freeze-feedback-at 1.0). If the plateau is surge-sustained,
steady feedback should let the ratio fall through 0.05. If it is geometric (the §7.8
blowout mechanism -- shell exits the cloud, ram-brake collapses, Pb and Lloss collapse,
ratio RISES), steady feedback will NOT help: the ratio still cannot reach 0.05.

Two panels, all six configs:
  TOP  cooling ratio (Lgain-Lloss)/Lgain -- FROZEN (solid, bold) vs REAL (dashed, faint),
       with the 0.05 threshold and each FROZEN run's own blowout epoch (vertical, config
       colour, dash-dot). THE answer panel.
  BOT  Lmech_total (= Lgain) -- proves the freeze: frozen is a flat horizontal line at the
       t=1.0 value; real surges (WR bump + SN onset). If the top panel's frozen curve does
       not drop, it is NOT because feedback fell.

Pure read of committed data/c0_*_h0.csv (REAL) + data/c0_*_frozen.csv (FROZEN). rCloud is
reconstructed from each .param via blowout_marker.rcloud (feedback-independent), so the
frozen blowout is computed from the frozen run's OWN R2 column.

    python plot_frozen_feedback.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import blowout_marker as bm

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"  # parents[3]=repo root
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False

WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]
CONFIGS = ["small_dense_highsfe", "pl2_steep", "simple_cluster",
           "midrange_pl0", "be_sphere", "large_diffuse_lowsfe"]
THRESHOLD = 0.05
FREEZE_T = 1.0  # Myr -- where feedback was frozen


def _f(row, key):
    try:
        v = float(row[key])
    except (ValueError, TypeError, KeyError):
        return None
    return v if math.isfinite(v) else None


def load(path):
    """Per-row t, cooling ratio, Lgain, R2 (finite only)."""
    t, ratio, lg, r2 = [], [], [], []
    if not Path(path).exists():
        return t, ratio, lg, r2
    for row in csv.DictReader(open(path)):
        tn = _f(row, "t_now")
        if tn is None or tn <= 0:
            continue
        lgain, lloss = _f(row, "bubble_Lgain"), _f(row, "bubble_Lloss")
        t.append(tn)
        ratio.append((lgain - lloss) / lgain if (lgain and lloss is not None) else None)
        lg.append(lgain)
        r2.append(_f(row, "R2"))
    return t, ratio, lg, r2


def frozen_blowout(cfg, t, r2):
    """First t where R2 > rCloud in the FROZEN run (rCloud is feedback-independent)."""
    rC = bm.rcloud(cfg)
    if rC is None or rC != rC:
        return None
    for ti, ri in zip(t, r2):
        if ri is not None and ri > rC:
            return ti
    return None


def _clean(xs, ys):
    a, b = [], []
    for x, y in zip(xs, ys):
        if y is not None:
            a.append(x); b.append(y)
    return a, b


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8.2, 7.6))
    summary = []
    for i, cfg in enumerate(CONFIGS):
        c = WONG[i % len(WONG)]
        tr, rr, lgr, _ = load(HERE / "data" / f"c0_{cfg}_h0.csv")        # REAL
        tf, rf, lgf, r2f = load(HERE / "data" / f"c0_{cfg}_frozen.csv")  # FROZEN
        short = cfg.split("_")[0]

        # TOP: cooling ratio, frozen solid+bold, real dashed+faint
        if tr:
            x, y = _clean(tr, rr)
            ax1.plot(x, y, color=c, lw=1.0, ls="--", alpha=0.35)
        if tf:
            x, y = _clean(tf, rf)
            ax1.plot(x, y, color=c, lw=1.9, ls="-", label=short)

        # BOT: Lmech_total (=Lgain) -- proves the freeze
        if tr:
            x, y = _clean(tr, lgr)
            ax2.plot(x, y, color=c, lw=1.0, ls="--", alpha=0.35)
        if tf:
            x, y = _clean(tf, lgf)
            ax2.plot(x, y, color=c, lw=1.9, ls="-")

        # frozen run's OWN blowout epoch (config colour, dash-dot) on both panels
        tb = frozen_blowout(cfg, tf, r2f)
        if tb is not None:
            for ax in (ax1, ax2):
                ax.axvline(tb, color=c, ls=bm.BLOWOUT_LS, lw=1.2, alpha=0.85, zorder=1.4)

        # summary row: does the FROZEN ratio ever reach the threshold?
        rf_fin = [v for v in rf if v is not None]
        rr_fin = [v for v in rr if v is not None]
        crossed = any(v < THRESHOLD for v in rf_fin)
        summary.append((cfg,
                        min(rf_fin) if rf_fin else float("nan"),
                        crossed,
                        tb,
                        min(rr_fin) if rr_fin else float("nan"),
                        tf[-1] if tf else float("nan")))

    ax1.axhline(THRESHOLD, ls=":", lw=1.2, color="0.3", label=f"threshold {THRESHOLD:g}")
    ax1.axvline(FREEZE_T, ls=(0, (1, 1)), lw=1.0, color="0.55")
    ax1.text(FREEZE_T, 0.02, " freeze t", color="0.45", fontsize=7,
             rotation=90, va="bottom", ha="left", transform=ax1.get_xaxis_transform())
    ax1.set_ylabel(r"cooling ratio $(L_{\rm gain}-L_{\rm loss})/L_{\rm gain}$")
    ax1.set_ylim(-0.05, 1.0)
    ax1.set_title("Frozen feedback (solid) vs real (dashed): does steady feedback let the "
                  "ratio reach 0.05?", fontsize=10, pad=8)

    ax2.set_yscale("log")
    ax2.set_ylabel(r"$L_{\rm mech,total}=L_{\rm gain}$  (frozen = flat)")
    ax2.set_xlabel("t  [Myr]")
    ax2.set_xscale("log")
    ax2.set_title(r"Proof of freeze: frozen $L_{\rm mech}$ is constant; real surges "
                  "(WR bump, SN onset ~3 Myr)", fontsize=10, pad=8)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=min(len(labels), 4), framealpha=0.9)
    fig.tight_layout()

    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"frozen_feedback.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {out}/frozen_feedback.(pdf,png)")
    print(f"{'config':24s} {'min_froz_r':>11s} {'<0.05?':>7s} {'froz_blowout':>13s} "
          f"{'min_real_r':>11s} {'froz_t_end':>11s}")
    for cfg, mnf, cr, tb, mnr, tend in summary:
        tbs = f"{tb:.4f}" if tb is not None else "none"
        print(f"{cfg:24s} {mnf:11.4f} {str(cr):>7s} {tbs:>13s} {mnr:11.4f} {tend:11.3f}")


if __name__ == "__main__":
    main()
