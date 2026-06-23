#!/usr/bin/env python3
"""Regenerate pdv_combined_trigger.csv — the combined-ratio ("reading B") trigger test.

Offline diagnostic, NO simulations. Pure reads of committed per-step CSVs:
  ../../cleanroom/data/c0_*_h0.csv            (6 normal configs, hybr h0 baseline)
  ../../../failed-large-clouds/data/budget_*.csv  (fail_repro = heavy 5e9; small_1e6 = control)

Tests whether the maintainer's "reading B" trigger
    r_comb = (Lmech - Lloss - PdV)/Lmech < 0.05
fires meaningfully differently from the shipped cooling_balance trigger
    r_rad  = (Lmech - Lloss)/Lmech < 0.05
where  PdV = 4*pi*R2^2*v2*Pb  (trinity code units, same term as get_betadelta.py:434
Edot_from_balance = Lmech - Lloss - 4*pi*R2^2*v2*Pb).

Conventions reused from make_pdv_regime_table.py: drop the first nstart=2 startup rows;
filter cleanroom rows on betadelta_converged==True (if present) and Eb>0; budget Lloss =
Lcool + Lleak; ratios cleaned with .replace([inf,-inf], nan).dropna().

Run from the repo root:
  python docs/dev/transition/pdv-trigger/data/make_combined_trigger_table.py
"""
import glob

import numpy as np
import pandas as pd

NSTART = 2
THRESH = 0.05  # legacy cooling_balance threshold
EBPEAK_THRESH = 0.0  # ebpeak / net-energy-turnover threshold (reading A)
BLIP_WINDOW = 2  # first-fire within the last BLIP_WINDOW rows = end-of-run blip

ROWS = []
# per-config series kept for the figure: cfg -> (t, r_rad, r_comb)
SERIES = {}


def first_fire(ratio, t, thresh):
    """Return (row, t, sustained, end_blip) for the first ratio < thresh.

    row/t are the *positional* index into the filtered+trimmed series and its t.
    sustained = stays < thresh for every subsequent row (vs a transient dip).
    end_blip  = first-fire is within the last BLIP_WINDOW rows of the run.
    Returns Nones if it never fires.
    """
    below = ratio < thresh
    if not below.any():
        return None, None, None, None
    pos = int(np.argmax(below.to_numpy()))  # first True position
    sustained = bool(below.iloc[pos:].all())
    end_blip = bool(pos >= len(ratio) - BLIP_WINDOW)
    return pos, round(float(t.iloc[pos]), 6), sustained, end_blip


def push(config, regime, df, Lmech, Lloss, R2, v2, Pb, Eb, tcol):
    d = df.reset_index(drop=True)
    PdV = 4 * np.pi * d[R2] ** 2 * d[v2] * d[Pb]
    lm = d[Lmech]
    lloss = d[Lloss]

    pdv_ratio = (PdV / lm).replace([np.inf, -np.inf], np.nan)
    lloss_ratio = (lloss / lm).replace([np.inf, -np.inf], np.nan)
    r_rad = ((lm - lloss) / lm).replace([np.inf, -np.inf], np.nan)
    r_comb = ((lm - lloss - PdV) / lm).replace([np.inf, -np.inf], np.nan)

    frame = pd.DataFrame({
        "t": d[tcol], "r_rad": r_rad, "r_comb": r_comb,
        "pdv_ratio": pdv_ratio, "lloss_ratio": lloss_ratio,
    }).iloc[NSTART:].dropna().reset_index(drop=True)

    n = len(frame)
    SERIES[config] = (frame["t"], frame["r_rad"], frame["r_comb"])

    rad_row, rad_t, rad_sus, rad_blip = first_fire(frame["r_rad"], frame["t"], THRESH)
    comb_row, comb_t, comb_sus, comb_blip = first_fire(frame["r_comb"], frame["t"], THRESH)
    eb_row, eb_t, eb_sus, eb_blip = first_fire(frame["r_comb"], frame["t"], EBPEAK_THRESH)

    ROWS.append(dict(
        config=config, regime=regime, n_rows=n,
        # r_rad < 0.05
        rrad_fire_row=rad_row, rrad_fire_t=rad_t,
        rrad_sustained=rad_sus, rrad_end_blip=rad_blip,
        # r_comb < 0.05  (reading B)
        rcomb_fire_row=comb_row, rcomb_fire_t=comb_t,
        rcomb_sustained=comb_sus, rcomb_end_blip=comb_blip,
        # r_comb <= 0  (ebpeak / reading A)
        rcomb0_fire_row=eb_row, rcomb0_fire_t=eb_t,
        rcomb0_sustained=eb_sus, rcomb0_end_blip=eb_blip,
        # how close each gets
        min_r_comb=round(float(frame["r_comb"].min()), 4),
        min_r_rad=round(float(frame["r_rad"].min()), 4),
        # medians
        med_r_comb=round(float(frame["r_comb"].median()), 4),
        med_r_rad=round(float(frame["r_rad"].median()), 4),
        med_pdv_over_Lmech=round(float(frame["pdv_ratio"].median()), 4),
        med_Lloss_over_Lmech=round(float(frame["lloss_ratio"].median()), 4),
    ))


def make_figure(dst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = ["simple_cluster", "large_diffuse_lowsfe", "fail_repro"]
    reps = [c for c in reps if c in SERIES]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cfg in enumerate(reps):
        t, r_rad, r_comb = SERIES[cfg]
        c = colors[i % len(colors)]
        ax.plot(t, r_rad, "-", color=c, label=f"{cfg}  r_rad")
        ax.plot(t, r_comb, "--", color=c, label=f"{cfg}  r_comb")
    ax.axhline(0.05, color="k", lw=0.8, ls=":", label="threshold 0.05")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":", label="threshold 0 (ebpeak)")
    # Clip to the physical ratio band: the fail_repro post-collapse final row has a
    # negative Pb/Eb (already-collapsed bubble) that sends r_comb to ~4e11 — off-axis.
    ax.set_ylim(-0.8, 1.05)
    ax.set_xlabel("t_now  [Myr]")
    ax.set_ylabel("ratio  (y clipped to [-0.8, 1.05];\nfail_repro collapse spike off-axis)")
    ax.set_title("Combined-ratio transition trigger (offline reconstruction)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(dst, dpi=130)
    plt.close(fig)


def main():
    for f in sorted(glob.glob("docs/dev/transition/cleanroom/data/c0_*_h0.csv")):
        cfg = f.split("/c0_")[1].rsplit("_h0", 1)[0]
        df = pd.read_csv(f)
        if "betadelta_converged" in df:
            df = df[df["betadelta_converged"] == True]  # noqa: E712
        df = df[df["Eb"] > 0]
        push(cfg, "normal", df, "Lmech_total", "bubble_Lloss", "R2", "v2", "Pb", "Eb", "t_now")

    for f in sorted(glob.glob("docs/dev/failed-large-clouds/data/budget_*.csv")):
        cfg = f.split("budget_")[1].replace(".csv", "")
        regime = "heavy_5e9" if "fail" in cfg else "normal_ctrl"
        df = pd.read_csv(f)
        df["Lloss_tot"] = df["Lcool"] + df.get("Lleak", 0)
        push(cfg, regime, df, "Lmech", "Lloss_tot", "R2", "v2", "Pb", "Eb", "t")

    cols = [
        "config", "regime", "n_rows",
        "rrad_fire_row", "rrad_fire_t", "rrad_sustained", "rrad_end_blip",
        "rcomb_fire_row", "rcomb_fire_t", "rcomb_sustained", "rcomb_end_blip",
        "rcomb0_fire_row", "rcomb0_fire_t", "rcomb0_sustained", "rcomb0_end_blip",
        "min_r_comb", "min_r_rad",
        "med_r_comb", "med_r_rad", "med_pdv_over_Lmech", "med_Lloss_over_Lmech",
    ]
    out = pd.DataFrame(ROWS)[cols]
    dst = "docs/dev/transition/pdv-trigger/data/pdv_combined_trigger.csv"
    out.to_csv(dst, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print(out.to_string(index=False))
    print(f"\nwrote {dst}")

    figdst = "docs/dev/transition/pdv-trigger/pdv_combined_trigger.png"
    make_figure(figdst)
    print(f"wrote {figdst}")


if __name__ == "__main__":
    main()
