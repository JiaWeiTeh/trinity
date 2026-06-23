#!/usr/bin/env python3
"""Regenerate pdv_combined_trigger.csv — the cooling-ratio trigger test, with vs. without PdV.

Offline diagnostic, NO simulations. Pure reads of committed per-step CSVs:
  ../../cleanroom/data/c0_*_h0.csv            (6 normal configs, hybr h0 baseline)
  ../../../failed-large-clouds/data/budget_*.csv  (fail_repro = heavy 5e9; small_1e6 = control)

Tests whether folding PdV into the energy->momentum trigger changes when it fires. Two ratios
(named so neither looks like a radius):
    cool    = (Lmech - Lloss)/Lmech          # radiative cooling ratio  (the shipped cooling_balance term)
    coolPdV = (Lmech - Lloss - PdV)/Lmech     # radiative cooling ratio WITH PdV  (the maintainer's "reading B")
where  PdV = 4*pi*R2^2*v2*Pb  (trinity code units, same term as get_betadelta.py:434
Edot_from_balance = Lmech - Lloss - 4*pi*R2^2*v2*Pb).  Note coolPdV = cool - PdV/Lmech, so PdV/Lmech
is exactly the offset between the two curves. The trigger fires when the ratio < 0.05.

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
# per-config series kept for the figure: cfg -> (t, cool, coolPdV)
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
    cool = ((lm - lloss) / lm).replace([np.inf, -np.inf], np.nan)          # no PdV
    coolPdV = ((lm - lloss - PdV) / lm).replace([np.inf, -np.inf], np.nan)  # with PdV

    frame = pd.DataFrame({
        "t": d[tcol], "cool": cool, "coolPdV": coolPdV,
        "pdv_ratio": pdv_ratio, "lloss_ratio": lloss_ratio,
    }).iloc[NSTART:].dropna().reset_index(drop=True)

    n = len(frame)
    SERIES[config] = (frame["t"], frame["cool"], frame["coolPdV"])

    cool_row, cool_t, cool_sus, cool_blip = first_fire(frame["cool"], frame["t"], THRESH)
    cp_row, cp_t, cp_sus, cp_blip = first_fire(frame["coolPdV"], frame["t"], THRESH)
    cp0_row, cp0_t, cp0_sus, cp0_blip = first_fire(frame["coolPdV"], frame["t"], EBPEAK_THRESH)

    ROWS.append(dict(
        config=config, regime=regime, n_rows=n,
        # cool < 0.05  (no PdV — the shipped cooling_balance trigger)
        cool_fire_row=cool_row, cool_fire_t=cool_t,
        cool_sustained=cool_sus, cool_end_blip=cool_blip,
        # coolPdV < 0.05  (with PdV — reading B)
        coolPdV_fire_row=cp_row, coolPdV_fire_t=cp_t,
        coolPdV_sustained=cp_sus, coolPdV_end_blip=cp_blip,
        # coolPdV <= 0  (ebpeak / reading A)
        coolPdV0_fire_row=cp0_row, coolPdV0_fire_t=cp0_t,
        coolPdV0_sustained=cp0_sus, coolPdV0_end_blip=cp0_blip,
        # how close each gets
        min_coolPdV=round(float(frame["coolPdV"].min()), 4),
        min_cool=round(float(frame["cool"].min()), 4),
        # medians
        med_coolPdV=round(float(frame["coolPdV"].median()), 4),
        med_cool=round(float(frame["cool"].median()), 4),
        med_pdv_over_Lmech=round(float(frame["pdv_ratio"].median()), 4),
        med_Lloss_over_Lmech=round(float(frame["lloss_ratio"].median()), 4),
    ))


def make_figure(dst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Normal clouds span the behaviour: never-fires -> closest-without-firing -> the one that fires.
    # (fail_repro plunges < 0 immediately and is a different, super-critical regime; with ymin=0 it
    # would just hug the floor, so it is excluded here and discussed in the table / PLAN.)
    reps = ["simple_cluster", "small_dense_highsfe", "large_diffuse_lowsfe"]
    reps = [c for c in reps if c in SERIES]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shade the trigger region: the transition fires when the cooling ratio < 0.05.
    ax.axhspan(0.0, THRESH, color="tab:red", alpha=0.12, zorder=0)
    ax.axhline(THRESH, color="tab:red", lw=0.9, ls="--", zorder=1)

    cfg_handles = []
    for i, cfg in enumerate(reps):
        t, cool, coolPdV = SERIES[cfg]
        c = colors[i % len(colors)]
        ax.plot(t, cool, "-", color=c, lw=1.7, zorder=3)      # no PdV
        ax.plot(t, coolPdV, "--", color=c, lw=1.7, zorder=3)  # with PdV
        cfg_handles.append(Line2D([0], [0], color=c, lw=1.7, label=cfg))

    style_handles = [
        Line2D([0], [0], color="0.25", lw=1.7, ls="-",
               label="no PdV:  (Lmech−Lloss)/Lmech"),
        Line2D([0], [0], color="0.25", lw=1.7, ls="--",
               label="with PdV:  (Lmech−Lloss−PdV)/Lmech"),
        Patch(facecolor="tab:red", alpha=0.12, label="trigger region (ratio < 0.05)"),
    ]

    leg1 = ax.legend(handles=cfg_handles, title="config (colour)", loc="upper right", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, title="trigger term (line style)", loc="lower left", fontsize=9)

    ax.set_ylim(0.0, 1.02)  # ymin capped at 0
    ax.set_xlabel("t  [Myr]")
    ax.set_ylabel("cooling ratio   (net energy gain / Lmech)")
    ax.set_title("Energy→momentum trigger: radiative cooling ratio, with vs. without PdV")
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
        "cool_fire_row", "cool_fire_t", "cool_sustained", "cool_end_blip",
        "coolPdV_fire_row", "coolPdV_fire_t", "coolPdV_sustained", "coolPdV_end_blip",
        "coolPdV0_fire_row", "coolPdV0_fire_t", "coolPdV0_sustained", "coolPdV0_end_blip",
        "min_coolPdV", "min_cool",
        "med_coolPdV", "med_cool", "med_pdv_over_Lmech", "med_Lloss_over_Lmech",
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
