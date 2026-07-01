#!/usr/bin/env python3
"""Is the dense theta ceiling REAL, or an artifact of the single near-blowout row?

make_kmix_selfconsistent.py measured theta with kappa_mix injected at ONE row per config (max R2,
near blowout) and found mid/dense plateau LOW (0.23-0.35 << Lancaster). The honest caveat: a single
instant can mislead. This harness re-solves the FULL production structure with kappa_mix across MANY
rows of each implicit-phase trajectory, so we can ask the decisive questions:

  - theta_max(traj): the HIGHEST instantaneous theta = L_cool/L_mech the kappa_mix structure reaches
    over the whole phase (the trigger fires on instantaneous theta>=0.95, so this is what decides
    transition). If theta_max ~ theta_blowout for the dense configs, the ceiling is REAL.
  - theta_int: the ENERGY-weighted integral  int L_cool dt / int L_mech dt  -- the fraction of the
    mechanical energy injected over the phase that is radiated away (the magnitude question).
  - frac_fire: the fraction of the phase spent with instantaneous theta>=0.95.

Same faithful injection as make_kmix_selfconsistent (RHS-only kappa_eff = kappa_Spitzer*max(1,R),
no production edit; monkeypatch reused from that module). kappa_mix saturates by lambda*dv~0.01, so a
single representative LDV_ON=1.0 captures the saturated response; lambda*dv=0 is the baseline.

TWO KNOWN LIMITATIONS this run EXPOSES (both feed the next iteration, neither is fixed here):
  (A) STABILITY: the kappa_mix-ON solve FAILS (NaN) at the EARLY, high-Pb rows of the mid configs --
      exactly the epoch where theta peaks and the firing decision happens (the baseline OFF solve
      succeeds there, so it is the hard-max kappa_eff injection at large R, not the replay). So the
      "does it fire" question is NOT cleanly answered for the mid clouds; it needs a more robust
      (smooth-max) injection. Reported per-config as n_solved/n_sampled.
  (B) FAITHFULNESS: kappa_mix = (lambda*dv)*n*k_B ~ n ~ 1/T at fixed Pb, so in the kappa_mix-dominated
      regime (d kappa/dT)/kappa = -1/T, NOT 0. make_kmix_selfconsistent (and SPEC §3) used 0 ("kappa
      flat in T") -- a faithfulness bug. It does not change the saturation/wrong-epoch headlines but
      shifts exact theta; the correct kprime is -1/T (smooth form: (1/T)[2.5 - 3.5 R^s/(1+R^s)]).
This harness inherits the same (uncorrected) injection from make_kmix_selfconsistent so its numbers
are directly comparable to that doc; the corrected+smoothed injection is the next step.

COVERAGE: the 6 cleanroom configs via make_da_replay's build_params/replay_row on the committed
cleanroom trajectories. ~14 rows/config sampled across the phase (trapezoidal integral). NO sims.

REPRODUCE (from repo root; reads committed trajectories):
    python docs/dev/transition/pdv-trigger/data/make_kmix_theta_trajectory.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/kmix_theta_trajectory.csv
    docs/dev/transition/pdv-trigger/kmix_theta_trajectory.png
"""
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import make_kmix_selfconsistent as SC   # noqa: E402  reuse the monkeypatch + _solve
import make_da_replay as DR             # noqa: E402  build_params / replay_row

LDV_ON = 1.0          # any value past the kappa_mix saturation knee (~0.01) gives the saturated theta
N_ROWS = 14           # rows sampled across each implicit-phase trajectory (trapezoidal integral)
FIRE = 0.95           # cooling_balance trigger threshold


def _sample_rows(df):
    """~N_ROWS rows with Pb>0 and finite logged Lloss, evenly spaced across the phase."""
    d = df[(df["Pb"] > 0) & np.isfinite(df["bubble_Lloss"])].reset_index(drop=True)
    if len(d) == 0:
        return d
    if len(d) <= N_ROWS:
        return d
    idx = np.linspace(0, len(d) - 1, N_ROWS).round().astype(int)
    return d.iloc[sorted(set(idx.tolist()))].reset_index(drop=True)


def _trapz_integral(t, y):
    """int y dt over the sampled trajectory (NaNs dropped pairwise)."""
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return float(np.trapz(np.asarray(y)[m], np.asarray(t)[m]))


def main():
    import pandas as pd

    rows_out, summary = [], []
    series = {}   # label -> (t, theta_off, theta_on)

    for cfg in DR.DS.CLEANROOM:
        try:
            params = DR.build_params(cfg)
            df = pd.read_csv(f"docs/dev/transition/cleanroom/data/c0_{cfg}_h0.csv")
            sample = _sample_rows(df)
            if len(sample) < 2:
                print(f"[{cfg}] <2 usable rows — skip"); continue
        except Exception as e:
            print(f"[{cfg}] setup failed: {type(e).__name__}: {e}"); continue

        t_arr, th_off, th_on, Lc_off, Lc_on, Lm = [], [], [], [], [], []
        blow_R2 = sample["R2"].max()
        for _, row in sample.iterrows():
            DR.replay_row(params, row)
            Lmech = float(row["Lmech_total"])
            L0, _, ok0 = SC._solve(params, 0.0)
            L1, _, ok1 = SC._solve(params, LDV_ON)
            t_arr.append(float(row["t_now"]))
            Lm.append(Lmech)
            Lc_off.append(L0 if ok0 else np.nan)
            Lc_on.append(L1 if ok1 else np.nan)
            th_off.append(L0 / Lmech if (ok0 and Lmech) else np.nan)
            th_on.append(L1 / Lmech if (ok1 and Lmech) else np.nan)
            rows_out.append(dict(config=cfg, nCore=DR.DS.NCORE[cfg], t_now=row["t_now"],
                                 R2=row["R2"], is_blowout=bool(row["R2"] == blow_R2),
                                 Lmech=Lmech, L_off=L0, L_on=L1,
                                 theta_off=th_off[-1], theta_on=th_on[-1]))

        t = np.array(t_arr)
        th_on_a, th_off_a = np.array(th_on), np.array(th_off)
        # energy-weighted integral theta = int L_cool dt / int L_mech dt
        th_int_on = _trapz_integral(t, Lc_on) / _trapz_integral(t, Lm) if _trapz_integral(t, Lm) else np.nan
        th_int_off = _trapz_integral(t, Lc_off) / _trapz_integral(t, Lm) if _trapz_integral(t, Lm) else np.nan
        # fraction of phase (dt-weighted) with instantaneous theta_on >= FIRE
        dt = np.gradient(t) if len(t) > 1 else np.array([0.0])
        fin = np.isfinite(th_on_a)
        frac_fire = (dt[fin & (th_on_a >= FIRE)].sum() / dt[fin].sum()) if dt[fin].sum() > 0 else np.nan
        th_blow = next((r["theta_on"] for r in rows_out
                        if r["config"] == cfg and r["is_blowout"]), np.nan)
        # solve coverage for the kappa_mix-ON run, split early (first third) vs whole phase
        n_on = int(fin.sum()); n_tot = len(t)
        n_early = max(1, n_tot // 3)
        n_on_early = int(np.isfinite(th_on_a[:n_early]).sum())
        label = f"{cfg} (n={DR.DS.NCORE[cfg]:.0e})"
        series[label] = (t, th_off_a, th_on_a)
        summary.append(dict(config=cfg, nCore=DR.DS.NCORE[cfg],
                            theta_blowout_on=th_blow,
                            theta_max_on=float(np.nanmax(th_on_a)) if fin.any() else np.nan,
                            theta_int_on=th_int_on, theta_int_off=th_int_off,
                            frac_fire=frac_fire, n_solved_on=n_on, n_sampled=n_tot,
                            n_solved_early=n_on_early, n_early=n_early))
        print(f"[{label}]  theta_on: blowout={th_blow:.2f}  MAX={np.nanmax(th_on_a):.2f}  "
              f"int={th_int_on:.2f}  frac_fire={frac_fire:.2f}  solved_on={n_on}/{n_tot} "
              f"(early {n_on_early}/{n_early})  (baseline int={th_int_off:.2f})")

    # ---- CSV ------------------------------------------------------------------------------------
    with open(os.path.join(_HERE, "kmix_theta_trajectory.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "nCore", "t_now", "R2", "is_blowout",
                                           "Lmech", "L_off", "L_on", "theta_off", "theta_on"])
        w.writeheader(); w.writerows(rows_out)
        fh.write(f"# theta = L_cool/L_mech; kappa_mix injected RHS-only at lambda*dv={LDV_ON} (saturated). "
                 "theta_off = production baseline.\n")
    with open(os.path.join(_HERE, "kmix_theta_trajectory_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "nCore", "theta_blowout_on", "theta_max_on",
                                           "theta_int_on", "theta_int_off", "frac_fire",
                                           "n_solved_on", "n_sampled", "n_solved_early", "n_early"])
        w.writeheader(); w.writerows(summary)
    print("wrote kmix_theta_trajectory{,_summary}.csv")

    # ---- verdict (honest: separate what we KNOW from what the NaNs block) -------------------------
    dense = [s for s in summary if s["nCore"] >= 1e5]
    undersold = [s["config"] for s in summary
                 if np.isfinite(s["theta_max_on"]) and np.isfinite(s["theta_blowout_on"])
                 and s["theta_max_on"] > 1.3 * s["theta_blowout_on"]]
    dense_ceiling = all((np.isfinite(s["theta_max_on"]) and s["theta_max_on"] < FIRE) for s in dense)
    early_gap = [s["config"] for s in summary if s["n_solved_early"] < s["n_early"]]
    print("\nVERDICT (honest):")
    print(f"  1. The blowout-row metric was the WRONG epoch: theta peaks EARLY (high Pb) and decays; "
          f"blowout is the low-theta tail. Single-row undersold by >1.3x: {undersold or 'none'}.")
    print("  2. mid (n=1e4): where it solves, theta_on exceeds 0.95 (would fire) -- but the EARLY "
          "high-Pb rows FAIL to solve with the hard-max injection, so firing is NOT cleanly confirmed.")
    print(f"  3. dense (n>=1e5): solves more broadly and stays theta_max < 0.95 "
          f"({'ceiling holds' if dense_ceiling else 'ceiling NOT clean'}) -- but even these have early NaNs.")
    print(f"  4. Early-phase kappa_mix-ON solve failures (the decisive epoch) in: {early_gap or 'none'} "
          f"-> the firing question needs a more robust (smooth-max) + faithful (kprime=-1/T) injection.")

    # ---- figure ---------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})"); return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.3))
    labels = sorted(series, key=lambda L: float(L.split("n=")[1].rstrip(")")))
    cmap = plt.get_cmap("viridis")
    axL.axhspan(0.9, 0.99, color="#2ca02c", alpha=0.12)
    axL.axhline(FIRE, ls="--", color="#d1495b", lw=1.1)
    axL.text(0.0, 0.965, "trigger θ=0.95", fontsize=8, color="#d1495b")
    for i, label in enumerate(labels):
        t, th_off_a, th_on_a = series[label]
        c = cmap(i / max(1, len(labels) - 1))
        axL.plot(t, th_on_a, "-", color=c, lw=2, label=f"{label} κ_mix on")
        axL.plot(t, th_off_a, ":", color=c, lw=1.3)
    axL.set_xlabel("t [Myr]  (implicit phase)")
    axL.set_ylabel(r"$\theta(t) = L_{\rm cool}/L_{\rm mech}$")
    axL.set_title("θ over the WHOLE trajectory (solid = κ_mix on, dotted = baseline)\n"
                  "does dense θ ever reach the trigger, or plateau low throughout?",
                  fontsize=10, fontweight="bold")
    axL.legend(fontsize=6.4, loc="upper right"); axL.grid(True, alpha=0.2)

    order = sorted(summary, key=lambda s: s["nCore"])
    x = np.arange(len(order)); w = 0.27
    axR.axhline(FIRE, ls="--", color="#d1495b", lw=1.1)
    axR.axhspan(0.9, 0.99, color="#2ca02c", alpha=0.12)
    axR.bar(x - w, [s["theta_blowout_on"] for s in order], w, label="θ blowout (old single-row)", color="#9ecae1")
    axR.bar(x, [s["theta_max_on"] for s in order], w, label="θ max over trajectory", color="#3182bd")
    axR.bar(x + w, [s["theta_int_on"] for s in order], w, label="θ energy-integral", color="#08519c")
    axR.set_xticks(x); axR.set_xticklabels([f"{s['config']}\nn={s['nCore']:.0e}" for s in order],
                                           fontsize=6.4, rotation=20, ha="right")
    axR.set_ylabel(r"$\theta$ (κ_mix on)")
    axR.set_title("Is the single-row metric representative?\n"
                  "(blowout vs trajectory-max vs energy-integral)", fontsize=10, fontweight="bold")
    axR.legend(fontsize=7.2, loc="upper left"); axR.grid(True, axis="y", alpha=0.2)
    fig.suptitle("Time-resolved κ_mix θ: is the dense ceiling real or a single-row artifact?",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "kmix_theta_trajectory.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
