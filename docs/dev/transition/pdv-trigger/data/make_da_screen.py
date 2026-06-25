#!/usr/bin/env python3
"""OFFLINE go/no-go screen for a Damkohler-VARYING transition target theta_target(Da).

NO simulations. NO solver edits. Pure reads of committed per-step CSVs, reusing the EXACT loaders,
filters, NSTART/EPS_TR and `first_fire` logic of make_closure_test.py. This script extends that
screen from CONSTANT closures (multiplier f, constant theta_target t) to a single DENSITY/TIME-
VARYING target, and asks the one question the constant screen could not answer: can ONE knob setting
fire all six cleanroom configs at blowout, where no single CONSTANT theta_target can?

------------------------------------------------------------------------------------------------
The Da target being screened (closure `theta_da`)
------------------------------------------------------------------------------------------------
  Lloss_eff = max(Lcool + Lleak, theta_target(row) * Lmech)
  theta_target(row) = theta_max * Da / (1 + Da),   Da = C * Da_shape,   Da_shape = (R2 / v2) * Pb

Rationale (recorded here, NOT a calibrated Da):
  Da = t_turb / t_cool,int, with the turbulent/dynamical interface-crossing time t_turb = R2 / v2 and
  the isochoric interface cooling time t_cool,int = (3/2) k_B T_int / (n_int * Lambda(T_int)). Using the
  ideal-gas interface density n_int = Pb / (k_B T_int) gives
      t_cool,int = (3/2) k_B^2 T_int^2 / (Pb * Lambda(T_int)),
  so, AT A FIXED CHARACTERISTIC INTERFACE TEMPERATURE T_int (hence fixed Lambda(T_int)),
      Da = t_turb / t_cool,int  =  (R2 / v2) * Pb * [ 2 Lambda(T_int) / (3 k_B^2 T_int^2) ]
         (proportional to)        (R2 / v2) * Pb.
  The bracket is a constant (it absorbs T_int, Lambda, k_B). We never know that constant offline, so
  we SWEEP it as C.

  ** UNIT-INDEPENDENCE (the reason this offline test is meaningful at all). **
  Because C is swept on a log grid that spans every plausible value of that bracket, the absolute
  units of Da_shape cancel: any choice of T_int / Lambda / k_B / unit-system only rescales Da_shape by
  a constant, which is exactly what C re-absorbs. So this is a UNIT-INDEPENDENT STRUCTURAL test of
  whether the (R2/v2)*Pb COMBINATION has the right SHAPE across configs -- i.e. whether its value at
  each config's blowout is the same number -- NOT a calibrated Damkohler number. If a single C cannot
  line the configs up, no choice of the absorbed constants (no real T_int/Lambda) can either.

  Da_shape is normalized by the MEDIAN Da_shape-at-blowout across the six cleanroom configs so that
  the interesting C is O(1); the log sweep then brackets it on both sides.

------------------------------------------------------------------------------------------------
Trigger mechanics (why theta_target fires the way it does -- read before interpreting results)
------------------------------------------------------------------------------------------------
  With Lloss_eff = max(Lcool, theta*Lmech) and Lleak=0 here, the no-PdV `cb` trigger is
      r = (Lmech - Lloss_eff)/Lmech = min(1 - Lcool/Lmech, 1 - theta_target).
  The resolved loss fraction Lcool/Lmech at blowout is only 0.25 (diffuse) -> 0.70 (dense) for these
  six configs -- it never reaches the 0.95 the EPS_TR=0.05 gate needs. So the fire is driven by the
  FLOOR term: r dips below EPS_TR exactly where theta_target(row) > 1 - EPS_TR = 0.95. The Da target's
  job is therefore to make theta_target(Da) CROSS 0.95 at each config's blowout. That happens at the
  same Da for every config (theta_max*Da/(1+Da)=0.95 => Da* = 0.95/(theta_max-0.95)), so a single
  (C,theta_max) fires all six at blowout IFF Da_shape@blowout is the SAME across configs. The screen
  measures that directly.

EVERY fire-time here is a FROZEN-TRAJECTORY SCREEN, not a prediction (identical caveat to
make_closure_test.py): boosting Lloss_eff in post does not move the Pb->PdV->blowout orbit the code
actually walked. It bounds the knob; it does not forecast the rerun.

Deliverables written by this script (run from repo root):
  docs/dev/transition/pdv-trigger/data/da_screen.csv   -- per-config screen result
  docs/dev/transition/pdv-trigger/da_screen.png        -- Da_shape trajectories + fire alignment

Run:
  python docs/dev/transition/pdv-trigger/data/make_da_screen.py
"""
import glob

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless; no usetex (mathtext only), matching make_theta_density_plot.py
import matplotlib.pyplot as plt

# --- reused verbatim from make_closure_test.py -------------------------------------------------
NSTART = 2          # drop the first 2 startup rows
EPS_TR = 0.05       # trigger threshold
BLIP_WINDOW = 2     # first-fire within the last BLIP_WINDOW rows = end-of-run blip

# --- Da screen grids ---------------------------------------------------------------------------
C_GRID = np.logspace(-2, 2, 25)     # sweep the absorbed-constant C, O(1) after normalization
THETA_MAX_GRID = [0.95, 0.99]       # ceiling on theta_target
THETA_FIRE = 1.0 - EPS_TR           # 0.95: theta_target must exceed this for the floor to fire

# ambient core density per config [cm^-3] (verbatim from make_theta_density_plot.py NCORE).
NCORE = {
    "large_diffuse_lowsfe": 1e2,
    "be_sphere":            1e4,
    "midrange_pl0":         1e4,
    "pl2_steep":            1e5,
    "simple_cluster":       1e5,
    "small_dense_highsfe":  1e6,
}
CLEANROOM = list(NCORE)  # the 6 configs that have a blowout reference

SERIES = {}  # cfg -> frozen frame + blowout info (Da_shape, loss fraction, etc.)


def first_fire(r, t):
    """(fires, t_fire, pos, sustained, birth, end_blip, rmin) for first r < EPS_TR. Verbatim."""
    below = r < EPS_TR
    rmin = round(float(r.min()), 4) if len(r) else np.nan
    if not below.any():
        return False, np.nan, None, False, False, False, rmin
    pos = int(np.argmax(below.to_numpy()))
    sustained = bool(below.iloc[pos:].all())
    birth = pos == 0
    end_blip = bool(pos >= len(r) - BLIP_WINDOW)
    return True, round(float(t.iloc[pos]), 6), pos, sustained, birth, end_blip, rmin


def load_rcloud():
    """rCloud per config from the 1b shadow (rCloud = blowout_R2 / blowout_R2overRc). Verbatim."""
    s = pd.read_csv("docs/dev/transition/pt4/r1shadow/r1_shadow_summary.csv").set_index("config")
    s = s[s["blowout_R2overRc"].notna()]
    return (s["blowout_R2"] / s["blowout_R2overRc"]).to_dict()


def add_config(config, df, c_Lmech, c_Lcool, c_Lleak, c_R2, c_v2, c_Pb, c_Eb, tcol, rcloud=None):
    """Filter + trim (identical to make_closure_test.add_config), compute Da_shape, stash series."""
    d = df.reset_index(drop=True)
    Lleak = d[c_Lleak] if c_Lleak in d else pd.Series(0.0, index=d.index)
    frame = pd.DataFrame({
        "t": d[tcol], "R2": d[c_R2], "v2": d[c_v2], "Pb": d[c_Pb],
        "Lmech": d[c_Lmech], "Lcool": d[c_Lcool], "Lleak": Lleak, "Eb": d[c_Eb],
    })
    frame = frame.iloc[NSTART:].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["t", "R2", "v2", "Pb", "Lmech", "Lcool", "Lleak"])
    frame = frame.reset_index(drop=True)
    # Da_shape = (R2/v2)*Pb. Lossfrac = (Lcool+Lleak)/Lmech (what the resolved cooling already gives).
    frame["Da_shape"] = (frame["R2"] / frame["v2"]) * frame["Pb"]
    frame["lossfrac"] = (frame["Lcool"] + frame["Lleak"]) / frame["Lmech"]
    # BULK red-herring proxy: Da_bulk = t_dyn/t_cool, t_dyn=R2/v2, t_cool=Eb/Lloss  (= 1/F2).
    # This uses the WHOLE-BUBBLE energy Eb, not the interface, and is known to fire ~60x too early.
    Lloss = frame["Lcool"] + frame["Lleak"]
    frame["Da_bulk"] = (frame["R2"] / frame["v2"]) * Lloss / frame["Eb"]

    t_blow = R2_blow = Da_blow = loss_blow = dabulk_blow = np.nan
    bi = None
    if rcloud is not None and (frame["R2"] >= rcloud).any():
        bi = int(np.argmax((frame["R2"] >= rcloud).to_numpy()))
        t_blow = round(float(frame["t"].iloc[bi]), 6)
        R2_blow = round(float(frame["R2"].iloc[bi]), 4)
        Da_blow = float(frame["Da_shape"].iloc[bi])
        loss_blow = float(frame["lossfrac"].iloc[bi])
        dabulk_blow = float(frame["Da_bulk"].iloc[bi])

    SERIES[config] = dict(frame=frame, rcloud=rcloud, bi=bi, t_blow=t_blow, R2_blow=R2_blow,
                          Da_shape_blow=Da_blow, lossfrac_blow=loss_blow, Da_bulk_blow=dabulk_blow,
                          duration=float(frame["t"].iloc[-1] - frame["t"].iloc[0]))


def load_all():
    rcloud = load_rcloud()
    # 6 cleanroom normal configs (hybr h0). Lleak absent -> 0.
    for f in sorted(glob.glob("docs/dev/transition/cleanroom/data/c0_*_h0.csv")):
        cfg = f.split("/c0_")[1].rsplit("_h0", 1)[0]
        df = pd.read_csv(f)
        if "betadelta_converged" in df:
            df = df[df["betadelta_converged"] == True]  # noqa: E712
        df = df[df["Eb"] > 0]
        add_config(cfg, df, "Lmech_total", "bubble_Lloss", "Lleak", "R2", "v2", "Pb", "Eb", "t_now",
                   rcloud=rcloud.get(cfg))
    # heavy/control budget CSVs: shown on the figure for context only (no rCloud -> no blowout anchor).
    for f in sorted(glob.glob("docs/dev/failed-large-clouds/data/budget_*.csv")):
        cfg = f.split("budget_")[1].replace(".csv", "")
        df = pd.read_csv(f)
        add_config(cfg, df, "Lmech", "Lcool", "Lleak", "R2", "v2", "Pb", "Eb", "t",
                   rcloud=rcloud.get(cfg))


def theta_da(Da_shape_norm, C, theta_max):
    """theta_target(row) = theta_max * Da/(1+Da), Da = C * Da_shape_norm."""
    Da = C * Da_shape_norm
    return theta_max * Da / (1.0 + Da)


def screen_one(cfg, C, theta_max, da_norm):
    """Fire result for one config at one (C, theta_max). Returns dict (fire_minus_blowout etc.).

    The cb trigger with the theta_da floor: r = (Lmech - max(Lcool+Lleak, theta*Lmech))/Lmech.
    """
    S = SERIES[cfg]
    fr = S["frame"]
    th = theta_da(fr["Da_shape"] / da_norm, C, theta_max)
    leff = np.maximum(fr["Lcool"] + fr["Lleak"], th * fr["Lmech"])
    r = ((fr["Lmech"] - leff) / fr["Lmech"]).replace([np.inf, -np.inf], np.nan)
    mask = r.notna()
    rr = r[mask].reset_index(drop=True)
    tt = fr["t"][mask].reset_index(drop=True)
    fires, t_fire, pos, sustained, birth, end_blip, rmin = first_fire(rr, tt)
    fmb = (round(t_fire - S["t_blow"], 6)
           if (fires and not np.isnan(S["t_blow"])) else np.nan)
    return dict(fires=fires, t_fire=t_fire, sustained=sustained, birth=birth, end_blip=end_blip,
                fire_minus_blowout=fmb, min_r=rmin,
                theta_at_blow=(float(th.iloc[S["bi"]]) if S["bi"] is not None else np.nan))


def valid_fire(res):
    """A fire we accept: fires, sustained, not a birth-fire, not an end-of-run blip."""
    return res["fires"] and res["sustained"] and not res["birth"] and not res["end_blip"]


def best_setting(da_norm):
    """Search (C, theta_max) for the knob that best fires all 6 cleanroom configs near blowout.

    Lexicographic objective (smaller is better), so the reported 'best' is the most informative
    setting even when no setting is fully valid:
      1. -n_valid          : prefer more VALID (sustained, non-birth, non-blip) fires;
      2. worst |fmb/dur| over valid fires (inf if none) : tighten the valid spread;
      3. -n_fire           : then prefer more configs that fire AT ALL;
      4. worst |fmb/dur| over all fires : tighten the all-fire spread.
    Steps 3-4 break the all-zero-valid degeneracy so the figure/CSV report the closest-aligned knob
    instead of the first grid point."""
    best = None  # (key, n_valid, n_fire, C, theta_max, per_cfg)
    for theta_max in THETA_MAX_GRID:
        for C in C_GRID:
            per = {c: screen_one(c, C, theta_max, da_norm) for c in CLEANROOM}
            valids = {c: valid_fire(per[c]) for c in CLEANROOM}
            n_valid = sum(valids.values())
            n_fire = sum(per[c]["fires"] for c in CLEANROOM)

            def spread(only_valid):
                vals = [abs(per[c]["fire_minus_blowout"]) / SERIES[c]["duration"]
                        for c in CLEANROOM
                        if (valids[c] if only_valid else per[c]["fires"])
                        and not np.isnan(per[c]["fire_minus_blowout"])]
                return max(vals) if vals else np.inf

            key = (-n_valid, spread(True), -n_fire, spread(False))
            if best is None or key < best[0]:
                best = (key, n_valid, n_fire, C, theta_max, per)
    return best


def constant_theta_contrast():
    """For contrast: the constant theta_target each config NEEDS so the cb trigger fires AT its
    blowout. Since r floors at 1-theta, the floor crosses EPS_TR at theta=0.95 for ALL configs
    simultaneously -- BUT the resolved loss fraction differs, so the FIRE TIME differs. We report,
    per config, the smallest constant theta in a fine grid whose fire is nearest blowout, and the
    resulting fire_minus_blowout, to expose that one constant cannot center them all."""
    out = {}
    grid = np.round(np.arange(0.90, 0.991, 0.005), 4)
    for cfg in CLEANROOM:
        S = SERIES[cfg]
        fr = S["frame"]
        best = None
        for th in grid:
            leff = np.maximum(fr["Lcool"] + fr["Lleak"], th * fr["Lmech"])
            r = ((fr["Lmech"] - leff) / fr["Lmech"]).replace([np.inf, -np.inf], np.nan)
            mask = r.notna()
            rr = r[mask].reset_index(drop=True)
            tt = fr["t"][mask].reset_index(drop=True)
            fires, t_fire, pos, sustained, birth, end_blip, _ = first_fire(rr, tt)
            if not fires:
                continue
            fmb = t_fire - S["t_blow"]
            if best is None or abs(fmb) < abs(best[1]):
                best = (th, fmb, sustained)
        out[cfg] = best  # (theta_needed, fmb_at_that_theta, sustained) or None
    return out


def main():
    load_all()

    da_blows = {c: SERIES[c]["Da_shape_blow"] for c in CLEANROOM}
    da_norm = float(np.median(list(da_blows.values())))  # normalize so C is O(1)

    key, n_valid, n_fire, C_best, tmax_best, per = best_setting(da_norm)
    const = constant_theta_contrast()

    # ---- per-config CSV --------------------------------------------------------------------
    rows = []
    for cfg in CLEANROOM:
        S = SERIES[cfg]
        p = per[cfg]
        cth = const[cfg]
        rows.append(dict(
            config=cfg,
            nCore=NCORE[cfg],
            t_blowout=S["t_blow"],
            duration=round(S["duration"], 6),
            Da_shape_blow=S["Da_shape_blow"],
            Da_shape_blow_norm=S["Da_shape_blow"] / da_norm,
            lossfrac_blow=round(S["lossfrac_blow"], 4),
            Da_bulk_blow=S["Da_bulk_blow"],  # red-herring baseline (=1/F2 at blowout)
            # best single (C, theta_max) Da-target result:
            C_best=C_best, theta_max_best=tmax_best,
            theta_target_at_blow=round(p["theta_at_blow"], 4),
            da_fires=p["fires"], da_sustained=p["sustained"], da_birth=p["birth"],
            da_end_blip=p["end_blip"],
            da_fire_minus_blowout=p["fire_minus_blowout"],
            da_fmb_over_duration=(round(p["fire_minus_blowout"] / S["duration"], 4)
                                  if not np.isnan(p["fire_minus_blowout"]) else np.nan),
            # constant-theta contrast:
            const_theta_needed=(cth[0] if cth else np.nan),
            const_fire_minus_blowout=(round(cth[1], 6) if cth else np.nan),
            const_sustained=(cth[2] if cth else np.nan),
        ))
    out = pd.DataFrame(rows)
    dst_csv = "docs/dev/transition/pdv-trigger/data/da_screen.csv"
    out.to_csv(dst_csv, index=False)

    # ---- figure ----------------------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6.0))

    # LEFT: Da_shape trajectories vs time, blowout marked (cleanroom solid, budget context dashed).
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(CLEANROOM)))
    cc = dict(zip(sorted(CLEANROOM, key=lambda c: NCORE[c]), colors))
    for cfg in sorted(CLEANROOM, key=lambda c: NCORE[c]):
        S = SERIES[cfg]
        fr = S["frame"]
        axL.plot(fr["t"], fr["Da_shape"], color=cc[cfg], lw=1.6,
                 label=f"{cfg} (n={NCORE[cfg]:.0e})")
        if S["bi"] is not None:
            axL.scatter([fr["t"].iloc[S["bi"]]], [fr["Da_shape"].iloc[S["bi"]]],
                        color=cc[cfg], s=110, edgecolor="k", linewidth=0.9, zorder=6)
    axL.axhline(np.median(list(da_blows.values())), color="0.4", ls=":", lw=1.2,
                label="median Da_shape@blowout (= norm)")
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlabel("t  [Myr]")
    axL.set_ylabel("Da_shape = (R2/v2) * Pb   [code units]")
    axL.set_title("Da_shape trajectories (markers = blowout)\nNOT vertically aligned -> "
                  "Da_shape@blowout is NOT config-invariant")
    axL.grid(True, which="both", alpha=0.25)
    axL.legend(fontsize=7.0, loc="lower right")

    # RIGHT: fire_minus_blowout across configs -- Da best knob vs best-constant-theta. SCATTER ONLY
    # (independent configs, no connecting line), matching make_theta_density_plot.py.
    xs = np.arange(len(CLEANROOM))
    order = sorted(CLEANROOM, key=lambda c: NCORE[c])
    da_fmb = [per[c]["fire_minus_blowout"] for c in order]
    da_dur = [SERIES[c]["duration"] for c in order]
    da_norm_fmb = [(f / d if not np.isnan(f) else np.nan) for f, d in zip(da_fmb, da_dur)]
    const_fmb = [(const[c][1] / SERIES[c]["duration"] if const[c] else np.nan) for c in order]
    da_valid = [valid_fire(per[c]) for c in order]

    axR.axhline(0.0, color="crimson", ls="--", lw=1.5, label="blowout (target)")
    # small horizontal offset so the Da (blue) and constant-theta (orange) markers do not occlude.
    axR.scatter(xs - 0.12, da_norm_fmb, s=130, color="tab:blue", edgecolor="k", linewidth=0.7,
                zorder=5, label=f"Da target best (C={C_best:.3g}, theta_max={tmax_best})")
    # mark invalid (non-sustained / birth / blip) Da fires with a red ring
    for i, ok in enumerate(da_valid):
        if not ok and not np.isnan(da_norm_fmb[i]):
            axR.scatter([xs[i] - 0.12], [da_norm_fmb[i]], s=260, facecolors="none",
                        edgecolors="red", linewidth=1.6, zorder=4)
    axR.scatter(xs + 0.12, const_fmb, s=130, marker="s", color="tab:orange", edgecolor="k",
                linewidth=0.7, zorder=5, label="best constant theta (per-config, contrast)")
    axR.set_xticks(xs)
    axR.set_xticklabels([f"{c}\nn={NCORE[c]:.0e}" for c in order], fontsize=7.0, rotation=20,
                        ha="right")
    axR.set_ylabel("(t_fire - t_blowout) / run duration")
    axR.set_title("Fire alignment vs blowout (0 = at blowout)\nred ring = Da fire NOT sustained / "
                  "birth / end-blip")
    axR.grid(True, axis="y", alpha=0.3)
    axR.legend(fontsize=8.0, loc="best")

    worst = np.nanmax([abs(v) for v in da_norm_fmb]) if any(
        not np.isnan(v) for v in da_norm_fmb) else np.nan
    verdict = "GO" if n_valid == len(CLEANROOM) else "NO-GO"
    fig.suptitle(
        f"Da-varying theta_target(Da) go/no-go screen  ->  {verdict}   |   best single knob: "
        f"C={C_best:.3g}, theta_max={tmax_best}, valid sustained fires={n_valid}/6 "
        f"(any-fire {n_fire}/6)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    dst_png = "docs/dev/transition/pdv-trigger/da_screen.png"
    fig.savefig(dst_png, dpi=150)

    # ---- console report --------------------------------------------------------------------
    print(f"wrote {dst_csv}  ({len(out)} rows)")
    print(f"wrote {dst_png}\n")
    print(f"Da_shape normalization (median Da_shape@blowout) = {da_norm:.4e}")
    print(f"Da_shape@blowout spread max/min = "
          f"{max(da_blows.values())/min(da_blows.values()):.2f}x\n")
    print(f"BEST single Da knob: C={C_best:.4g}, theta_max={tmax_best}  "
          f"(valid sustained fires {n_valid}/6, configs firing at all {n_fire}/6)\n")
    hdr = (f"{'config':22s} {'nCore':>7s} {'Da@blow/norm':>12s} {'lossfrac':>8s} "
           f"{'th(Da)@blow':>11s} {'da_fmb':>9s} {'/dur':>7s} {'valid':>5s} "
           f"{'const_th':>8s} {'const_fmb':>10s}")
    print(hdr)
    for cfg in order:
        S, p, cth = SERIES[cfg], per[cfg], const[cfg]
        dur = S["duration"]
        fmb = p["fire_minus_blowout"]
        print(f"{cfg:22s} {NCORE[cfg]:7.0e} {S['Da_shape_blow']/da_norm:12.3f} "
              f"{S['lossfrac_blow']:8.3f} {p['theta_at_blow']:11.4f} "
              f"{(fmb if not np.isnan(fmb) else float('nan')):9.4f} "
              f"{(fmb/dur if not np.isnan(fmb) else float('nan')):7.3f} "
              f"{str(valid_fire(p)):>5s} "
              f"{(cth[0] if cth else float('nan')):8.4f} "
              f"{(cth[1] if cth else float('nan')):10.4f}")

    # ---- BULK red-herring baseline: Da_bulk = t_dyn/t_cool, t_cool = Eb/Lloss (= 1/F2) -----------
    # A 'transition' on the bulk proxy would trip when cooling outpaces dynamics, Da_bulk >= 1. Show
    # where that happens relative to blowout to expose why the bulk Da is the wrong knob.
    print("\nBULK proxy Da_bulk = (R2/v2)*Lloss/Eb (= 1/F2) -- the known red herring:")
    print(f"  {'config':22s} {'Da_bulk@blow':>12s} {'t(Da_bulk>=1)':>13s} {'t_blow':>8s} "
          f"{'fire-blow/dur':>13s}")
    for cfg in order:
        S = SERIES[cfg]
        fr = S["frame"]
        cross = (fr["Da_bulk"] >= 1.0)
        if cross.any():
            ci = int(np.argmax(cross.to_numpy()))
            t_b1 = float(fr["t"].iloc[ci])
            frac = (t_b1 - S["t_blow"]) / S["duration"]
        else:
            t_b1 = np.nan
            frac = np.nan
        print(f"  {cfg:22s} {S['Da_bulk_blow']:12.3f} {t_b1:13.4f} {S['t_blow']:8.4f} "
              f"{frac:13.3f}")
    print("  Da_bulk>=1 (its natural transition) fires FAR before blowout (fire-blow/dur strongly")
    print("  negative) -- consistent with the prior '1/F2 fires ~60x too early' finding. The bulk")
    print("  energy Eb is not the interface energy, so Da_bulk is not the closure to use.")

    # ---- density scaling: theta/(1-theta) at blowout vs nCore (sqrt(n) El-Badry vs linear-n Da) --
    # The empirical 'theta at blowout' is TRINITY's RESOLVED loss fraction at blowout (lossfrac_blow):
    # that is the actual cooling efficiency the code exhibits, the quantity El-Badry/Lancaster predict
    # to scale with density. (The constant-theta-NEEDED column is ~0.95 for every config -- an artifact
    # of the floor mechanism, NOT a density signal -- so it cannot test the scaling and is not used.)
    print("\nDensity scaling of the RESOLVED loss fraction at blowout (theta_emp = lossfrac_blow):")
    print("  test theta/(1-theta) ~ n^p  (El-Badry sqrt(n) => p~0.5 ; Da ansatz linear-n => p~1.0).")
    ns, ratios = [], []
    for cfg in order:
        th = SERIES[cfg]["lossfrac_blow"]
        ratios.append(th / (1.0 - th))
        ns.append(NCORE[cfg])
    ns = np.array(ns, float)
    ratios = np.array(ratios, float)
    p_fit = np.polyfit(np.log10(ns), np.log10(ratios), 1)
    print(f"  log-log slope p (theta/(1-theta) ~ n^p) = {p_fit[0]:.3f}")
    for cfg, n, rr in zip(order, ns, ratios):
        print(f"    {cfg:22s} n={n:.0e}  theta_emp={SERIES[cfg]['lossfrac_blow']:.3f}  "
              f"theta/(1-theta)={rr:.3f}")
    decades = np.log10(ns.max() / ns.min())
    print(f"  Observed rise: theta/(1-theta) {ratios[np.argmin(ns)]:.2f} -> {ratios[np.argmax(ns)]:.2f}"
          f" over {decades:.0f} decades of n (factor {ratios[np.argmax(ns)]/ratios[np.argmin(ns)]:.1f}).")
    print(f"  sqrt(n) would predict factor {10**(0.5*decades):.0f}; linear-n factor {10**decades:.0f}.")
    print(f"  -> the data rises SLOWER than even sqrt(n) (p~{p_fit[0]:.2f}, far below 0.5): it prefers")
    print("     NEITHER the El-Badry sqrt(n) NOR the linear-n Da ansatz; but see CAVEAT (cannot decide).")
    print("  CAVEAT: 6 points, and nCore is degenerate (be_sphere & midrange_pl0 both 1e4; pl2_steep")
    print("  & simple_cluster both 1e5) with 2 confounded SFEs -- a single slope cannot cleanly")
    print("  separate sqrt(n) from linear-n; read p only as 'closer to 0.5 or 1.0'.")

    # ---- VERDICT ----------------------------------------------------------------------------
    GATE = 0.15  # 'tight window': |fire - blowout| must be < 15% of each run's duration
    go = (n_valid == len(CLEANROOM)) and all(
        valid_fire(per[c]) and abs(per[c]["fire_minus_blowout"]) < GATE * SERIES[c]["duration"]
        for c in CLEANROOM)
    print("\n" + "=" * 78)
    print(f"VERDICT: {'GO' if go else 'NO-GO'}  "
          f"(valid sustained fires {n_valid}/6 at the best single knob)")
    if not go:
        print("  Reason: no single (C, theta_max) fires all 6 cleanroom configs sustained near blowout.")
        print(f"   - Da_shape@blowout is NON-MONOTONIC in nCore and spans {max(da_blows.values())/min(da_blows.values()):.1f}x")
        print("     (pl2_steep n=1e5 has the LOWEST Da_shape, below large_diffuse n=1e2), so the")
        print("     theta_max*Da/(1+Da) crossing of 0.95 cannot coincide with blowout across configs.")
        print("   - Da_shape is large EARLY (high Pb at small R2): any C high enough to fire the diffuse")
        print("     configs fires the dense ones at BIRTH, long before blowout. No knob centers all six.")
    print("=" * 78)
    return out


if __name__ == "__main__":
    main()
