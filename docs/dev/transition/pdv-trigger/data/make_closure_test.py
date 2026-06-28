#!/usr/bin/env python3
"""Offline shadow test of interface-cooling closures for the energy->momentum trigger.

NO simulations. Pure reads of committed per-step CSVs. Builds directly on
make_combined_trigger_table.py: same filters, same `PdV = 4*pi*R2^2*v2*Pb` convention, same
load_rcloud() recovery of rCloud from the 1b shadow, same blowout detection. This script asks a
different question: which *cooling closure* (effective radiative loss fed to the trigger) makes the
energy->momentum transition fire at a physically sensible place -- near blowout (R2 = rCloud) --
across every reconstructable config, WITHOUT touching production.

Definitions (held EXACTLY; do not invent variants):
  Lmech  : mechanical luminosity (cleanroom Lmech_total / budget Lmech)
  Lcool  : resolved radiative cooling (cleanroom bubble_Lloss / budget Lcool)
  Lleak  : separate leak term (cleanroom: ABSENT -> 0; budget Lleak, which is identically 0 here)
  Lloss  : Lcool + Lleak
  PdV    : 4*pi*R2^2*v2*Pb   (trinity code units, same term as get_betadelta.py Edot_from_balance)

Closures -> Lloss_eff (the effective loss the trigger sees):
  none            Lloss_eff = Lloss
  multiplier(f)   Lloss_eff = Lleak + f*Lcool                    f in {1,1.5,2,3,5,10,30}
  theta_target(t) Lloss_eff = max(Lcool+Lleak, t*Lmech)         t in {0.3,0.5,0.7,0.8,0.9,0.95}

Trigger forms (both computed; PRIMARY = the note's no-PdV `cb`):
  cb   r = (Lmech - Lloss_eff)/Lmech                    (the note's cooling-balance trigger)
  pdv  r = (Lmech - Lloss_eff - PdV)/Lmech              (PdV folded in, "combined")
Fires when r < EPS_TR (0.05). Sustained = stays < EPS_TR for all later rows. Birth-fire = first row.

Blowout = first row R2 >= rCloud (rCloud via load_rcloud()) -- the physical reference epoch.

EVERY fire-time here is a FROZEN-TRAJECTORY SCREEN, not a prediction: these CSVs were produced with
production cooling, so boosting Lloss_eff in post does NOT move the actual Pb -> PdV -> blowout that
the boosted run would have taken. The test answers "where would this closure have tripped the gate on
the trajectory the code actually walked", which bounds the knob, it does not forecast the new orbit.

Run from the repo root:
  python docs/dev/transition/pdv-trigger/data/make_closure_test.py
"""
import glob

import numpy as np
import pandas as pd

NSTART = 2          # drop the first 2 startup rows (matches make_combined_trigger_table.py)
EPS_TR = 0.05       # trigger threshold
BLIP_WINDOW = 2     # first-fire within the last BLIP_WINDOW rows = end-of-run blip

F_GRID = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]
THETA_GRID = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

# Per-config frozen series kept for the figures: cfg -> dict with t, R2, Lmech, Lcool, Lleak, PdV,
# regime, rcloud, blowout (t_blow, R2_blow or None).
SERIES = {}
ROWS = []


def first_fire(r, t):
    """(fires, t_fire, R2pos, sustained, birth, end_blip, rmin) for first r < EPS_TR.

    r, t are aligned filtered+trimmed Series (positional index = row order).
    sustained: r stays < EPS_TR for every later row.  birth: first-fire is the first row.
    end_blip: first-fire sits in the last BLIP_WINDOW rows (a likely end-of-run transient).
    """
    below = r < EPS_TR
    rmin = round(float(r.min()), 4) if len(r) else np.nan
    if not below.any():
        return False, np.nan, None, False, False, False, rmin
    pos = int(np.argmax(below.to_numpy()))
    sustained = bool(below.iloc[pos:].all())
    birth = pos == 0
    end_blip = bool(pos >= len(r) - BLIP_WINDOW)
    return True, round(float(t.iloc[pos]), 6), pos, sustained, birth, end_blip, rmin


def lloss_eff(closure, value, Lcool, Lleak, Lmech):
    if closure == "none":
        return Lcool + Lleak
    if closure == "multiplier":
        return Lleak + value * Lcool
    if closure == "theta_target":
        return np.maximum(Lcool + Lleak, value * Lmech)
    raise ValueError(closure)


def add_config(config, regime, df, c_Lmech, c_Lcool, c_Lleak, c_R2, c_v2, c_Pb, c_Eb, tcol,
                rcloud=None):
    """Filter, trim, stash the frozen series, and emit one ROWS entry per (closure,value,form)."""
    d = df.reset_index(drop=True)
    Lleak = d[c_Lleak] if c_Lleak in d else pd.Series(0.0, index=d.index)
    frame = pd.DataFrame({
        "t": d[tcol], "R2": d[c_R2], "v2": d[c_v2], "Pb": d[c_Pb],
        "Lmech": d[c_Lmech], "Lcool": d[c_Lcool], "Lleak": Lleak,
    })
    frame["PdV"] = 4 * np.pi * frame["R2"] ** 2 * frame["v2"] * frame["Pb"]
    # Match make_combined_trigger_table: trim startup, drop non-finite core columns.
    frame = frame.iloc[NSTART:].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["t", "R2", "v2", "Pb", "Lmech", "Lcool", "Lleak"])
    frame = frame.reset_index(drop=True)

    # Blowout = first row R2 >= rCloud. rCloud not logged in these CSVs -> from the 1b shadow.
    t_blow = R2_blow = np.nan
    blow = None
    if rcloud is not None and (frame["R2"] >= rcloud).any():
        bi = int(np.argmax((frame["R2"] >= rcloud).to_numpy()))
        t_blow = round(float(frame["t"].iloc[bi]), 6)
        R2_blow = round(float(frame["R2"].iloc[bi]), 4)
        blow = (frame["t"].iloc[bi], frame["R2"].iloc[bi])

    SERIES[config] = dict(frame=frame, regime=regime, rcloud=rcloud, blowout=blow,
                          t_blow=t_blow, R2_blow=R2_blow)

    lm = frame["Lmech"]
    plans = [("none", None)]
    plans += [("multiplier", f) for f in F_GRID]
    plans += [("theta_target", th) for th in THETA_GRID]

    for closure, value in plans:
        leff = lloss_eff(closure, value, frame["Lcool"], frame["Lleak"], lm)
        r_cb = ((lm - leff) / lm).replace([np.inf, -np.inf], np.nan)
        r_pdv = ((lm - leff - frame["PdV"]) / lm).replace([np.inf, -np.inf], np.nan)
        for form, r in (("cb", r_cb), ("pdv", r_pdv)):
            rr = r.copy()
            tt = frame["t"]
            mask = rr.notna()
            rr, tt2 = rr[mask].reset_index(drop=True), tt[mask].reset_index(drop=True)
            R2m = frame["R2"][mask].reset_index(drop=True)
            fires, t_fire, pos, sustained, birth, end_blip, rmin = first_fire(rr, tt2)
            R2_fire = round(float(R2m.iloc[pos]), 4) if fires else np.nan
            fmb = round(t_fire - t_blow, 6) if (fires and not np.isnan(t_blow)) else np.nan
            ROWS.append(dict(
                config=config, regime=regime, closure=closure,
                value=(np.nan if value is None else value), trigger_form=form,
                fires=fires, t_fire=t_fire, R2_fire=R2_fire,
                sustained=sustained, birth_fire=birth, end_blip=end_blip,
                t_blowout=t_blow, R2_blowout=R2_blow, fire_minus_blowout=fmb,
                min_r=rmin,
            ))


def load_rcloud():
    """rCloud per config from the 1b shadow (rCloud = blowout_R2 / blowout_R2overRc).

    rCloud is a phase-0 run-const the cleanroom per-step CSVs export as all-NaN, so it is recovered
    from r1_shadow_summary.csv, which logged the real blowout R2 and R2/rCloud. (Reused verbatim
    from make_combined_trigger_table.py.)
    """
    s = pd.read_csv("docs/dev/transition/pt4/r1shadow/r1_shadow_summary.csv").set_index("config")
    s = s[s["blowout_R2overRc"].notna()]
    return (s["blowout_R2"] / s["blowout_R2overRc"]).to_dict()


def main():
    rcloud = load_rcloud()

    # 6 normal configs: cleanroom hybr h0 baseline. Lleak absent -> 0 (confirmed; no leak column).
    for f in sorted(glob.glob("docs/dev/transition/cleanroom/data/c0_*_h0.csv")):
        cfg = f.split("/c0_")[1].rsplit("_h0", 1)[0]
        df = pd.read_csv(f)
        if "betadelta_converged" in df:
            df = df[df["betadelta_converged"] == True]  # noqa: E712
        df = df[df["Eb"] > 0]
        add_config(cfg, "normal", df, "Lmech_total", "bubble_Lloss", "Lleak",
                   "R2", "v2", "Pb", "Eb", "t_now", rcloud=rcloud.get(cfg))

    # Heavy 5e9 (fail_repro) + small_1e6 control: budget CSVs. Lleak column exists but is 0 here.
    # No rCloud for these (not in the 1b shadow); fail_repro is super-critical and never reaches
    # rCloud anyway, so blowout is intentionally NaN for them.
    for f in sorted(glob.glob("docs/dev/failed-large-clouds/data/budget_*.csv")):
        cfg = f.split("budget_")[1].replace(".csv", "")
        regime = "heavy_5e9" if "fail" in cfg else "normal_ctrl"
        df = pd.read_csv(f)
        add_config(cfg, regime, df, "Lmech", "Lcool", "Lleak",
                   "R2", "v2", "Pb", "Eb", "t", rcloud=rcloud.get(cfg))

    # fail_helix: NOT offline-reconstructable. Its only per-step trajectories
    # (docs/dev/transition/pt4/traj/h{3,4}_traj_fail_helix_V0.csv) have bubble_Lloss and rCloud
    # entirely NaN -- it collapses in phase 1a before resolved cooling is computed. Needs an
    # in-solver shadow run; recorded here as a stub so the inventory is explicit.
    ROWS.append(dict(
        config="fail_helix", regime="heavy_helix", closure="none", value=np.nan,
        trigger_form="cb", fires=False, t_fire=np.nan, R2_fire=np.nan, sustained=False,
        birth_fire=False, end_blip=False, t_blowout=np.nan, R2_blowout=np.nan,
        fire_minus_blowout=np.nan, min_r=np.nan,
    ))

    cols = ["config", "regime", "closure", "value", "trigger_form", "fires", "t_fire", "R2_fire",
            "sustained", "birth_fire", "end_blip", "t_blowout", "R2_blowout",
            "fire_minus_blowout", "min_r"]
    out = pd.DataFrame(ROWS)[cols]
    dst = "docs/dev/transition/pdv-trigger/data/closure_test.csv"
    out.to_csv(dst, index=False)
    print(f"wrote {dst}  ({len(out)} rows)")
    return out


if __name__ == "__main__":
    main()
