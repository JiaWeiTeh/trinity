#!/usr/bin/env python3
"""Batch 9 -- the P_HII identity and the C3c branch, measured on the PRODUCTION grid.

Batches 0/1 established `P_HII == Pb` on a nine-configuration test matrix. This
script asks the same question of the Paper II survey grid itself (10,560 runs), and
then screens the C3c branch over it. No solver is run.

Why it is possible at all: on the pre-C3c scheme the capped-Stromgren `P_HII` was an
exact algebraic relabelling of the confining pressure, so the stored `F_HII` in
`plots/budget_vs_t.csv` divided by `4 pi R2^2` IS `Pb(t)` -- and `Pb` is exactly what
`get_bubbleParams.get_phii_c3c` reads as `P_conf`. `Qi(t)` is reconstructed from the
bundled SPS table (`Qi` is mass-scaled; linearity in `f_mass` is verified at import).
`P_C3a` is then closed form in quantities already on disk.

Two measurements:

  1. IDENTITY. `F_HII / F_hot` over the hot phases (F_hot = |F_ram| = the ramped
     confining force) and `F_HII / F_wind` over the momentum phase, where the
     confining pressure IS the wind ram pressure. Both should sit at exactly 1.
  2. BRANCH SCREEN. `P_C3a / P_conf` per snapshot, per run, bracketing the absorbed
     fraction between each run's `fAbsIon_final` and 1. Reports how many runs ever
     reach the unconfined branch, when, and on which side of the handover.

CAVEAT, carried from the Batch 5 screen and just as binding here: this evaluates the
candidate pressure ON the delivered trajectory. It answers "what would this pressure
have been", NOT "what would the run have done". Once a run goes unconfined the drive
rises and the trajectory changes, so the numbers below bound neither direction.

R2(t) and F_HII(t) live on independently downsampled grids (64 and 200 points per
run). F_HII is interpolated log-log onto the trajectory grid -- fine to coarse, the
safer direction -- and the screened ratio depends on radius only as R2^(+1/2), so the
interpolation is not load-bearing.

Usage:
    python docs/dev/phii-identity/harness/screen_production_grid.py <repo_root>
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:                                    # trinity's SPS reader wants scipy.interpolate
    import scipy.interpolate            # noqa: F401
except ModuleNotFoundError:             # numpy-backed stand-in; linear, and Qi enters
    import types                        # as a square root on a table denser than the
    _st = types.ModuleType("scipy")     # variation, so the order does not matter here
    _si = types.ModuleType("scipy.interpolate")

    class _Interp1d:
        def __init__(self, x, y, kind="linear", **kw):
            self.x, self.y = np.asarray(x, float), np.asarray(y, float)

        def __call__(self, t):
            return np.interp(t, self.x, self.y)

        def derivative(self):
            return _Interp1d(self.x, np.gradient(self.y, self.x))

    class _RGI:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            raise NotImplementedError("stub: this screen never evaluates that table")

    _si.interp1d = _Interp1d
    _si.RegularGridInterpolator = _RGI
    _si.make_interp_spline = lambda x, y, k=3, **kw: _Interp1d(x, y)
    _st.interpolate = _si
    sys.modules["scipy"], sys.modules["scipy.interpolate"] = _st, _si

FOUR_PI = 4.0 * math.pi
REPO = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(REPO))
from trinity._input.read_param import read_param                      # noqa: E402
from trinity.sps import read_sps                                      # noqa: E402

PLOTS = REPO / "paper" / "II-survey" / "plots"
PARAM = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "docs/dev/phii-identity/harness/screen_grid.param"


def constants(param_path):
    p = read_param(str(param_path))
    A = ((p["mu_convert"].value / p["mu_ion_shell"].value)
         * p["k_B"].value * p["TShell_ion"].value
         * math.sqrt(3.0 / (FOUR_PI * p["chi_e_shell"].value * p["caseB_alpha"].value)))
    f1 = read_sps.get_interpolation(read_sps.read_sps(1.0, p))["fQi"]
    f3 = read_sps.get_interpolation(read_sps.read_sps(3.0, p))["fQi"]
    r = float(f3(1.0)) / float(f1(1.0))
    assert abs(r - 3.0) < 1e-9, f"Qi is not linear in f_mass (got {r})"
    return A, float(p["sps_refmass"].value), f1


def main():
    A, refmass, fQi = constants(PARAM)
    print(f"# A = {A:.6e}   sps_refmass = {refmass:g}")

    summ = pd.read_csv(PLOTS / "summary.csv")
    summ = summ[summ["phii"] == True]                                 # noqa: E712
    meta = summ.set_index("run_name")
    keep = set(meta.index)
    print(f"# P_HII-on runs: {len(meta)}")

    bud = pd.read_csv(PLOTS / "budget_vs_t.csv",
                      usecols=["run_name", "t", "F_HII", "F_hot", "F_wind"])
    bud = bud[bud["run_name"].isin(keep)].join(meta["t_phase_transition"], on="run_name")
    traj = pd.read_csv(PLOTS / "trajectory_points.csv", usecols=["run_name", "t", "R2"])
    traj = traj[traj["run_name"].isin(keep)]

    # ---- 1. identity -----------------------------------------------------
    hot = bud[(bud["F_hot"] > 0) & (bud["F_HII"] > 0)].copy()
    hot["era"] = np.where(hot["t_phase_transition"].isna()
                          | (hot["t"] < hot["t_phase_transition"]),
                          "energy+implicit", "transition")
    print("\n=== IDENTITY  F_HII / F_hot  (F_hot = ramped Pb_eff * 4 pi R2^2)")
    for era, g in hot.groupby("era"):
        x = (g["F_HII"] / g["F_hot"]).to_numpy()
        print(f"  {era:16s} n={len(x):8d}  median {np.median(x):.8f}  "
              f"bit-equal {np.mean(np.abs(x - 1) < 1e-9):.4f}  max {x.max():.4f}")
    mom = bud[(bud["F_hot"] == 0) & (bud["F_wind"] > 0)
              & (bud["t"] >= bud["t_phase_transition"])]
    x = (mom["F_HII"] / mom["F_wind"]).to_numpy()
    print(f"  momentum         n={len(x):8d}  median {np.median(x):.8f}  "
          f"bit-equal {np.mean(np.abs(x - 1) < 1e-9):.4f}")
    print("  => pre-C3c momentum drive P_HII + P_ram = 2 x P_ram")

    # ---- 2. branch screen ------------------------------------------------
    rows, bud_g = [], dict(list(bud.groupby("run_name", sort=False)))
    for run, tr in traj.groupby("run_name", sort=False):
        b = bud_g.get(run)
        if b is None:
            continue
        md = meta.loc[run]
        tt, R2 = tr["t"].to_numpy(float), tr["R2"].to_numpy(float)
        o = np.argsort(tt)
        tt, R2 = tt[o], R2[o]
        ok = (R2 > 0) & (tt > 0)
        tt, R2 = tt[ok], R2[ok]
        bt, bh = b["t"].to_numpy(float), b["F_HII"].to_numpy(float)
        o2 = np.argsort(bt)
        bt, bh = bt[o2], bh[o2]
        good = bh > 0
        if len(tt) < 3 or good.sum() < 3:
            continue
        FH = np.exp(np.interp(np.log(tt), np.log(bt[good]), np.log(bh[good])))
        Pconf = FH / (FOUR_PI * R2 ** 2)
        Qi = float(md["Mstar"]) / refmass * np.asarray(fQi(tt), float)
        fab = md["fAbsIon_final"]
        fab = float(fab) if np.isfinite(fab) and 0 < fab <= 1 else 1.0
        tX = md["t_phase_transition"]
        handed = bool(np.isfinite(tX))
        pre = tt < tX if handed else np.ones_like(tt, bool)
        rec = {"run_name": run, "nCore": md["nCore"], "PISM": md["PISM"],
               "nISM": md["nISM"], "mCloud": md["mCloud"], "sfe": md["sfe"],
               "fate": md["fate"], "handed_off": handed,
               "t_phase_transition": tX, "fAbsIon_final": fab}
        for tag, fa in (("f1", 1.0), ("ffin", fab)):
            P = A * np.sqrt(np.maximum(Qi * fa, 0.0)) * R2 ** -1.5
            r = np.where(Pconf > 0, P / Pconf, np.nan)
            unc = np.isfinite(r) & (r > 1.0)
            rec[f"ratio_max_{tag}"] = np.nanmax(r)
            rec[f"ratio_max_pre_{tag}"] = np.nanmax(r[pre]) if pre.any() else np.nan
            rec[f"ratio_max_post_{tag}"] = np.nanmax(r[~pre]) if (~pre).any() else np.nan
            rec[f"ratio_first_{tag}"] = r[0]
            rec[f"frac_unconf_{tag}"] = float(unc.mean())
            rec[f"t_cross_{tag}"] = float(tt[unc][0]) if unc.any() else np.nan
            rec[f"R2_cross_{tag}"] = float(R2[unc][0]) if unc.any() else np.nan
        rows.append(rec)

    df = pd.DataFrame(rows)
    out = REPO / "docs/dev/phii-identity/data/b9_production_branch_screen.csv"
    df.to_csv(out, index=False)
    print(f"\n=== BRANCH SCREEN  n_runs = {len(df)}   handed off "
          f"{int(df['handed_off'].sum())} ({100 * df['handed_off'].mean():.1f}%)")
    for tag, lab in (("f1", "f_abs = 1 (upper bound on P_C3a)"),
                     ("ffin", "f_abs = fAbsIon_final (lower bound)")):
        ever = df[f"ratio_max_{tag}"] > 1
        h = df[df["handed_off"]]
        print(f"\n--- {lab}")
        print(f"  ever unconfined           : {ever.sum():5d}/{len(df)} "
              f"({100 * ever.mean():.1f}%)")
        print(f"  unconfined BEFORE handoff : "
              f"{(df[f'ratio_max_pre_{tag}'] > 1).sum():5d}/{len(df)}")
        print(f"  unconfined AFTER  handoff : "
              f"{(h[f'ratio_max_post_{tag}'] > 1).sum():5d}/{len(h)}")
        print(f"  ratio_max whole run  med {df[f'ratio_max_{tag}'].median():.4g}  "
              f"p90 {df[f'ratio_max_{tag}'].quantile(.9):.4g}")
        print(f"  ratio_max pre-handoff med {df[f'ratio_max_pre_{tag}'].median():.4g}  "
              f"p90 {df[f'ratio_max_pre_{tag}'].quantile(.9):.4g}")
        print(f"  ratio at FIRST snapshot: frac>1 "
              f"{(df[f'ratio_first_{tag}'] > 1).mean():.4f}  "
              "(0 => not the early-init artifact)")
        s = df[ever]
        if len(s):
            print(f"  t_cross  [Myr] p10/med/p90: {s[f't_cross_{tag}'].quantile(.1):.4g}"
                  f" / {s[f't_cross_{tag}'].median():.4g}"
                  f" / {s[f't_cross_{tag}'].quantile(.9):.4g}")
            print(f"  R2_cross [pc]  p10/med/p90: {s[f'R2_cross_{tag}'].quantile(.1):.4g}"
                  f" / {s[f'R2_cross_{tag}'].median():.4g}"
                  f" / {s[f'R2_cross_{tag}'].quantile(.9):.4g}")
        print("  ever-unconfined by fate:",
              {k: f"{100 * v:.0f}%" for k, v in ever.groupby(df["fate"]).mean().items()})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
