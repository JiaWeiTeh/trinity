#!/usr/bin/env python3
"""Does PdV alone trigger the energy->momentum transition? — the ebpeak ACTUAL-code-path test.

THE QUESTION (user: "what if we include also PdV?"). TRINITY's default transition trigger is
`cooling_balance`: fire when radiative Lloss/Lgain >= 0.95. TRINITY's 1D radiative cooling is weak,
so that needs a big f_kappa boost (compact ~4x, diffuse ~60x). The `ebpeak` trigger instead watches
the PdV-INCLUSIVE balance: fire when edot_balance = Lgain - Lloss - PdV <= 0, i.e. (Lloss+PdV)/Lgain
>= 1 -- the bubble's net energy stops growing, so it rolls into momentum NATURALLY. PdV (=4*pi*R2^2*
v2*Pb, the work the bubble does on the shell) is the DOMINANT sink here, so the PdV-inclusive ratio
is already ~0.6-0.9 at f_kappa=1. Does that tip the transition WITHOUT a cooling boost?

WHAT THIS HARNESS DOES. Reads the per-segment shadow CSV (shadow_R1_1b.csv, written for EVERY run --
it logs edot_balance, Lgain, Lloss every segment regardless of which trigger is active) from the
calibration runs in outputs/kcal/, plus the two dedicated ebpeak runs (transition_trigger=
cooling_balance,ebpeak) that test the ACTUAL code path (the run self-terminates when ebpeak fires).
It reconstructs PdV = Lgain - Lloss - edot_balance per segment and reports, per run:
  - ebpeak firing time (first segment edot_balance<=0), or NEVER within the run;
  - the PdV-inclusive ratio (Lloss+PdV)/Lgain at cloud dispersal and at its peak;
  - the radiative-only ratio Lloss/Lgain at the same points (what cooling_balance sees);
  - PdV/Lgain (the dominant-sink fraction).

THE FINDING (MEASURED on the actual code path; see docs/dev/transition/pdv-trigger/PLAN.md ledger).
  * ebpeak does NOT fire at f_kappa=1 for EITHER config. The two dedicated ebpeak-ACTIVE runs
    (transition_trigger=cooling_balance,ebpeak) both ran to stop_t and ended on STOPPING_TIME with
    shadow ebpeak_t=None -- so the PdV-inclusive trigger never fired:
      - COMPACT: PdV-incl ratio peaks ~0.91 at t~0.12 (just after dispersal), then DECLINES.
      - DIFFUSE: peaks ~0.86 at t~1.06 (well past dispersal), then DECLINES as the bubble
        RE-ACCELERATES in the low-density ISM (R2->191 pc, v2->168 km/s, Eb still GROWING at t=1.5).
    (This CORRECTS an earlier linear extrapolation that wrongly predicted diffuse would fire ~1.2-1.3;
    the ratio is not monotone -- it turns over because in the deep ISM both sinks shrink vs Lmech.)
  * PdV is the dominant sink (PdV/Lgain ~ 0.20 compact, 0.46 diffuse) and lifts the transition balance
    from radiative-only (Lloss/Lgain ~ 0.66 compact / 0.17 diffuse) up to ~0.86-0.91 -- much closer to
    the 1.0 ebpeak threshold, but it peaks SHORT of it. So including PdV NARROWS the gap; it does not
    close it. A cooling boost (kappa_eff) is still required to actually trigger the transition.
  * The cooling<->PdV TRADE-OFF caps the PdV path, especially for diffuse: boosting f_kappa drains
    Eb -> lowers Pb -> lowers PdV, so the PdV-INCLUSIVE peak is nearly f_kappa-INSENSITIVE for diffuse
    (0.848 -> 0.849 -> 0.853 across f_kappa 1,2,4 -- flat), while the RADIATIVE ratio nearly doubles
    (0.165 -> 0.297). => for diffuse, the only path to fire is radiative cooling_balance (f_kappa~60),
    NOT the PdV-inclusive ebpeak. PdV helps the COMPACT case (fires by f_kappa~2-4), not the diffuse one.
HONEST FRAMING. ebpeak addresses WHEN the bubble transitions, kappa_eff WHETHER it is efficiently
COOLED to the observed theta~0.9 (Lancaster/El-Badry/Gronke). They remain complementary, but this run
DOWNGRADES the optimistic "PdV alone fixes the diffuse f_kappa~60 problem": PdV is a genuine assist
(raises the diffuse floor 0.17->0.85) but NOT a substitute for the cooling boost.

REPRODUCE (from repo root; needs the cal runs + the two ebpeak runs in outputs/kcal/):
    python docs/dev/transition/pdv-trigger/runs/params/  # see cal_*__k{1,2,4}.param + cal_*__ebpeak.param
    python docs/dev/transition/pdv-trigger/data/make_ebpeak_trigger_test.py
Deliverables (committed -- outputs/ is gitignored, so the DERIVED summary is the durable artifact):
    docs/dev/transition/pdv-trigger/data/ebpeak_trigger_test.csv
    docs/dev/transition/pdv-trigger/ebpeak_trigger_test.png
"""

import csv
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_REPO = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 5)))
_OUT = os.path.join(_REPO, "outputs", "kcal")

# (label, run_dir, config). The k1/k2/k4 are cooling_balance-only (shadow logs ebpeak passively);
# the __ebpeak / __ek{1,2,4} runs have transition_trigger=cooling_balance,ebpeak (ACTUAL code path).
# compact=simple_cluster, diffuse=f1edge_lowdens (2 density edges); mid=midrange_pl0, dense=
# small_dense_highsfe extend the f_κ trade-off test to 2 more of the 8-config universe (2026-06-28).
_RUNS = [
    ("compact f_κ=1", "cal_compact__k1", "compact"),
    ("compact f_κ=2", "cal_compact__k2", "compact"),
    ("compact f_κ=4", "cal_compact__k4", "compact"),
    ("compact f_κ=1 [ebpeak ACTIVE]", "cal_compact__ebpeak", "compact"),
    ("diffuse f_κ=1", "cal_diffuse__k1", "diffuse"),
    ("diffuse f_κ=2", "cal_diffuse__k2", "diffuse"),
    ("diffuse f_κ=4", "cal_diffuse__k4", "diffuse"),
    ("diffuse f_κ=1 [ebpeak ACTIVE]", "cal_diffuse__ebpeak", "diffuse"),
    ("mid f_κ=1", "cal_mid__ek1", "mid"),
    ("mid f_κ=2", "cal_mid__ek2", "mid"),
    ("mid f_κ=4", "cal_mid__ek4", "mid"),
    ("dense f_κ=1", "cal_dense__ek1", "dense"),
    ("dense f_κ=2", "cal_dense__ek2", "dense"),
    ("dense f_κ=4", "cal_dense__ek4", "dense"),
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def harvest(run_dir):
    p = os.path.join(_OUT, run_dir, "shadow_R1_1b.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    t = np.array([_f(r["t_now"]) for r in rows])
    R2 = np.array([_f(r["R2"]) for r in rows])
    rC = _f(rows[0]["rCloud"])
    Lg = np.array([_f(r["Lgain"]) for r in rows])
    Ll = np.array([_f(r["Lloss"]) for r in rows])
    edot = np.array([_f(r["edot_balance"]) for r in rows])
    PdV = Lg - Ll - edot  # edot_balance = Lgain - Lloss - PdV
    with np.errstate(divide="ignore", invalid="ignore"):
        rad = Ll / Lg            # what cooling_balance watches
        pdv_incl = (Ll + PdV) / Lg  # what ebpeak watches (>=1 -> fire)
        pdv_frac = PdV / Lg
    # ebpeak firing: first segment edot<=0 (== pdv_incl>=1)
    fire = np.where(edot <= 0)[0]
    ebpeak_t = float(t[fire[0]]) if len(fire) else None
    # dispersal: first R2>rCloud
    disp = np.where(R2 > rC)[0]
    di = int(disp[0]) if len(disp) else int(np.nanargmax(pdv_incl))
    # active-trigger termination is recorded by the run; here we just note the shadow row count/end
    return dict(
        run=run_dir, t_end=float(t[-1]), n=len(rows), rCloud=rC,
        ebpeak_t=ebpeak_t, disp_t=(float(t[di]) if len(disp) else None),
        rad_disp=float(rad[di]), pdvincl_disp=float(pdv_incl[di]),
        pdvfrac_disp=float(pdv_frac[di]),
        pdvincl_peak=float(np.nanmax(pdv_incl)),
        pdvincl_peak_t=float(t[int(np.nanargmax(pdv_incl))]),
        t=t, rad=rad, pdv_incl=pdv_incl, pdv_frac=pdv_frac, R2=R2,
    )


def main():
    summary = []
    traj = {}
    for label, run_dir, cfg in _RUNS:
        h = harvest(run_dir)
        if h is None:
            print(f"{label:34s}  MISSING ({run_dir})")
            continue
        traj[label] = (cfg, h)
        eb = f"{h['ebpeak_t']:.4f}" if h["ebpeak_t"] is not None else "NEVER"
        disp = f"{h['disp_t']:.4f}" if h["disp_t"] is not None else "n/a"
        print(f"{label:34s}  ebpeak@={eb:>8s}  disp@={disp:>7s}  "
              f"radiative(Lloss/Lg)@disp={h['rad_disp']:.3f}  "
              f"PdV-incl@disp={h['pdvincl_disp']:.3f}  peak={h['pdvincl_peak']:.3f}"
              f"@t={h['pdvincl_peak_t']:.3f}  PdV/Lg@disp={h['pdvfrac_disp']:.3f}")
        summary.append({
            "run": h["run"], "t_end": round(h["t_end"], 4),
            "ebpeak_fired_t": ("" if h["ebpeak_t"] is None else round(h["ebpeak_t"], 4)),
            "dispersal_t": ("" if h["disp_t"] is None else round(h["disp_t"], 4)),
            "radiative_ratio_disp": round(h["rad_disp"], 4),
            "pdv_incl_ratio_disp": round(h["pdvincl_disp"], 4),
            "pdv_incl_ratio_peak": round(h["pdvincl_peak"], 4),
            "pdv_incl_peak_t": round(h["pdvincl_peak_t"], 4),
            "pdv_over_lgain_disp": round(h["pdvfrac_disp"], 4),
        })

    csv_path = os.path.join(_HERE, "ebpeak_trigger_test.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "run", "t_end", "ebpeak_fired_t", "dispersal_t", "radiative_ratio_disp",
            "pdv_incl_ratio_disp", "pdv_incl_ratio_peak", "pdv_incl_peak_t",
            "pdv_over_lgain_disp"])
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return

    # One panel per config that HAS data (so the committed figure is always self-consistent as the
    # live runs land one by one). Plot the PdV-inclusive ratio (ebpeak watches this) and the radiative
    # ratio (cooling_balance watches this) vs time, firing line at 1.0. For the f_κ=1 curve use the
    # LONGEST run available (the ebpeak-ACTIVE run, run to stop_t without firing) so the peak-and-
    # decline turnover is visible; f_κ=2,4 show the cooling<->PdV trade-off (PdV-incl ~flat across f_κ).
    cfg_sub = {"compact": "simple_cluster", "diffuse": "f1edge_lowdens",
               "mid": "midrange_pl0", "dense": "small_dense_highsfe"}
    present = [c for c in ("compact", "diffuse", "mid", "dense")
               if any(cc == c for (cc, _) in traj.values())]
    ncol = 2 if len(present) > 1 else 1
    nrow = (len(present) + ncol - 1) // ncol
    fig, axflat = plt.subplots(nrow, ncol, figsize=(6.75 * ncol, 5.1 * nrow), squeeze=False)
    axlist = [ax for row in axflat for ax in row]
    for extra in axlist[len(present):]:
        extra.axis("off")
    colors = {1: "#1f77b4", 2: "#9467bd", 4: "#d62728"}
    for cfg, ax in zip(present, axlist):
        # f_κ=1 trajectory: prefer the ebpeak-ACTIVE (longest) run so the turnover shows.
        long1 = traj.get(f"{cfg} f_κ=1 [ebpeak ACTIVE]") or traj.get(f"{cfg} f_κ=1")
        for label, (c, h) in traj.items():
            if c != cfg:
                continue
            is_long1 = (h is (long1[1] if long1 else None))
            fk = int(label.split("f_κ=")[1].split()[0])
            if fk == 1 and not is_long1:
                continue  # skip the short f_κ=1 (use the long ebpeak-active one instead)
            if fk == 1:
                ax.plot(h["t"], h["pdv_incl"], color=colors[1], lw=2.1,
                        label="PdV-incl (Lloss+PdV)/Lgain, f_κ=1 (ran to stop_t, never fired)")
                ax.plot(h["t"], h["rad"], color=colors[1], lw=1.3, ls=":",
                        label="radiative Lloss/Lgain, f_κ=1")
                ax.plot(h["t"], h["pdv_frac"], color="#ff7f0e", lw=1.2, ls="-.",
                        label="PdV/Lgain (dominant sink), f_κ=1")
                ip = int(np.nanargmax(h["pdv_incl"]))
                ax.plot(h["t"][ip], h["pdv_incl"][ip], "v", color=colors[1], ms=8)
                ax.text(h["t"][ip], h["pdv_incl"][ip] + 0.02,
                        f"peak {h['pdv_incl'][ip]:.2f}", ha="center", fontsize=8, color=colors[1])
                if h["disp_t"] is not None:
                    ax.axvline(h["disp_t"], color="grey", ls="--", lw=1.0)
                    ax.text(h["disp_t"], 0.03, " cloud\n dispersal", fontsize=7.5,
                            color="grey", va="bottom")
            else:
                ax.plot(h["t"], h["pdv_incl"], color=colors[fk], lw=1.6,
                        label=f"PdV-incl, f_κ={fk}")
        ax.axhline(1.0, color="#2ca02c", lw=1.8, label="ebpeak fires (ratio=1.0)")
        ax.axhline(0.95, color="crimson", ls="--", lw=1.2, label="cooling_balance fires (0.95, radiative)")
        ax.set_title(f"{cfg}  ({cfg_sub.get(cfg, '')})", fontsize=11, fontweight="bold")
        ax.set_xlabel("t  [Myr]")
        ax.set_ylabel("loss / Lgain")
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=7.0, loc="lower right")
    regimes = ", ".join(f"{c} ({cfg_sub.get(c, '')})" for c in present)
    fig.suptitle(f"Include PdV in the trigger? Live full runs — {len(present)} regime(s): {regimes}. PdV is the "
                 "dominant sink, so\nthe PdV-inclusive ratio (solid blue) sits far above radiative-only (dotted) "
                 "— but it PEAKS BELOW 1.0 and declines:\nebpeak does NOT fire at f_κ=1 for any of them. The "
                 "cooling<->PdV trade-off keeps PdV-incl nearly f_κ-flat (f_κ=2,4\ncurves hug f_κ=1), so boosting "
                 "cooling helps the radiative path, not the PdV-inclusive one.",
                 fontsize=9.6, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92 if nrow == 1 else 0.95))
    png = os.path.join(_PDV, "ebpeak_trigger_test.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
